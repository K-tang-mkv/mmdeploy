import argparse
import os
import time

import cv2
import numpy as np
from mmdeploy_runtime import Detector, PoseDetector


def parse_args():
    parser = argparse.ArgumentParser(
        description='show how to use SDK Python API')
    parser.add_argument('device_name', help='name of device, cuda or cpu')
    parser.add_argument(
        'det_person_model_path',
        help='path of mmdeploy SDK model dumped by model converter')
    parser.add_argument(
        'det_shose_model_path',
        help='path of mmdeploy SDK model dumped by model converter')
    parser.add_argument('source', default=None, help='path of input file')
    parser.add_argument('output_dir', default='runs/', help='save result in output dir')
    parser.add_argument('--shose_only', default=False, help="whether only det shose")
    args = parser.parse_args()
    return args


def load_data(filename, output):
    if filename.endswith('.jpg'):
        # support jpg
        img = cv2.imread(filename)
        return [img], None
    elif filename.endswith('.mp4'):
        cap = cv2.VideoCapture(filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(os.path.join(output, os.path.basename(filename)), fourcc, 60.0, (width, height))

        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(total_frames):
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        cap.release()
        return frames, out
    elif os.path.isdir(filename):
        # detect image folder
        imgs = []
        for imgfile in os.listdir(filename):
            if imgfile.endswith('.jpg'):
                img = cv2.imread(os.path.join(filename, imgfile))
                imgs.append(img)
        
        if not imgs:
            print(f'invalid folder, no images in {filename}')
            exit(1)
        return imgs, _
    else:
        print(f'invalid input file{filename}')
        exit(1)


def draw_box(img, pairs_person_shose):
    for pair_person_shose in pairs_person_shose:
        person_bbox, shose_boxes = pair_person_shose
        cv2.rectangle(img, (int(person_bbox[0]), int(person_bbox[1])), (int(person_bbox[2]), int(person_bbox[3])), (0, 255, 0), 2)
        for shose_box in shose_boxes:
            xmin, ymin, xmax, ymax = shose_box[0]+person_bbox[0], shose_box[1]+person_bbox[1], shose_box[2]+person_bbox[0], shose_box[3]+person_bbox[1]
            cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)

    return img


def main():
    args = parse_args()
    source = args.source
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    # load image
    data, out_vid =  load_data(source, output_dir)
    count = 0
    for img in data: 
        if args.shose_only:
            # create shose detector
            shose_detector = Detector(
                model_path=args.det_shose_model_path, device_name=args.device_name)
            start = time.time()
            bboxes, labels, _ = shose_detector(img)
            end = time.time()
            print("time consumed: ", end-start)
            print(bboxes[..., 4])
            keep = np.logical_and(labels == 0, bboxes[..., 4] > 0.09)
            bboxes = bboxes[keep, :4]
            print(len(bboxes))
            for bbox in bboxes:
                print(bbox)
                xmin, ymin, xmax, ymax = bbox
                cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
                
        else:    
            # create object detector
            person_detector = Detector(
                model_path=args.det_person_model_path, device_name=args.device_name)
        # create shose detector
            shose_detector = Detector(
                model_path=args.det_shose_model_path, device_name=args.device_name)

        # apply detector
            start = time.time()
            bboxes, labels, _ = person_detector(img)

        # filter detections
            keep = np.logical_and(labels == 0, bboxes[..., 4] > 0.8)
            bboxes = bboxes[keep, :4]
            print(f'person box: {bboxes} {labels}')

            pairs_person_shose = []
            for bbox in bboxes:
                # detect shose for each person
                person_xmin, person_ymin, person_xmax, person_ymax = [int(x) for x in bbox]
                person_area = img[person_ymin:person_ymax, person_xmin:person_xmax]
                cv2.imwrite(os.path.join(output_dir, str(count)+".jpg"), person_area)
                count+=1
                shose_bboxes, labels, _ = shose_detector(person_area)
                # filter
                shose_keep = np.logical_and(labels == 0, shose_bboxes[..., 4] > 0.2)
                shose_bboxes = shose_bboxes[shose_keep, :4]

                pairs_person_shose.append((bbox, shose_bboxes))
            end = time.time()
            print("time consumed: ", end-start)
            img = draw_box(img, pairs_person_shose)

        if source.endswith('.mp4'):
            out_vid.write(img)
        elif os.path.isdir(source):
            cv2.imwrite(os.path.join(output_dir,str(count)+'.jpg'), img)
        else:
            cv2.imwrite(os.path.join(output_dir, os.path.basename(source)), img)

    if source.endswith('.mp4'):
        out_vid.release()


if __name__ == '__main__':
    main()