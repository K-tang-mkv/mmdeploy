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

    args = parser.parse_args()
    return args

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
    img = cv2.imread(source)

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
    keep = np.logical_and(labels == 0, bboxes[..., 4] > 0.6)
    bboxes = bboxes[keep, :4]

    pairs_person_shose = []
    for bbox in bboxes:
        # detect shose for each person
        person_xmin, person_ymin, person_xmax, person_ymax = [int(x) for x in bbox]
        person_area = img[person_ymin:person_ymax, person_xmin:person_xmax]
    #             cv2.imwrite(output_dir+str(count)+".jpg", person_area)
    
        # 检测鞋子
        shose_bboxes, labels, _ = shose_detector(person_area)
        
        # filter
        shose_keep = np.logical_and(labels == 0, shose_bboxes[..., 4] > 0.1)
        shose_bboxes = shose_bboxes[shose_keep, :4]

        pairs_person_shose.append((bbox, shose_bboxes))
    end = time.time()
    print("time consumed: ", end-start)
    img_with_box = draw_box(img, pairs_person_shose)

    
    cv2.imwrite(os.path.join(output_dir, os.path.basename(source)), img_with_box)


if __name__ == '__main__':
    main()