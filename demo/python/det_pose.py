import argparse
import os
import psutil
import cv2
import numpy as np
from mmdeploy_runtime import Detector, PoseDetector
import matplotlib.pyplot as plt
from threading import Thread

import time
import GPUtil
from tabulate import tabulate

def parse_args():
    parser = argparse.ArgumentParser(
        description='show how to use SDK Python API')
    parser.add_argument('device_name', help='name of device, cuda or cpu')
    parser.add_argument(
        'det_model_path',
        help='path of mmdeploy SDK model dumped by model converter')
    parser.add_argument(
        'pose_model_path',
        help='path of mmdeploy SDK model dumped by model converter')
    parser.add_argument('image_path', help='path of input image')
    args = parser.parse_args()
    return args


def visualize(frame, bboxes, keypoints, filename, thr=0.5, resize=1280):
    skeleton = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11),
                (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2),
                (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]
    palette = [(255, 128, 0), (255, 153, 51), (255, 178, 102), (230, 230, 0),
               (255, 153, 255), (153, 204, 255), (255, 102, 255),
               (255, 51, 255), (102, 178, 255),
               (51, 153, 255), (255, 153, 153), (255, 102, 102), (255, 51, 51),
               (153, 255, 153), (102, 255, 102), (51, 255, 51), (0, 255, 0),
               (0, 0, 255), (255, 0, 0), (255, 255, 255)]
    link_color = [
        0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16
    ]
    point_color = [16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0]

    scale = 1

    scores = keypoints[..., 2]
    keypoints = (keypoints[..., :2] * scale).astype(int)

    img = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    
    for kpts, score in zip(keypoints, scores):
        show = [0] * len(kpts)
        for (u, v), color in zip(skeleton, link_color):
            if score[u] > thr and score[v] > thr:
                cv2.line(img, kpts[u], tuple(kpts[v]), palette[color], 1,
                         cv2.LINE_AA)
                show[u] = show[v] = 1
        for kpt, show, color in zip(kpts, show, point_color):
            if show:
                cv2.circle(img, kpt, 1, palette[color], 2, cv2.LINE_AA)
    cv2.imwrite(filename, img)
    return img

def load_data(filename, output):
    cap = cv2.VideoCapture(filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(width, height)
    out = cv2.VideoWriter(os.path.join(output, os.path.basename(filename)), fourcc, 60.0, (width, height))

    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(total_frames):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames, out

# 标志变量    
active = True

def monitor_gpu():
    
    while active:
        gpus = GPUtil.getGPUs()
        list_gpus = []
        for gpu in gpus:
            # get the GPU id
            gpu_id = gpu.id
            # name of GPU
            gpu_name = gpu.name
            # get % percentage of GPU usage of that GPU
            gpu_load = f"{gpu.load*100}%"

            # get free memory in MB format
            gpu_free_memory = f"{gpu.memoryFree}MB"
            # get used memory
            gpu_used_memory = f"{gpu.memoryUsed}MB"
            # get total memory
            gpu_total_memory = f"{gpu.memoryTotal}MB"
            # get GPU temperature in Celsius
            gpu_temperature = f"{gpu.temperature} °C"
            gpu_mem_usage.append(gpu.memoryUsed) 
            gpu_temp.append(gpu.temperature)
            gpu_loads.append(gpu.load*100)
            list_gpus.append((
                gpu_id, gpu_name, gpu_load, gpu_free_memory, gpu_used_memory,
                gpu_total_memory, gpu_temperature
            ))

        print(tabulate(list_gpus, headers=("id", "name", "load", "free memory", "used memory", "total memory",
                                           "temperature")))
        with open('gpu_utils_1080.log', 'a') as f:
            f.write(tabulate(list_gpus, headers=("id", "name", "load", "free memory", "used memory", "total memory",
                                           "temperature")))
            f.write('\n')
        time.sleep(5)
    
def monitor_cpu():

  while active:
    
    cpu_percent = psutil.cpu_percent(1)
    memory_percent = psutil.virtual_memory().percent
    cpu_percents.append(cpu_percent)
    mem_percents.append(memory_percent)
    
    print(f"CPU Usage: {cpu_percent}%") 
    print(f"Memory Usage: {memory_percent}%")
    with open('cpu_mem_utils_1080.log', 'a') as f:
        f.write(f"CPU Usage: {cpu_percent}%\t\t")
        f.write(f"Memory Usage: {memory_percent}%\n")
    # 也可以写入日志等操作
    
    time.sleep(5)

frames_dir = 'runs/test6/frames/'
box_dir = "runs/test6/boxes/"
os.makedirs('runs/test6', exist_ok=True)
os.makedirs(frames_dir, exist_ok=True)
os.makedirs(box_dir, exist_ok=True)
gpu_mem_usage = [] 
gpu_temp = []
gpu_loads = []

cpu_percents = []
mem_percents = []

def main():
    # 创建并启动线程  
    t = Thread(target=monitor_gpu)
    t.daemon = True
    t.start()
    t2 = Thread(target=monitor_cpu)
    t2.daemon = True
    t2.start()
    args = parse_args()

#     imgs, out_vid = load_data('tests/06.mp4', 'runs/test6')
    
    
    os.makedirs('runs/test6', exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(box_dir, exist_ok=True)
# create object detector
    detector = Detector(
        model_path=args.det_model_path, device_name=args.device_name)
    # create pose detector
    pose_detector = PoseDetector(
        model_path=args.pose_model_path, device_name=args.device_name)
    
    
    
    startc = time.time()
    
    inference_total_time = 0
    count = 0
    fig, axs = plt.subplots(1, 5, figsize=(20, 5))
    axs[0].set_title('GPU Memory Usage (MB)')
    axs[1].set_title('GPU Temperature (Celsius)')
    axs[2].set_title('GPU Load (%)')
    axs[3].set_title('CPU Usage (%)')
    axs[4].set_title('Memory Usage (%)')
    while time.time() - startc < 300*60:
        cap = cv2.VideoCapture('tests/06.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        print(time.time())
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(width, height)
    
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        count += 1
        inference_time = 0
        for i in range(total_frames):
            ret, img = cap.read()
            if ret:
                inference_start = time.time()
                # apply detector
                bboxes, labels, _ = detector(img)
                print("good")
                # filter detections
                keep = np.logical_and(labels == 0, bboxes[..., 4] > 0.6)
                bboxes = bboxes[keep, :4]
                for box in bboxes:
                    left_top_x, left_top_y, right_down_x, right_down_y = [int(x) for x in box]
                    cv2.rectangle(img, (left_top_x, left_top_y), (right_down_x, right_down_y), (255, 0, 0), 2)
                # apply pose detector
                poses = pose_detector(img, bboxes)
                inference_end = time.time()
                print(f'inference_time: {inference_end-inference_start}\n')
                inference_time += inference_end-inference_start
                img_box = visualize(img, bboxes, poses, frames_dir+f'{i}.jpg', 0.5, 1280)
                with open(box_dir+f"{i}.txt", 'w') as f:
                    for box in bboxes:
                        left_top_x, left_top_y, right_down_x, right_down_y = [str(x) for x in box]
                        text = left_top_x + ' ' + left_top_y + ' ' + right_down_x + ' ' + right_down_y + '\n'
                        f.write(text)
            else:
                break
        axs[0].plot(gpu_mem_usage)
        
    
        axs[1].plot(gpu_temp)
        
    
        axs[2].plot(gpu_loads)
        
    
            # 新增CPU利用率子图 
        axs[3].plot(cpu_percents) 
        
    
        # 新增内存利用率子图
        axs[4].plot(mem_percents)
        
    
        plt.tight_layout()
        plt.savefig('system_stats.png')
        inference_time = inference_time / total_frames
        inference_total_time += inference_time
        with open('inference_time_1080.log', 'a') as f:
                f.write(f'===========================average inference_time: {inference_time}==========================\n')
        
        print(f'===========================average inference_time: {inference_time}==========================\n')
    with open('inference_time_1080.log', 'a') as f:
                f.write(f'**************************************************Average inference_time: {inference_total_time / count}***********************************************\n')
    print(f'**************************************************Average inference_time: {inference_total_time / count}***********************************************\n')
    active = False
    
    average_gpu_temp = sum(gpu_temp)/len(gpu_temp)
    max_gpu_temp = max(gpu_temp)
    average_gpu_utils = sum(gpu_loads)/len(gpu_loads)
    max_gpu_utils = max(gpu_loads)
    
    with open('gpu_utils_1080.log', 'a') as f:
        f.write(f'\naverage gpu temp: {average_gpu_temp},\t max gpu temp: {max_gpu_temp}\n')
        f.write(f'average gpu utils: {average_gpu_utils}%,\t max gpu utils: {max_gpu_utils}%\n')
    print(f'\naverage gpu temp: {average_gpu_temp},\t max gpu temp: {max_gpu_temp}\n')
    print(f'average gpu utils: {average_gpu_utils}%,\t max gpu utils: {max_gpu_utils}%\n')
    

    
    average_cpu_utils = sum(cpu_percents)/len(cpu_percents)
    average_mem_utils = sum(mem_percents)/len(mem_percents)
    with open('cpu_mem_utils_1080.log', 'a') as f:
        f.write(f'\naverage cpu utils: {average_cpu_utils},\t max cpu utils: {max(cpu_percents)}\n')
        f.write(f'average mem utils: {average_mem_utils}%,\t max mem utils: {max(mem_percents)}%\n')
    print(f'\naverage cpu utils: {average_cpu_utils},\t max cpu utils: {max(cpu_percents)}\n')
    print(f'average mem utils: {average_mem_utils}%,\t max mem utils: {max(mem_percents)}%\n')
    
if __name__ == '__main__':
    main()
