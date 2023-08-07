import os
import cv2


img_width = 640
img_height = 480

person_width = 94
person_height = 175

person_id_5_bbox = [486.25, 164.34375, 580.0, 338.625]

area_bounding_right = (person_id_5_bbox[2], person_id_5_bbox[3])
area_bounding_left = (img_width-area_bounding_right[0], person_id_5_bbox[3])

area_bbox = [area_bounding_left[0], area_bounding_left[1], area_bounding_right[0], img_height]


# 设置字体和大小
font = cv2.FONT_HERSHEY_DUPLEX
font_scale = 0.5


def cal_iou(bbox1, bbox2):
    xmin = max(bbox1[0], bbox2[0])
    ymin = max(bbox1[1], bbox2[1])
    xmax = min(bbox1[2], bbox2[2])
    ymax = min(bbox1[3], bbox2[3])
    
    inter_w = xmax - xmin
    inter_h = ymax - ymin
    
    inter_area = inter_w * inter_h
    
    iou = inter_area / (person_width*person_height*2 - inter_area)
    return iou


if __name__ == "__main__":
    front_person_standup_pos_y = 394.125
    front_person_height = 250
    
    bboxes = []
    # find person id 2 by id 5:
    person_id_2_bbox = []
    for x_shift in range(int(area_bbox[2] - area_bbox[0])):
        person_id_2_right_down_x = person_id_5_bbox[2] - x_shift
        person_id_2_left_top_x = person_id_2_right_down_x - 127
        person_id_2_left_top_y = front_person_standup_pos_y - front_person_height
        person_id_2_bbox = [person_id_2_left_top_x, person_id_2_left_top_y, person_id_2_right_down_x, front_person_standup_pos_y]
        
        # calculate iou for id 4 and 5
        iou = cal_iou(person_id_2_bbox, person_id_5_bbox)
        
        if iou < 0.005:
            print("got it")
            break
    img = cv2.imread('runs/test3/frames/0.jpg')
    cv2.putText(img, 'id: 5', (int(person_id_5_bbox[0]+(person_id_5_bbox[2]-person_id_5_bbox[0])/4), int(person_id_5_bbox[1]-2)), font, font_scale, (0, 0, 255), 2)
    cv2.rectangle(img, (int(person_id_5_bbox[0]), int(person_id_5_bbox[1])), (int(person_id_5_bbox[2]), int(person_id_5_bbox[3])), (0, 0, 255), 2)
    id_2_bbox = [int(x) for x in person_id_2_bbox]
    cv2.rectangle(img, (id_2_bbox[0], id_2_bbox[1]), (id_2_bbox[2], id_2_bbox[3]), (0, 0, 255), 2)
    # 在坐标(10,30)处写文字,颜色为红色,字体与大小设置为之前设定
    cv2.putText(img, 'id: 2', (int(id_2_bbox[0]+(id_2_bbox[2]-id_2_bbox[0])/4), int(id_2_bbox[1]-2)), font, font_scale, (0, 0, 255), 2)
    # find person id 4 by id 2:
    person_id_4_bbox = []
    for x_shift in range(int(area_bbox[2] - area_bbox[0])):
        person_id_4_right_down_x = person_id_2_bbox[2] - x_shift
        person_id_4_left_top_x = person_id_4_right_down_x - person_width
        person_id_4_left_top_y = person_id_5_bbox[3] - person_height
        person_id_4_bbox = [person_id_4_left_top_x, person_id_4_left_top_y, person_id_4_right_down_x, person_id_5_bbox[3]]
        
        # calculate iou for id 4 and 5
        iou = cal_iou(person_id_4_bbox, person_id_2_bbox)
        
        if iou < 0.005:
            print("got it")
            break
    id_4_bbox = [int(x) for x in person_id_4_bbox]
    cv2.rectangle(img, (id_4_bbox[0], id_4_bbox[1]), (id_4_bbox[2], id_4_bbox[3]), (0, 0, 255), 2)
    cv2.putText(img, 'id: 4', (int(id_4_bbox[0]+(id_4_bbox[2]-id_4_bbox[0])/4), int(id_4_bbox[1]-2)), font, font_scale, (0, 0, 255), 2)
    
    # find person id 1 by id 4:
    person_id_1_bbox = []
    for x_shift in range(int(area_bbox[2] - area_bbox[0])):
        person_id_1_right_down_x = person_id_4_bbox[2] - x_shift
        person_id_1_left_top_x = person_id_1_right_down_x - 127
        person_id_1_left_top_y = front_person_standup_pos_y - front_person_height
        person_id_1_bbox = [person_id_1_left_top_x, person_id_1_left_top_y, person_id_1_right_down_x, front_person_standup_pos_y]
        
        # calculate iou for id 4 and 5
        iou = cal_iou(person_id_4_bbox, person_id_1_bbox)
        
        if iou < 0.005:
            print("got it")
            break
    id_1_bbox = [int(x) for x in person_id_1_bbox]
    cv2.rectangle(img, (id_1_bbox[0], id_1_bbox[1]), (id_1_bbox[2], id_1_bbox[3]), (0, 0, 255), 2)
    cv2.putText(img, 'id: 1', (int(id_1_bbox[0]+(id_1_bbox[2]-id_1_bbox[0])/4), int(id_1_bbox[1]-2)), font, font_scale, (0, 0, 255), 2)
    
    # find person id 3 by id 1:
    person_id_3_bbox = []
    for x_shift in range(int(area_bbox[2] - area_bbox[0])):
        person_id_3_right_down_x = person_id_1_bbox[2] - x_shift
        person_id_3_left_top_x = person_id_3_right_down_x - person_width
        person_id_3_left_top_y = person_id_5_bbox[3] - person_height
        person_id_3_bbox = [person_id_3_left_top_x, person_id_3_left_top_y, person_id_3_right_down_x, person_id_5_bbox[3]]
        
        # calculate iou for id 4 and 5
        iou = cal_iou(person_id_1_bbox, person_id_3_bbox)
        
        if iou < 0.005:
            print("got it")
            break
    id_3_bbox = [int(x) for x in person_id_3_bbox]
    cv2.rectangle(img, (id_3_bbox[0], id_3_bbox[1]), (id_3_bbox[2], id_3_bbox[3]), (0, 0, 255), 2)
    cv2.putText(img, 'id: 3', (int(id_3_bbox[0]+(id_3_bbox[2]-id_3_bbox[0])/4), int(id_3_bbox[1]-2)), font, font_scale, (0, 0, 255), 2)
    cv2.imwrite('test.jpg', img)
    bboxes.append(person_id_1_bbox)
    bboxes.append(person_id_2_bbox)
    bboxes.append(person_id_3_bbox)
    bboxes.append(person_id_4_bbox)
    bboxes.append(person_id_5_bbox)
    for i in range(1, 6):
        with open(f'runs/test4/id_{i}.txt', 'w') as f:
            box = bboxes[i-1]
            left_top_x, left_top_y, right_down_x, right_down_y = [str(x) for x in box]
            text = left_top_x + ' ' + left_top_y + ' ' + right_down_x + ' ' + right_down_y + '\n'
            f.write(text)