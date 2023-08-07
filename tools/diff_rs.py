import cv2


if __name__ == "__main__":
    w_scale = 1
    h_scale = 1
    
    bbox = [471.5, 208.125, 551.0, 371.625]
    new_bbox = [bbox[0]*w_scale, bbox[1]*h_scale, bbox[2]*w_scale, bbox[3]*h_scale]
    
    new_bbox = [int(x) for x in new_bbox]
    
    img = cv2.imread('runs/test2/frames/11.jpg')
    cv2.rectangle(img, (new_bbox[0], new_bbox[1]), (new_bbox[2], new_bbox[3]), (255, 0, 0), 2)
    
    bbox = [153.0, 184.0, 273.0, 424.5]
    new_bbox = [bbox[0]*w_scale, bbox[1]*h_scale, bbox[2]*w_scale, bbox[3]*h_scale]
    
    new_bbox = [int(x) for x in new_bbox]
    cv2.rectangle(img, (new_bbox[0], new_bbox[1]), (new_bbox[2], new_bbox[3]), (255, 0, 0), 2)
    
    bbox = [352.0, 184.0, 472.0, 424.5]
    new_bbox = [bbox[0]*w_scale, bbox[1]*h_scale, bbox[2]*w_scale, bbox[3]*h_scale]
    
    new_bbox = [int(x) for x in new_bbox]
    cv2.rectangle(img, (new_bbox[0], new_bbox[1]), (new_bbox[2], new_bbox[3]), (255, 0, 0), 2)
    
    bbox = [74.0, 207.625, 153.0, 371.625]
    new_bbox = [bbox[0]*w_scale, bbox[1]*h_scale, bbox[2]*w_scale, bbox[3]*h_scale]
    
    new_bbox = [int(x) for x in new_bbox]
    cv2.rectangle(img, (new_bbox[0], new_bbox[1]), (new_bbox[2], new_bbox[3]), (255, 0, 0), 2)
    
    bbox = [273.0, 207.625, 352.0, 371.625]
    new_bbox = [bbox[0]*w_scale, bbox[1]*h_scale, bbox[2]*w_scale, bbox[3]*h_scale]
    
    new_bbox = [int(x) for x in new_bbox]
    cv2.rectangle(img, (new_bbox[0], new_bbox[1]), (new_bbox[2], new_bbox[3]), (255, 0, 0), 2)
    cv2.imwrite('test.jpg', img)