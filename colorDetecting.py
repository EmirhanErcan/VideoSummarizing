from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image

model = YOLO('yolov8n-seg.pt')

def detect_color(image):

    predict = model.predict(image, save=False, classes=[0], conf=0.4, save_txt=False)
     
    human_mask = (predict[0].masks.data[0].cpu().numpy() * 255).astype("uint8")
    
    human_mask_resized = cv2.resize(human_mask, (image.shape[1], image.shape[0]))
    
    background_mask = cv2.bitwise_not(human_mask_resized)
    
    image[background_mask == 255] = [0, 0, 0]
    
    human_mask_binary = cv2.threshold(human_mask_resized, 127, 255, cv2.THRESH_BINARY)[1]
    
    human_color = cv2.bitwise_and(image, image, mask=human_mask_binary)
    
    input_image = human_color

    color_ranges = [
        ((94, 80, 2), (120, 255, 255)),
        ((25, 52, 72), (102, 255, 255)),
        ((136, 87, 111), (180, 255, 255))
    ]

    masks = []
    areas = []

    hsv_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    for lower, upper in color_ranges:

        mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))
        
        masks.append(mask)
        
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        total_area = 0

        for contour in contours:

            area = cv2.contourArea(contour)

            total_area += area

        areas.append(total_area)

    max_area_index = areas.index(max(areas))

    colors = ['Blue', 'Green', 'Red']
    
    max_color = colors[max_area_index]

    color_text = max_color

    return color_text