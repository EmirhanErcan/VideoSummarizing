from ultralytics import YOLO
import numpy as np
import cv2
import os

model = YOLO('yolov8n-seg.pt')

def colorFilter(input_video, inputColor, output_path, dict_frame_colors, dict_id_detected_time_seconds, dict_time_ids_xyxy, dict_id_color, dict_id_og_frames):

    cap = cv2.VideoCapture(input_video)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    output_name = "filtered_"
    
    output_video_path = os.path.join(output_path, output_name + os.path.basename(input_video))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    color_values = {
        "Blue":(255, 0, 0),
        "Green":(0, 255, 0),
        "Red":(0, 0, 255)
    }
    rect_color = color_values[inputColor]
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if ret == False:
            return output_video_path
        

        current_frame_index = cap.get(cv2.CAP_PROP_POS_FRAMES)
        current_frame_color = dict_frame_colors.get(current_frame_index)
        
        
        if current_frame_color is not None:
            if inputColor in current_frame_color:

                for track_id, color in dict_id_color.items():
                    
                    if color == inputColor and (current_frame_index in dict_id_og_frames[track_id]):
                        print("hello")
                        detected_time = dict_id_detected_time_seconds[track_id]
                        print(f"track_id = {track_id}")
                        print(f"detected_time = {detected_time}")
                        hour = int(detected_time / 3600)
                        minute = int((detected_time % 3600) / 60)
                        second = int(detected_time % 60)
                        x1, y1, x2, y2 = dict_time_ids_xyxy[current_frame_index][track_id]
                        w = x2 - x1
                        h = y2 - y1
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), rect_color, 1)
                        cv2.putText(frame, f"({hour}h {minute}m {second}s) {inputColor}",
                                   (int(x1) + 10, int(y1) - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, rect_color, 1, cv2.LINE_AA)

            out.write(frame)
        
    cap.release()
    out.release()




# matrix formatted image
def detect_color(image):

    predict = model.predict(image, save=False, classes=[0], conf=0.25, save_txt=False)
     
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




