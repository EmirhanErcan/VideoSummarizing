from ultralytics import YOLO
import numpy as np
import cv2
import os
import time

model = YOLO('yolov8x-seg.pt')

def colorFilter(input_video, inputColor, output_path, dict_frame_colors, dict_id_detected_time_seconds, dict_time_ids_xyxy, dict_id_color, dict_id_og_frames, total_frames, progress_callback= None):

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
    processFrameCounter = 0
    #anyColor = True
    while cap.isOpened():
        start = time.time()
        ret, frame = cap.read()
        current_frame_index = cap.get(cv2.CAP_PROP_POS_FRAMES)
        current_frame_color = dict_frame_colors.get(current_frame_index)
        
        if ret == False:
            return output_video_path
        
        
        if current_frame_color is not None:
            if inputColor in current_frame_color:
                for track_id, color in dict_id_color.items():
                    
                    if color == inputColor and (current_frame_index in dict_id_og_frames[track_id]):
                        detected_time = dict_id_detected_time_seconds[track_id]
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
        if progress_callback:
            end = time.time()
            totalTime = end-start
            progress_callback(int(current_frame_index), total_frames, totalTime)
            
    cap.release()
    out.release()




# matrix formatted image
def detect_color(image):

    predict = model.predict(image, save=False, classes=[0], conf=0.05, save_txt=False)
    if predict[0].masks is None:
        return "Unknown"
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

    hsv_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    colors = ['Blue', 'Green', 'Red']
    pixel_counts = []
    for lower, upper in color_ranges:
        mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))
        pixel_count = np.sum(mask == 255)  # Count the number of white pixels in the mask
        pixel_counts.append(pixel_count)

    max_pixel_count_index = pixel_counts.index(max(pixel_counts))
    dominant_color = colors[max_pixel_count_index]
    
    white_pixel_count = np.sum(human_mask_binary == 255)

    if pixel_counts[max_pixel_count_index] > white_pixel_count/7:
        return dominant_color
    return "Unknown"

