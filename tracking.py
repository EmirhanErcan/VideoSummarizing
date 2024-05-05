import numpy as np
import cv2
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from ultralytics import YOLO  # Import your YOLO segmentation model class
from PIL import Image

deep_sort_weights = 'deep_sort/deep/checkpoint/ckpt.t7'
tracker = DeepSort(model_path=deep_sort_weights, min_confidence=0.3, max_iou_distance=0.5, max_age=20)

# Initialize YOLO model for segmentation
segment_model = YOLO('yolov8x-seg.pt')

def update_tracker(results, frame, fps, track_detection_times, cap, track_colors):
    og_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detected_frame_index = cap.get(cv2.CAP_PROP_POS_FRAMES)
    detected_time = detected_frame_index / fps

    # Dictionary to store color counts for each track
    track_colors = track_colors

    for result in results:
        boxes = result.boxes
        conf = boxes.conf
        # for color detecting - segmentation START
        xyxy = boxes.xyxy.cpu().numpy()
        bboxes_xyxy = np.array(xyxy, dtype= float)
        # for color detecting - segmentation END
        xywh = boxes.xywh.cpu().numpy()
        bboxes_xywh = np.array(xywh, dtype=float)
        tracks = tracker.update(bboxes_xywh, conf, og_frame)

        for track in tracker.tracker.tracks:
            track_id = track.track_id
            if track_id not in track_detection_times:
                track_detection_times[track_id] = detected_time
                
                # color filtering - segmentation START
                image = cv2.cvtColor(og_frame.copy(), cv2.COLOR_RGB2BGR)
                predict = segment_model.predict(image, save=False, classes=[0], save_txt=False)
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
                track_colors[track_id] = color_text
                #x_min, y_min, x_max, y_max = bboxes_xyxy[0]
                #copy_og_frame = cv2.cvtColor(og_frame, cv2.COLOR_RGB2BGR)
                #cropped_img = Image.fromarray(copy_og_frame)
                #cropped_img = cropped_img.crop((x_min, y_min, x_max, y_max))
                #cropped_img_array = np.array(cropped_img)
                #color_text = detect_color(cropped_img_array)
                # color filtering - segmentation END

                #if track_id in color_counts:
                #    if hue not in color_counts[track_id]:
                #        color_counts[track_id][hue] = 0
                #else:
                #   color_counts[track_id] = {}  # Initialize color count for new track
            color_values = {
                "Blue":(0, 0, 255),
                "Green":(0, 255, 0),
                "Red":(255, 0, 0)
            }
            detection_time = track_detection_times[track_id]
            hour = int(detection_time / 3600)
            minute = int((detection_time % 3600) / 60)
            second = int(detection_time % 60)

            x1, y1, x2, y2 = track.to_tlbr()
            w = x2 - x1
            h = y2 - y1

            

            cv2.rectangle(og_frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color_values[track_colors[track_id]], 1)
            cv2.putText(og_frame, f"Person-{track_id} ({hour}h {minute}m {second}s) {track_colors[track_id]}",
                        (int(x1) + 10, int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return og_frame
