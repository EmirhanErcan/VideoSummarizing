import numpy as np
import cv2
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from ultralytics import YOLO  # Import your YOLO segmentation model class
from colorDetecting import detect_color
from PIL import Image

deep_sort_weights = 'deep_sort/deep/checkpoint/ckpt.t7'
tracker = DeepSort(model_path=deep_sort_weights, min_confidence=0.3, max_iou_distance=0.5, max_age=20)

# Initialize YOLO model for segmentation
segment_model = YOLO('yolov8n-seg.pt')

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
                color_text = detect_color(image)
                track_colors[track_id] = color_text
            
            color_values = {
                "Blue":(0, 0, 255),
                "Green":(0, 255, 0),
                "Red":(255, 0, 0)
            }
            if len(track_colors) > 0:
                rect_color = color_values[track_colors[track_id]]
            

            detection_time = track_detection_times[track_id]
            hour = int(detection_time / 3600)
            minute = int((detection_time % 3600) / 60)
            second = int(detection_time % 60)

            x1, y1, x2, y2 = track.to_tlbr()
            w = x2 - x1
            h = y2 - y1

            

            cv2.rectangle(og_frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), rect_color, 1)
            cv2.putText(og_frame, f"Person-{track_id} ({hour}h {minute}m {second}s) {track_colors[track_id]}",
                        (int(x1) + 10, int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return og_frame
