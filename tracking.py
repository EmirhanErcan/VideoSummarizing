import numpy as np
import cv2
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from ultralytics import YOLO  # Import your YOLO segmentation model class
from colorFilterFile import detect_color

deep_sort_weights = 'deep_sort/deep/checkpoint/ckpt.t7'
tracker = DeepSort(model_path=deep_sort_weights, min_confidence=0.3, max_iou_distance=0.7, max_age=20, n_init=4)


def update_tracker(results, frame, fps, cap, dict_id_color, dict_id_og_frames, dict_id_detected_time_seconds, dict_time_ids_xyxy, dict_frame_colors):
    og_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detected_frame_index = cap.get(cv2.CAP_PROP_POS_FRAMES)
    detected_time_seconds = detected_frame_index / fps


    dict_time_ids_xyxy[detected_frame_index] = {} # CHANGED 08.05.2024 20:25
    
    # Dictionary to store color counts for each track
    dict_id_color = dict_id_color
    willWeReturnFrame = False
    for result in results:
        boxes = result.boxes
        conf = boxes.conf
        xyxy = boxes.xyxy.cpu().numpy()
        bboxes_xyxy = np.array(xyxy, dtype= float)
        xywh = boxes.xywh.cpu().numpy()
        bboxes_xywh = np.array(xywh, dtype=float)

        tracks = tracker.update(bboxes_xywh, conf, og_frame)

        rect_color = (0, 0, 0)
        for track in tracker.tracker.tracks:
            track_id = track.track_id

            # track.to_tlbr() -> xmin, ymin, xmax, ymax
            dict_time_ids_xyxy[detected_frame_index][track_id] = track.to_tlbr()
            

            if track_id not in dict_id_og_frames:
                dict_id_og_frames[track_id] = []
            dict_id_og_frames[track_id].append(detected_frame_index) # CHANGED 08.05.2024 19:49

            # insan ilk defa algılanıyorsa algılanma zamanını yazıyoruz ve insanın rengini hesaplayıp dictionary içine atıyoruz 
            if track_id not in dict_id_detected_time_seconds: # CHANGED 08.05.2024 19:49
                dict_id_detected_time_seconds[track_id] = detected_time_seconds
                image = cv2.cvtColor(og_frame.copy(), cv2.COLOR_RGB2BGR)
                color_text = detect_color(image)
                dict_id_color[track_id] = color_text
            if detected_frame_index not in dict_frame_colors:
                dict_frame_colors[detected_frame_index] = []
            track_id_color = dict_id_color[track_id]
            dict_frame_colors[detected_frame_index].append(track_id_color)

            color_values = {
                "Blue":(0, 0, 255),
                "Green":(0, 255, 0),
                "Red":(255, 0, 0)
            }
            if len(dict_id_color) > 0:
                rect_color = color_values[dict_id_color[track_id]]
            

            detection_time = dict_id_detected_time_seconds[track_id]
            hour = int(detection_time / 3600)
            minute = int((detection_time % 3600) / 60)
            second = int(detection_time % 60)

            x1, y1, x2, y2 = track.to_tlbr()
            w = x2 - x1
            h = y2 - y1

            willWeReturnFrame = True

            cv2.rectangle(og_frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), rect_color, 1)
            cv2.putText(og_frame, f"Person-{track_id} ({hour}h {minute}m {second}s) {dict_id_color[track_id]}",
                        (int(x1) + 10, int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


    return og_frame


