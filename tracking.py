import numpy as np
import cv2
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

deep_sort_weights = 'deep_sort/deep/checkpoint/ckpt.t7'
tracker = DeepSort(model_path=deep_sort_weights, min_confidence=0.3, max_iou_distance=0.5, max_age=20)

def update_tracker(results, frame, fps, track_detection_times, cap, color_counts):
    og_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detected_frame_index = cap.get(cv2.CAP_PROP_POS_FRAMES)
    detected_time = detected_frame_index / fps

    # Dictionary to store color counts for each track
    color_counts = color_counts

    for result in results:
        boxes = result.boxes
        conf = boxes.conf
        xywh = boxes.xywh.cpu().numpy()
        bboxes_xywh = np.array(xywh, dtype=float)
        tracks = tracker.update(bboxes_xywh, conf, og_frame)

        for track in tracker.tracker.tracks:
            track_id = track.track_id
            if track_id not in track_detection_times:
                track_detection_times[track_id] = detected_time
                if track_id in color_counts:
                    if hue not in color_counts[track_id]:
                        color_counts[track_id][hue] = 0
                else:
                    color_counts[track_id] = {}  # Initialize color count for new track

            detection_time = track_detection_times[track_id]
            hour = int(detection_time / 3600)
            minute = int((detection_time % 3600) / 60)
            second = int(detection_time % 60)

            x1, y1, x2, y2 = track.to_tlbr()
            w = x2 - x1
            h = y2 - y1

            # COLOR DETECTION FOR EACH TRACK
            person_roi = og_frame[int(y1):int(y2), int(x1):int(x2)]
            hsv_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2HSV)
            dominant_color_threshold = 0.8

            for y in range(hsv_roi.shape[0]):
                for x in range(hsv_roi.shape[1]):
                    pixel_color = hsv_roi[y, x]
                    hue = pixel_color[0]
                    if hue not in color_counts[track_id]:
                        color_counts[track_id][hue] = 0
                    color_counts[track_id][hue] += 1

            dominant_color = max(color_counts[track_id], key=color_counts[track_id].get)

            if 0 <= dominant_color <= 15:
                color = (0, 0, 255)  # Blue
                color_text = "Blue"
            elif 16 <= dominant_color <= 45:
                color = (255, 0, 0)  # Red
                color_text = "Red"
            else:
                color = (0, 255, 0)  # Green
                color_text = "Green"

            cv2.rectangle(og_frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 1)
            cv2.putText(og_frame, f"Person-{track_id} ({hour}h {minute}m {second}s) {color_text}",
                        (int(x1) + 10, int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return og_frame
