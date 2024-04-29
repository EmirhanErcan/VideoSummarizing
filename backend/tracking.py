import numpy as np
import cv2
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

deep_sort_weights = 'backend/deep_sort/deep/checkpoint/ckpt.t7'
tracker = DeepSort(model_path=deep_sort_weights, max_age=5)

def update_tracker(results, frame, fps, track_detection_times, cap):
    """Update DeepSort tracker with bounding boxes and confidence scores."""
    og_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detected_frame_index = cap.get(cv2.CAP_PROP_POS_FRAMES)
    detected_time = detected_frame_index / fps

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

            detection_time = track_detection_times[track_id]
            hour = int(detection_time / 3600)
            minute = int((detection_time % 3600) / 60)
            second = int(detection_time % 60)


            x1, y1, x2, y2 = track.to_tlbr()
            w = x2 - x1
            h = y2 - y1

            color_id = track_id % 3
            if color_id == 0:
                color = (0, 0, 255)  # Red
            elif color_id == 1:
                color = (255, 0, 0)  # Blue
            else:
                color = (0, 255, 0)  # Green

            cv2.rectangle(og_frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
            cv2.putText(og_frame, f"Person-{track_id} ({hour}h {minute}m {second}s)", (int(x1) + 10, int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return og_frame