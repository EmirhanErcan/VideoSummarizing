# %%

# %%

# %%
import ultralytics
ultralytics.__version__

# %%

# %%
import torch
import torchvision
torch.__version__
torchvision.__version__

# %%
torch.cuda.get_device_name(0)

# %% [markdown]
# # Detect, track and count Persons

# %%
from ultralytics import YOLO

import time
import torch
import cv2
import torch.backends.cudnn as cudnn
from PIL import Image
import colorsys
import numpy as np

from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from deep_sort.sort.tracker import Tracker

deep_sort_weights = 'deep_sort/deep/checkpoint/ckpt.t7'
tracker = DeepSort(model_path=deep_sort_weights, max_age=10)

# %%
# Define the video path
video_path = 'securitycam.mp4'

cap = cv2.VideoCapture(video_path)

# Get the video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = 'output5.avi'
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = YOLO("yolov8n.pt") 

frames = []

unique_track_ids = set()

# %%
start_time = time.perf_counter()
detection_start_time = None  # Time göstergesi için ekledim
track_detection_times = {} # NEW1


def frame_contains_humans(frame, min_confidence=0.5):
    # Detect humans in the frame using your YOLO model
    results = model(frame, classes=0, conf=min_confidence)
    
    # Check if any bounding boxes are detected
    if isinstance(results, list):
        for result in results:
            if len(result.boxes.xyxy) > 0:
                return True
    else:
        if len(results.boxes.xyxy) > 0:
            return True
    
    return False



while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break


    og_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # frame = og_frame.copy() saat 18:33 27.04.2024

    
    results = model(frame, classes=0, conf=0.5)

    if detection_start_time is None:  # Time göstergesi için ekledim
        detection_start_time = time.perf_counter()  # Time göstergesi için ekledim

    current_time = time.perf_counter() # NEW1
    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        conf = boxes.conf
        xywh = boxes.xywh.cpu().numpy()  # box with xywh format, (N, 4)
        bboxes_xywh = np.array(xywh, dtype=float)
        tracks = tracker.update(bboxes_xywh, conf, og_frame)


        for track in tracker.tracker.tracks:
            track_id = track.track_id

            if track_id not in track_detection_times: # NEW1
                track_detection_times[track_id] = current_time # NEW1

            detection_time = track_detection_times[track_id] # NEW1

            x1, y1, x2, y2 = track.to_tlbr()  # Get bounding box coordinates in (x1, y1, x2, y2) format
            w = x2 - x1  # Calculate width
            h = y2 - y1  # Calculate height

            # Set color values for red, blue, and green
            red_color = (0, 0, 255)  # (B, G, R)
            blue_color = (255, 0, 0)  # (B, G, R)
            green_color = (0, 255, 0)  # (B, G, R)

            # Determine color based on track_id
            color_id = track_id % 3
            if color_id == 0:
                color = red_color
            elif color_id == 1:
                color = blue_color
            else:
                color = green_color

            cv2.rectangle(og_frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

            text_color = (0, 0, 0)  # Black color for text
            cv2.putText(og_frame, f"Person-{track_id} ({detection_time - detection_start_time:.1f}s)", (int(x1) + 10, int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

            # Add the track_id to the set of unique track IDs
            # unique_track_ids.add(track_id)

   

    # Update FPS and place on frame
    # current_time = time.perf_counter() satırını yukarı aldım
    
    # Append the frame to the list
    # frames.append(og_frame) saat 18:33 27.04.2024

    # Write the frame to the output video file
    if frame_contains_humans(frame):
         out.write(cv2.cvtColor(og_frame, cv2.COLOR_BGR2RGB))

    # Show the frame
    #cv2.imshow("Video", og_frame)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #   break

cap.release()
out.release()

# %%



