import cv2
from detection import detect_objects
from tracking import update_tracker
from segmentation import segmentate_objects
import os
import pickle


def process_video(video_path, output_path, colorFilter=None):
    """Process input video and save the output video."""
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'H264')

    output_name = "output_"
    # Define the output video path in the specified folder
    
    output_video_path = os.path.join(output_path, output_name + os.path.basename(video_path))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    

    track_colors = {}
    detected_frame_numbers = {}
    detection_times_tracks = {}
    bbox_tracks = {}

    detected_frame_index = cap.get(cv2.CAP_PROP_POS_FRAMES)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"detected_frame_numbers::: {detected_frame_numbers}")
            print(f"detection_times_tracks::: {detection_times_tracks}")
            print(f"bbox_tracks::: {len(bbox_tracks)}")
            return output_video_path, detected_frame_numbers, detection_times_tracks, bbox_tracks


        percentage = f"{(detected_frame_index / total_frames)*100}%"
        with open("timer_stubs/percentage.pkl", 'wb') as f:
            pickle.dump(percentage, f)
        

        if not frame_contains_humans(frame):
            continue

        results = detect_objects(frame)
        processed_frame = update_tracker(results, frame, fps, cap, track_colors, colorFilter, detected_frame_numbers, detection_times_tracks, bbox_tracks)

        if processed_frame is not None:
            out.write(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))

    
    cap.release()
    out.release()
    
    

    


def frame_contains_humans(frame, min_confidence=0.5):
    # Detect humans in the frame using your YOLO model
    results = detect_objects(frame, min_confidence)
    
    for result in results:
        if len(result.boxes.xyxy) > 0:
            return True
    return False
    
    
    
