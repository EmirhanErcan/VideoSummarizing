import cv2
from detection import detect_objects
from tracking import update_tracker

def process_video(video_path, output_path):
    """Process input video and save the output video."""
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    track_detection_times = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if not frame_contains_humans(frame):
            continue

        results = detect_objects(frame)
        processed_frame = update_tracker(results, frame, fps, track_detection_times, cap)
        out.write(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))

    cap.release()
    out.release()

def frame_contains_humans(frame, min_confidence=0.5):
    # Detect humans in the frame using your YOLO model
    results = detect_objects(frame, min_confidence)
    
    # Check if any bounding boxes are detected
    if isinstance(results, list):
        for result in results:
            if len(result.boxes.xyxy) > 0:
                return True
    else:
        if len(results.boxes.xyxy) > 0:
            return True
    
    return False
