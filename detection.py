from ultralytics import YOLO

model = YOLO("yolov8s.pt")

def detect_objects(frame, min_confidence=0.4):
    """Detect objects in a frame using YOLO model."""
    results = model(frame, classes=0, conf=min_confidence)
    return results