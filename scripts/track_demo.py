from rich import print
from ultralytics import YOLO

from hawkeye.utils import MODELS_DIR

# Build a YOLOv8l model from pretrained weight
model = YOLO(MODELS_DIR / "cfg" / "yolov8" / "yolov8l.yaml")\
    .load(MODELS_DIR / "yolov8" / "yolov8l.pt")

# Perform tracking with the model
# results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True)  # Tracking with default tracker
results = model.track(
    source="https://youtu.be/N0gzsIzzPJ4?si=mBAaBRN4ysb1YuCx",
    show=True, tracker=MODELS_DIR / "cfg" / "trackers" / "bytetrack.yaml"
)  # Tracking with ByteTrack tracker

print(results)
