from rich import print
from ultralytics import YOLO

from hawkeye.utils import MODELS_DIR, DATA_DIR

# Build a YOLOv8l model from pretrained weight
model = YOLO(MODELS_DIR / "cfg" / "yolov8" / "yolov8n.yaml")\
    .load(MODELS_DIR / "yolov8" / "yolov8n.pt")

# Display model information (optional)
print(f"Model info: {model.info()}")
