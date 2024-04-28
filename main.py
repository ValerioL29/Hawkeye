from rich import print
from ultralytics import YOLO

from hawkeye.utils import MODELS_DIR, DATA_DIR

# Build a YOLOv9c model from scratch
# model = YOLO('yolov9c.yaml')

# Build a YOLOv9c model from pretrained weight
model = YOLO(MODELS_DIR / "yolov9" / "yolov9e-seg.pt")

# Display model information (optional)
print(f"Model info: {model.info()}")

# Train the model on the COCO8 example dataset for 100 epochs
# results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

# Run inference with the YOLOv9c model on the 'bus.jpg' image
# results = model(DATA_DIR / "demo" / "bus.jpg")
# print(type(results))
# print(type(results[0]))
