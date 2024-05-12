from rich import print
from ultralytics import YOLO

from hawkeye.utils import MODELS_DIR, DATA_DIR, ASSETS_DIR

# Build a YOLOv9c model from scratch
# model = YOLO('yolov9c.yaml')

# Build a YOLOv9c model from config file and transferring weights from a pre-trained model
# This is the recommended way to load a model for training
model = YOLO(MODELS_DIR / "cfg" / "yolov9" / "yolov9c-seg.yaml") \
    .load(MODELS_DIR / "yolov9" / "yolov9c-seg.pt")

print(f"Model info: {model.info()}")

model_ret = model.train(
    data=DATA_DIR / "lane_labels_trainval_new" / "lane.yaml",
    epochs=1,
    imgsz=(1280, 720),
    amp=False,
    device="mps"
)

results = model(ASSETS_DIR / "demo" / "bus.jpg")
