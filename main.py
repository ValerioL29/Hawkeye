from PIL import Image
from rich import print
from ultralytics import YOLO
from ultralytics.engine.results import Results

from hawkeye.utils import MODELS_DIR, DATA_DIR, OUTPUTS_DIR

# Build a YOLOv9c model from scratch
# model = YOLO('yolov9c.yaml')

# Build a YOLOv9c model from config file and transferring weights from a pre-trained model
# This is the recommended way to load a model for training
model = YOLO(MODELS_DIR / "cfg" / "yolov9" / "yolov9c.yaml") \
    .load(MODELS_DIR / "yolov9" / "yolov9c.pt")

results = model.train(data='test.yaml', epochs=1, imgsz=(1280, 720), device="gpu")

# Display model information (optional)
# print(f"Model info: {model.info()}")
print(f"Results: {results}")

# # Run inference with the YOLOv9c model on the 'bus.jpg' image
# results: list[Results] = model(DATA_DIR / "demo" / "bus.jpg")
#
# for result in results:
#     # Save the result image
#     image_arr = result.plot()
#     image = Image.fromarray(image_arr[..., ::-1])  # RGB PIL image
#     image.save(OUTPUTS_DIR / "bus_yolov9c.jpg")
