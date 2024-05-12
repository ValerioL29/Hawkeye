from rich import print

from ultralytics import YOLO

from hawkeye.utils import MODELS_DIR
from hawkeye.core.task import MultitaskModel

model = YOLO(MODELS_DIR / "cfg" / "yolov8" / "yolov8n.yaml", verbose=True)
print(model.model.args)

multi_model = MultitaskModel(
    cfg=MODELS_DIR / "cfg" / "yolov8" / "yolov8n-multi.yaml",
    ch=3
)
