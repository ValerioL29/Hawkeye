from rich import print

from ultralytics import YOLO

from hawkeye.model import HOLO
from hawkeye.core.task import MultitaskModel
from hawkeye.utils import MODELS_DIR

model = YOLO(MODELS_DIR / "cfg" / "yolov8" / "yolov8n.yaml", verbose=True)

multi_model = MultitaskModel(
    cfg=MODELS_DIR / "cfg" / "yolov8" / "yolov8n-multi.yaml",
    ch=3
)

holo_model = HOLO(
    model=MODELS_DIR / "cfg" / "yolov8" / "yolov8n-multi.yaml",
    task="multitask",
    verbose=True
)

print(holo_model.info())

holo_model.load(weights={
    "detect": MODELS_DIR / "yolov8" / "yolov8n.pt",
    "drivable": MODELS_DIR / "yolov8" / "yolov8n-seg.pt",
    "lane": MODELS_DIR / "yolov8" / "yolov8n-seg.pt",
})
