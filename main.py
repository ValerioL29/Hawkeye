from rich import print

from hawkeye.model import HOLO
from hawkeye.utils import MODELS_DIR, DATA_DIR

holo_model = HOLO(
    model=MODELS_DIR / "cfg" / "yolov8" / "yolov8n-multi.yaml",
    task="multitask",
    verbose=True
)

holo_model.load(weights={
    "detect": MODELS_DIR / "yolov8" / "yolov8n.pt",
    "drivable": MODELS_DIR / "yolov8" / "yolov8n-seg.pt",
    "lane": MODELS_DIR / "yolov8" / "yolov8n-seg.pt",
})

# Train
dataset_name = "test_data"
train_ret = holo_model.train(
    data=DATA_DIR / dataset_name / "data.yaml",
    epochs=1,
    batch=1,
    device="mps"
)
