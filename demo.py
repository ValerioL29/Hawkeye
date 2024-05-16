import logging
from rich.logging import RichHandler
from pathlib import Path
from ultralytics import YOLO

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",  # %(asctime)s [%(levelname)s] %(message)s
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("rich")

ROOT_DIR = Path("/public/home/acsy8g9pue/")
MODEL_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "datasets" / "bdd100k"

PROJECT_DIR = ROOT_DIR / "codes" / "ljc"
CHECKPOINTS_DIR = PROJECT_DIR / "checkpoints"
DATASETS_DIR = PROJECT_DIR / "datasets"

logger.info("Start training script".center(50, "="))

dataset_name = "det_20_labels_trainval"
model_conf = CHECKPOINTS_DIR / "cfg" / "yolov8" / "yolov8l.yaml"
model = YOLO(model_conf).load(CHECKPOINTS_DIR / "yolov8" / "yolov8l.pt")

logger.info(f"Model info: {model.info()}")

model.train(
    name="size_l_freeze_10_epochs_10_adam_single",
    data=DATASETS_DIR / dataset_name / "detection_traindata.yaml",
    imgsz=1280,
    batch=-1,  # Enable auto batch
    freeze=10,
    single_cls=True,  # Single Classification for objects
    amp=False,
    cache=True,  # Cache datasets in Memory
    workers=11,  # Number of workers threads for data loading, per RANK for Multi-GPU
    epochs=20,  # About 10 is fair enough, after 10 there's a sharp descent of performance
    cos_lr=True,  # CosineLRScheduler
    optimizer="Adam",
    lr0=0.1,  # Initial learning rate
    dropout=0.2,  # Dropout for preventing overfitting
    plots=True,
    device=0  # [0, 1] Multi-GPU Training
)

logger.info("End training script".center(50, "="))
