from pathlib import Path

from ultralytics.engine.model import Model

from hawkeye.core.task import MultitaskModel
from hawkeye.engine import MultitaskTrainer, MultitaskValidator, MultitaskPredictor


class HOLO(Model):
    """The Hawkeye YOLO (You Only Look Once) Multi-task model."""

    def __init__(self, model="yolov8n.pt", task=None, verbose=False):
        """Initialize HOLO model"""
        path = Path(model)
        # Continue with default HOLO initialization
        super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "multitask": {
                "model": MultitaskModel,
                "trainer": MultitaskTrainer,
                "validator": MultitaskValidator,
                "predictor": MultitaskPredictor,
            }
        }
