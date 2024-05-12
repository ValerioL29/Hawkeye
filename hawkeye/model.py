from pathlib import Path
from typing import Union

from ultralytics.engine.model import Model
from ultralytics.nn import attempt_load_one_weight

from hawkeye.core.task import MultitaskModel
from hawkeye.engine import MultitaskTrainer, MultitaskValidator, MultitaskPredictor


class HOLO(Model):
    """The Hawkeye YOLO (You Only Look Once) Multi-task model."""
    model: MultitaskModel

    def __init__(self, model="yolov8n.pt", task=None, verbose=False):
        """Initialize HOLO model"""
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

    def load(self, weights: Union[str, Path, dict] = "yolov8n.pt") -> Model:
        """
        Loads parameters from the specified weights file into the model.

        This method supports loading weights from a file or directly from a weights object. It matches parameters by
        name and shape and transfers them to the model.

        Args:
            weights (str | Path | dict): Path to the weights file or a weights object. Defaults to 'yolov8n.pt'.

        Returns:
            self (ultralytics.engine.model.Model): The instance of the class with loaded weights.

        Raises:
            AssertionError: If the model is not a PyTorch model.
        """
        self._check_is_pytorch_model()
        if isinstance(weights, (str, Path)):
            weights, self.ckpt = self.attempt_load_one_pretrain_multitask_weights(weights)
        elif isinstance(weights, dict):
            weights, self.ckpt = self.attempt_load_multi_pretrained_weights(weights)
        # Load the model within MultiTaskModel class
        self.model.load(weights)
        return self

    def attempt_load_one_pretrain_multitask_weights(self, weights: Union[str, Path]) -> Model:
        """
        Attempts to load the model from a single-pretrained weights file.

        Args:
            weights (str | Path): Path to the single-pretrained weights file.

        Returns:
            self (ultralytics.engine.model.Model): The instance of the class with loaded weights.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support loading single-pretrained weights.")

    @staticmethod
    def attempt_load_multi_pretrained_weights(weights_list: dict):
        """
        Attempts to load the model from a multi-pretrained weights file.

        Args:
            weights_list (dict): List of path to the multi-pretrained weights files.

        Returns:
            self (ultralytics.engine.model.Model): The instance of the class with loaded weights.
        """
        # Load pretrained weights separately for each task
        ret_weights, ret_ckpt = {}, {}
        for task, weights_path in weights_list.items():
            weights, ckpt = attempt_load_one_weight(weights_path)
            ret_weights[task] = weights
            ret_ckpt[task] = ckpt
        return ret_weights, ret_ckpt
