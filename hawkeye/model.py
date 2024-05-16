from pathlib import Path
from typing import Union

from ultralytics.cfg import TASK2DATA
from ultralytics.engine.model import Model
from ultralytics.nn import attempt_load_one_weight
from ultralytics.utils import checks, yaml_load, DEFAULT_CFG_DICT, RANK

from hawkeye.core.task import MultitaskModel
from hawkeye.engine import MultitaskTrainer, MultitaskValidator, MultitaskPredictor
from hawkeye.utils import logger as LOGGER


class HOLO(Model):
    """The Hawkeye YOLO (You Only Look Once) Multi-task model."""
    model: MultitaskModel
    trainer: MultitaskTrainer
    validator: MultitaskValidator
    predictor: MultitaskPredictor

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

    def train(
            self,
            trainer=None,
            **kwargs,
    ):
        """
        Trains the model using the specified dataset and training configuration.

        This method facilitates model training with a range of customizable settings and configurations. It supports
        training with a custom trainer or the default training approach defined in the method. The method handles
        different scenarios, such as resuming training from a checkpoint, integrating with Ultralytics HUB, and
        updating model and configuration after training.

        Args:
            trainer (BaseTrainer, optional): An instance of a custom trainer class for training the model. If None, the
                method uses a default trainer. Defaults to None.
            **kwargs (any): Arbitrary keyword arguments representing the training configuration. These arguments are
                used to customize various aspects of the training process.

        Returns:
            (dict | None): Training metrics if available and training is successful; otherwise, None.

        Raises:
            AssertionError: If the model is not a PyTorch model.
            PermissionError: If there is a permission issue with the HUB session.
            ModuleNotFoundError: If the HUB SDK is not installed.
        """
        # Prerequisites
        self._check_is_pytorch_model()
        checks.check_pip_update_available()

        overrides = (
            yaml_load(checks.check_yaml(kwargs["cfg"]))
            if kwargs.get("cfg")
            else self.overrides
        )
        custom = {
            # NOTE: handle the case when 'cfg' includes 'data'.
            "data": overrides.get("data") or DEFAULT_CFG_DICT["data"] or TASK2DATA['detect'],
            "model": self.overrides["model"],
            "task": self.task,
        }  # method defaults
        args = {
            **overrides,
            **custom,
            **kwargs,
            "mode": "train",
        }  # highest priority args on the right

        self.trainer = (trainer or self._smart_load("trainer"))(
            overrides=args, _callbacks=self.callbacks
        )
        self.trainer.set_multitask_model(self.model)
        LOGGER.info(f"Training '{self.task}' model. Config: {dict(self.trainer.args)}")

        # Trainer training
        self.trainer.multitask_train(
            world_size=1, verbose=True
        )

        # Update model and cfg after training
        if RANK in {-1, 0}:
            ckpt = (
                self.trainer.best if self.trainer.best.exists() else self.trainer.last
            )
            self.model, _ = attempt_load_one_weight(ckpt)
            self.overrides = self.model.args
            self.metrics = getattr(
                self.trainer.validator, "metrics", None
            )
        return self.metrics
