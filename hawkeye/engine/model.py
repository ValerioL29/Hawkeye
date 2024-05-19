from pathlib import Path
from typing import Union, List

import numpy as np
import torch
from torch import nn
from ultralytics.cfg import TASK2DATA, get_cfg, get_save_dir, ARGV
from ultralytics.engine.model import Model
from ultralytics.engine.results import Results
from ultralytics.nn import attempt_load_one_weight
from ultralytics.utils import checks, yaml_load, DEFAULT_CFG_DICT, RANK, ASSETS

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
            "dedicated": {
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

    def set_module_mode(self, mode: str = "eval"):
        """Sets the model to evaluation mode."""
        if mode == "eval":
            # Set backbone
            self.model.model.eval()
            # Set task heads
            for task_layer in self.model.task_layers.values():
                task_layer['fc'].eval()
        elif mode == "train":
            # Set backbone
            self.model.model.train()
            # Set task heads
            for task_layer in self.model.task_layers.values():
                task_layer['fc'].train()
        else:
            raise ValueError(f"Invalid mode: {mode}. Expected 'eval' or 'train'.")

        return self

    def predict(
        self,
        source: Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        predictor=None,
        **kwargs,
    ) -> List[Results]:
        """
        Performs predictions on the given image source using the YOLO model.

        This method facilitates the prediction process, allowing various configurations through keyword arguments.
        It supports predictions with custom predictors or the default predictor method. The method handles different
        types of image sources and can operate in a streaming mode. It also provides support for SAM-type models
        through 'prompts'.

        The method sets up a new predictor if not already present and updates its arguments with each call.
        It also issues a warning and uses default assets if the 'source' is not provided. The method determines if it
        is being called from the command line interface and adjusts its behavior accordingly, including setting defaults
        for confidence threshold and saving behavior.

        Args:
            source (str | int | PIL.Image | np.ndarray, optional): The source of the image for making predictions.
                Accepts various types, including file paths, URLs, PIL images, and numpy arrays. Defaults to ASSETS.
            stream (bool, optional): Treats the input source as a continuous stream for predictions. Defaults to False.
            predictor (BasePredictor, optional): An instance of a custom predictor class for making predictions.
                If None, the method uses a default predictor. Defaults to None.
            **kwargs (any): Additional keyword arguments for configuring the prediction process. These arguments allow
                for further customization of the prediction behavior.

        Returns:
            (List[ultralytics.engine.results.Results]): A list of prediction results, encapsulated in the Results class.

        Raises:
            AttributeError: If the predictor is not properly set up.
        """
        if source is None:
            source = ASSETS
            LOGGER.warning(f"WARNING ⚠️ 'source' is missing. Using 'source={source}'.")

        is_cli = (ARGV[0].endswith("yolo") or ARGV[0].endswith("ultralytics")) and any(
            x in ARGV for x in ("predict", "track", "mode=predict", "mode=track")
        )

        custom = {"conf": 0.25, "batch": 1, "save": is_cli, "mode": "predict"}  # method defaults
        args = {**self.overrides, **custom, **kwargs}  # highest priority args on the right

        if not self.predictor:
            self.predictor = predictor or self._smart_load("predictor")(overrides=args, _callbacks=self.callbacks)
            self.predictor.setup_model(
                model=nn.Sequential(*[
                    *self.model.model,
                    *self.model.task_layers['detect']['fc']
                ]), verbose=is_cli
            )
        else:  # only update args if predictor is already setup
            self.predictor.args = get_cfg(self.predictor.args, args)
            if "project" in args or "name" in args:
                self.predictor.save_dir = get_save_dir(self.predictor.args)

        return self.predictor(source=source, stream=stream)

    def track(
        self,
        source: Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        persist: bool = False,
        **kwargs,
    ) -> List[Results]:
        """
        Conducts object tracking on the specified input source using the registered trackers.

        This method performs object tracking using the model's predictors and optionally registered trackers. It is
        capable of handling different types of input sources such as file paths or video streams. The method supports
        customization of the tracking process through various keyword arguments. It registers trackers if they are not
        already present and optionally persists them based on the 'persist' flag.

        The method sets a default confidence threshold specifically for ByteTrack-based tracking, which requires low
        confidence predictions as input. The tracking mode is explicitly set in the keyword arguments.

        Args:
            source (str, optional): The input source for object tracking. It can be a file path, URL, or video stream.
            stream (bool, optional): Treats the input source as a continuous video stream. Defaults to False.
            persist (bool, optional): Persists the trackers between different calls to this method. Defaults to False.
            **kwargs (any): Additional keyword arguments for configuring the tracking process. These arguments allow
                for further customization of the tracking behavior.

        Returns:
            (List[ultralytics.engine.results.Results]): A list of tracking results, encapsulated in the Results class.

        Raises:
            AttributeError: If the predictor does not have registered trackers.
        """
        if not hasattr(self.predictor, "trackers"):
            from ultralytics.trackers import register_tracker

            register_tracker(self, persist)
        kwargs["conf"] = kwargs.get("conf") or 0.1  # ByteTrack-based method needs low confidence predictions as input
        kwargs["batch"] = kwargs.get("batch") or 1  # batch-size 1 for tracking in videos
        kwargs["mode"] = "track"
        return self.predict(source=source, stream=stream, **kwargs)
