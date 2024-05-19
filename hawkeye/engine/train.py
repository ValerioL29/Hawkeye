# Ultralytics YOLO 🚀, AGPL-3.0 license

import math
import random

import torch
from torch import nn
from ultralytics.data import build_dataloader
from ultralytics.data.build import InfiniteDataLoader
from ultralytics.utils import RANK
from ultralytics.utils.torch_utils import torch_distributed_zero_first

from hawkeye.core.task import MultitaskModel
from hawkeye.data.dataset import build_holo_dataset
from hawkeye.engine.base import BaseTrainer
from hawkeye.utils import logger as LOGGER


class MultitaskTrainer(BaseTrainer):
    """
    A class extending the BaseTrainer class for training based on a dedicated model.
    """
    multitask_model: MultitaskModel
    train_loader: dict[str, InfiniteDataLoader]
    test_loader: dict[str, InfiniteDataLoader]

    def set_multitask_model(self, model: MultitaskModel):
        """Sets the model for the trainer."""
        _ = model.move_to_device(device=self.device)
        self.multitask_model = model
        return self

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = torch.tensor([
          task_object['stride'].max()
          for _, task_object in self.multitask_model.task_layers.items()
        ]).max().item()  # grid size

        return {
            task_name: build_holo_dataset(
                self.args,
                img_path,
                batch,
                self.data,
                task=task_name,
                mode=mode,
                rect=mode == "val",
                stride=int(max(gs, 32)),
            )
            for task_name in self.multitask_model.task_layers.keys()
        }

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Construct and return dataloader."""
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        # init dataset *.cache only once if DDP
        with torch_distributed_zero_first(rank):
            datasets = self.build_dataset(dataset_path, mode, batch_size)

        def ensemble_dataloader(dataset):
            shuffle = mode == "train"
            if getattr(dataset, "rect", False) and shuffle:
                LOGGER.warning(
                    "WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False"
                )
                shuffle = False
            workers = self.args.workers if mode == "train" else self.args.workers * 2
            return build_dataloader(
                dataset, batch_size, workers, shuffle, rank
            )  # return dataloader

        return {
            task: ensemble_dataloader(dataset)
            for task, dataset in datasets.items()
        }

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        if self.args.multi_scale:
            imgs = batch["img"]
            sz = (
                random.randrange(
                    self.args.imgsz * 0.5, self.args.imgsz * 1.5 + self.stride
                )
                // self.stride
                * self.stride
            )  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride
                    for x in imgs.shape[2:]
                ]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(
                    imgs, size=ns, mode="bilinear", align_corners=False
                )
            batch["img"] = imgs
        return batch

    def set_model_attributes(self):
        """Nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)."""
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = self.data["nc"]  # attach number of classes to model
        self.model.names = self.data["names"]  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        raise NotImplementedError("'get_model()' function is not implemented in MultitaskTrainer. Please use "
                                  "'model.load()' to load a pre-trained model.")

    # def get_dataset(self):
    #     """Return a YOLO dataset."""
    #     raise NotImplementedError("'get_dataset()' function is not implemented in MultitaskTrainer. Please use "
    #                               "'build_dataset()' to build a dataset.")

    def prepare_dataloaders(self, world_size: int = 1, verbose: bool = True):
        """Prepare dataloaders."""
        train_img_path, test_img_path = self.get_dataset()
        batch_size = self.batch_size // max(world_size, 1)
        train_loader = self.get_dataloader(
            train_img_path, batch_size=batch_size, rank=RANK, mode="train"
        )
        test_loader = self.get_dataloader(
            test_img_path,
            batch_size=batch_size * 2,  # For Detect and Segment
            rank=-1,
            mode="val",
        )
        return train_loader, test_loader

    def multitask_train(self, world_size: int = 1, verbose: bool = True):
        """Trains the YOLO model."""

        # TODO: Multitask Setup Dataloaders
        self.train_loader, self.test_loader = self.prepare_dataloaders(world_size, verbose)
        LOGGER.info(f"Train: {len(self.train_loader)} | Test: {len(self.test_loader)}")
        pbar = enumerate(self.train_loader)

        sample_batch = next(iter(self.train_loader))
        preprocessed_batch: dict = self.preprocess_batch(sample_batch)
        # LOGGER.info(preprocessed_batch.keys())
        # LOGGER.info(f"Element types: {[type(v) for v in preprocessed_batch.values()]}")
        # LOGGER.info(self.multitask_model(preprocessed_batch))

        # TODO: Multitask Setup Optimizers

        # TODO: Multitask Forward

        # TODO: Multitask Backward

        # TODO: Multitask Validation

        return

    def get_validator(self):
        """Returns a MultitaskValidator for HOLO model validation."""
        raise NotImplementedError("'get_validator()' function is not implemented in MultitaskTrainer.")

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Returns a loss dict with labelled training loss items tensor.

        Not needed for classification but necessary for segmentation & detection
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [
                round(float(x), 5) for x in loss_items
            ]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        """Returns a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def plot_training_samples(self, batch, ni):
        """Plots training samples with their annotations."""
        raise NotImplementedError("'plot_training_samples()' function is not implemented in MultitaskTrainer.")

    def plot_metrics(self):
        """Plots metrics from a CSV file."""
        raise NotImplementedError("'plot_metrics()' function is not implemented in MultitaskTrainer.")

    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        raise NotImplementedError("'plot_training_labels()' function is not implemented in MultitaskTrainer.")
