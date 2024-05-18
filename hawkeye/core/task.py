import contextlib
from pathlib import Path
from typing import Union, OrderedDict

import torch
from torch import nn
from ultralytics.nn import BaseModel
from ultralytics.nn.modules import (
    AIFI,
    C1,
    C2,
    C3,
    C3TR,
    OBB,
    SPP,
    SPPELAN,
    SPPF,
    ADown,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C3Ghost,
    C3x,
    CBFuse,
    CBLinear,
    Classify,
    Concat,
    Conv,
    ConvTranspose,
    Detect,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostBottleneck,
    GhostConv,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    Pose,
    RepC3,
    RepNCSPELAN4,
    ResNetLayer,
    RTDETRDecoder,
    Segment,
    WorldDetect,
)
from ultralytics.utils.torch_utils import (
    make_divisible, initialize_weights, intersect_dicts,
)

from hawkeye.core.loss import MultitaskDetectionLoss, MultitaskSegmentationLoss
from hawkeye.core.utils import load_yaml_model_config, model_info
from hawkeye.utils import logger as LOGGER


def parse_model(d, ch, verbose=True, use_extra_head: bool = False):  # model_dict, input_channels(3)
    """Parse a YOLO model.yaml dictionary into a PyTorch model."""
    import ast

    # Args
    max_channels = float("inf")
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    if scales:
        scale = d.get("scale")
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        if verbose:
            LOGGER.info(f"'activation:' {act}")  # print

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        m = getattr(torch.nn, m[3:]) if "nn." in m else globals()[m]  # get module
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)

        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        if m in {
            Classify,
            Conv,
            ConvTranspose,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            DWConv,
            Focus,
            BottleneckCSP,
            C1,
            C2,
            C2f,
            RepNCSPELAN4,
            ADown,
            SPPELAN,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
            RepC3,
        }:
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            if m is C2fAttn:
                args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)  # embed channels
                args[2] = int(
                    max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2]
                )  # num heads

            args = [c1, c2, *args[1:]]
            if m in {BottleneckCSP, C1, C2, C2f, C2fAttn, C3, C3TR, C3Ghost, C3x, RepC3}:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is AIFI:
            args = [ch[f], *args]
        elif m in {HGStem, HGBlock}:
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)  # number of repeats
                n = 1
        elif m is ResNetLayer:
            c2 = args[1] if args[3] else args[1] * 4
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in {Detect, WorldDetect, Segment, Pose, OBB, ImagePoolingAttn}:
            args.append([ch[x] for x in f])
            if m is Segment:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
        elif m is RTDETRDecoder:  # special case, channels arg must be passed in index 1
            args.insert(1, [ch[x] for x in f])
        elif m is CBLinear:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        elif m is CBFuse:
            c2 = ch[f[-1]]
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        m.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        if verbose:
            LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}{m.np:10.0f}  {t:<45}{str(args):<30}")  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    # End layer transform
    return nn.Sequential(*layers), sorted(save), ch


def parse_task_output_layer(
        d, ch, head_module: nn.Module,
        width: float, inplace: bool = True,
        module_idx: int = 23, verbose: bool = True
):  # model_dict, input_channels(3)
    """Parse a YOLO model.yaml dictionary into a PyTorch model."""
    import ast

    # Args
    max_channels = float("inf")
    nc = d['nc']

    save, c2 = [], ch[-1]  # layers, savelist, ch out
    (f, n, m, args) = d["head"]  # from, number, module, args
    i = module_idx  # Output layer index is 23 in the model
    m = Detect if m == "Detect" else Segment  # get module
    for j, a in enumerate(args):
        if isinstance(a, str):
            with contextlib.suppress(ValueError):
                args[j] = locals()[a] if a in locals() else ast.literal_eval(a)

    args.append([ch[x] for x in f])
    if m is Segment:
        args[2] = make_divisible(min(args[2], max_channels) * width, 8)

    n_ = n  # depth gain
    m_ = m(*args)  # module
    t = str(m)[8:-2].replace("__main__.", "")  # module type
    m.np = sum(x.numel() for x in m_.parameters())  # number params
    m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type

    if verbose:
        LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}{m.np:10.0f}  {t:<45}{str(args):<30}")  # print# print
    save.extend(
        x % i for x in ([f] if isinstance(f, int) else f) if x != -1
    )  # append to savelist

    layers = [
        (str(i + 10), module)
        for i, module in enumerate([*head_module, m_])
    ]
    return nn.Sequential(OrderedDict[str, nn.Module](layers)), sorted(save)


class MultitaskModel(BaseModel):
    device: torch.device
    criterion: dict

    def __init__(
            self,
            cfg: Union[str, Path, dict] = "yolov8n.yaml",
            ch: int = 3, nc: int = None,
            verbose: bool = True
    ):  # model, input channels, number of classes
        """Initialize the YOLOv8 detection model with the given config and parameters."""
        super().__init__()

        # Load model config
        #
        self.yaml = cfg if isinstance(cfg, dict) else load_yaml_model_config(path=cfg)  # cfg dict
        self.tasks_dict: dict = self.yaml['tasks']
        self.inplace = self.yaml.get("inplace", True)
        self.extra_head = self.yaml.get("extra_head", False)

        # Define model
        #
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override YAML value
        # Load Backbone and head
        original_model, self.save, original_ch = parse_model(
            self.yaml, ch=ch, verbose=verbose
        )
        self.model = original_model[:10]  # 10 is the number of layers in the backbone
        head = original_model[10:]
        # _ = initialize_weights(self.backbone)
        # _ = initialize_weights(self.head)

        # Task specific heads
        width = self.yaml.get("width_multiple", 1.0)
        # detect, drivable, lane
        self.task_layers = dict()
        for i, (task_name, task_obj) in enumerate(self.tasks_dict.items()):
            # Parse task output layer
            task_fc, task_save = parse_task_output_layer(
                task_obj, head_module=head, ch=original_ch, module_idx=23 + i,
                width=width, verbose=verbose
            )
            names = {i: f"{i}" for i in range(task_obj['nc'])}

            # Save task layer
            self.task_layers[task_name] = dict(
                fc=task_fc, names=names, stride=task_fc[-1].stride
            )
            self.save.extend(task_save)
        # end for
        self.save = sorted(list(set(self.save)))

        # TODO: Implement the extra head arch
        # Check if we use the "extra head" arch
        # if self.extra_head:
        #     for _, task_layer_obj in task_layers.items():
        #         task_layer_obj['fc'] = nn.Sequential(*self.head, task_layer_obj['fc'])
        #         _ = initialize_weights(task_layer_obj['fc'])
        # else:

        # Build strides for Detect() and Segment() modules
        for task_name, task_layer_obj in self.task_layers.items():
            def fwd(x):
                """Forward pass for stride calculation."""
                fwd_ret = self.forward(x)  # Dict with task_name as key
                task_ret = fwd_ret[task_name]
                return task_ret[0] if task_name != "detect" else task_ret

            s = 256  # 2x min stride
            task_head = task_layer_obj['fc'][-1]
            task_head.inplace = self.inplace
            task_head.stride = torch.tensor(
                [s / x.shape[-2] for x in fwd(torch.zeros(1, ch, s, s))]
            )  # forward
            task_head.bias_init()  # only run once

        # Initialize the model weights
        _ = initialize_weights(model=self.model)
        for _, task_layer_obj in self.task_layers.items():
            _ = initialize_weights(task_layer_obj['fc'])

        # Done initializing the model, log the model info if set to verbose
        if verbose:
            _ = self.info(detailed=False, verbose=True)

    def info(self, detailed=False, verbose=True, imgsz=1280):
        """Print model information."""
        full_model = nn.Sequential(self.model, *[task['fc'] for task in self.task_layers.values()])
        return model_info(full_model, detailed=False, verbose=True, imgsz=1280)

    def load(self, weights, verbose=True):
        """
        Load the weights into the model.

        Args:
            weights (dict | torch.nn.Module): The pre-trained weights to be loaded.
            verbose (bool, optional): Whether to log the transfer progress. Defaults to True.
        """
        def load_state_dict_into_module(m, loaded_module):
            """Load the state_dict into the module."""
            local_csd = loaded_module.float().state_dict()  # checkpoint state_dict as FP32
            intersected_csd = intersect_dicts(local_csd, m.state_dict())  # intersect
            LOGGER.info([
                k for k, v in local_csd.items()
                if k not in m.state_dict() or v.shape != m.state_dict()[k].shape
            ])
            LOGGER.info([
                (v.shape, m.state_dict()[k].shape) for k, v in local_csd.items()
                if k not in m.state_dict() or v.shape != m.state_dict()[k].shape
            ])
            m.load_state_dict(intersected_csd, strict=False)  # load
            return len(intersected_csd), len(m.state_dict())

        # weights is a dictionary of task_name: TaskModel
        backbone_loaded_flag = False
        acc_csd_and_state_dict = dict(csd=0, state_dict=0, holo_layers=0)
        for task_name, task_model in weights.items():
            loaded_model = task_model.model
            if not backbone_loaded_flag:
                # Load backbone
                csd_len_bnh, state_dict_len_bnh = load_state_dict_into_module(self.model, loaded_model[:10])
                acc_csd_and_state_dict['csd'] += csd_len_bnh
                acc_csd_and_state_dict['state_dict'] += state_dict_len_bnh
                acc_csd_and_state_dict['holo_layers'] += len(self.model)
                backbone_loaded_flag = True
                LOGGER.info(f"Loaded 'backbone' weights for {csd_len_bnh}/{state_dict_len_bnh} items")
                LOGGER.info(f"Loaded 'backbone' layers: {len(self.model)}")

            # Load task head
            task_head = self.task_layers[task_name]['fc']
            csd_len_out, state_dict_len_out = load_state_dict_into_module(task_head, loaded_model[10:])
            # Update the accumulated loaded values
            acc_csd_and_state_dict['csd'] += csd_len_out
            acc_csd_and_state_dict['state_dict'] += state_dict_len_out
            acc_csd_and_state_dict['holo_layers'] += len(task_head)
            LOGGER.info(f"Loaded '{task_name}' weights for {len(task_head)} layers")
            LOGGER.info(f"Loaded '{task_name}' weights for {csd_len_out}/{state_dict_len_out} items")
        # End loading

        # Log the transfer progress
        if verbose:
            LOGGER.info(
                f"Transferred {acc_csd_and_state_dict['csd']}/{acc_csd_and_state_dict['state_dict']} items from "
                f"pretrained weights. Total layers: {acc_csd_and_state_dict['holo_layers']}"
            )
            LOGGER.info(f"Model config: {self.args}")

        return self

    def move_to_device(self, device: torch.device):
        """Move the model to the specified device."""
        # Move backbone and head
        self.device = device
        _ = self.model.to(device)
        # Move task heads
        for _, task_layer_obj in self.task_layers.items():
            task_layer_obj['fc'].to(device)
        return self

    def _predict_once(self, x, profile=False, visualize=False, embed=None) -> dict:
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool):  Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        def multi_input_fwd(input_x, module, previous_outputs):
            """Forward pass for modules with multiple inputs."""
            if module.f != -1:  # if not from previous layer
                input_x = (
                    y[module.f] if isinstance(module.f, int)
                    # from earlier layers
                    else [input_x if j == -1 else previous_outputs[j] for j in module.f]
                )
                if input_x is None:
                    raise RuntimeError(f"Invalid 'from' config: {module.f}")
            return module(input_x)  # run

        y = []  # outputs
        for m in self.model:
            x = multi_input_fwd(x, m, y)
            y.append(x if m.i in self.save else None)  # save output
        # End backbone

        # Task heads forward
        ret = dict()
        for i, (task_name, task_layer) in enumerate(self.task_layers.items()):
            yy = y.copy()  # outputs
            xx = torch.clone(x)  # inputs
            for mm in task_layer['fc']:
                xx = multi_input_fwd(xx, mm, yy)
                yy.append(xx if mm.i in self.save else None)  # save output
            # End sub-task forward

            # Save the output
            ret[task_name] = xx
        # End task heads forward

        return ret

    def init_criterion(self):
        """Initialize the loss criterion for the MultitaskModel."""
        criterion = dict()
        for task_name, task_layer_obj in self.task_layers.items():
            # Only Detection and Segmentation tasks are supported
            criterion[task_name] = MultitaskDetectionLoss(self, task_name, self.device) if task_name == "detect" \
                else MultitaskSegmentationLoss(self, task_name, self.device)
        # end for
        return criterion

    def loss(self, batch, preds=None):
        """
        Compute loss.

        Args:
            batch (dict): Batch to compute loss on
            preds (torch.Tensor | List[torch.Tensor]): Predictions.
        """
        if not hasattr(self, "criterion"):
            # Initialize criterion
            self.criterion = self.init_criterion()

        preds = self.forward(batch["img"]) if preds is None else preds

        # Compute loss for each task
        # total_loss = torch.tensor([
        #     self.criterion[task_name](preds[task_name], batch[task_name])
        #     for task_name in self.tasks_dict.keys()
        # ]).sum()
        total_loss = self.criterion['detect'](preds['detect'], batch)

        return total_loss

    def forward(self, x, *args, **kwargs):
        """
        Forward pass of the model on a single scale. Wrapper for `_forward_once` method.

        Args:
            x (torch.Tensor | dict): The input image tensor or a dict including image tensor and gt labels.

        Returns:
            (torch.Tensor): The output of the network.
        """
        if isinstance(x, dict):  # for cases of training and validating while training.
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool):  Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False.
            augment (bool): Augment image during prediction, defaults to False.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        if augment:
            return self._predict_augment(x)
        return self._predict_once(x, profile, visualize, embed)
