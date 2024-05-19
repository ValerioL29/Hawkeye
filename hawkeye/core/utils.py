from pathlib import Path
from typing import Union

from ultralytics.nn import guess_model_scale
from ultralytics.utils import yaml_load
from ultralytics.utils.checks import check_yaml
from ultralytics.utils.torch_utils import get_num_params, get_num_gradients, get_flops

from hawkeye.utils import logger as LOGGER


def load_yaml_model_config(path: Union[str, Path]):
    """Load a YOLOv8 model from a YAML file."""
    import re

    # Path check
    if isinstance(path, str):
        path = Path(path)

    if path.stem in (f"yolov{d}{x}6" for x in "nsmlx" for d in (5, 8)):
        new_stem = re.sub(r"(\d+)([nslmx])6(.+)?$", r"\1\2-p6\3", path.stem)
        LOGGER.warning(
            f"WARNING ⚠️ Ultralytics YOLO P6 models now use -p6 suffix. Renaming {path.stem} to {new_stem}."
        )
        path = path.with_name(new_stem + path.suffix)

    # Unified path
    unified_path = re.sub(
        r"(\d+)([nslmx])(.+)?$", r"\1\3", str(path)
    )  # i.e. yolov8x.yaml -> yolov8.yaml
    LOGGER.debug(f"Loading {unified_path}...")
    yaml_file_path = check_yaml(file=unified_path, hard=False) or check_yaml(file=path, hard=True)

    # Load YAML
    LOGGER.debug(f"Loading {yaml_file_path}...")
    d = yaml_load(file=yaml_file_path)  # model dict
    d["scale"] = guess_model_scale(path)
    d["yaml_file"] = str(path)
    return d


def model_info(model, detailed=False, verbose=True, imgsz=640):
    """
    Model information.

    imgsz may be int or list, i.e. imgsz=640 or imgsz=[640, 320].
    """
    if not verbose:
        return
    n_p = get_num_params(model)  # number of parameters
    n_g = get_num_gradients(model)  # number of gradients
    n_l = len(list(model.modules()))  # number of layers
    if detailed:
        LOGGER.info(
            f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}"
        )
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace("module_list.", "")
            LOGGER.info(
                "%5g %40s %9s %12g %20s %10.3g %10.3g %10s"
                % (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std(), p.dtype)
            )

    flops = get_flops(model, imgsz)
    fused = " (fused)" if getattr(model, "is_fused", lambda: False)() else ""
    fs = f", {flops:.1f} GFLOPs" if flops else ""
    yaml_file = getattr(model, "yaml_file", "") or getattr(model, "yaml", {}).get("yaml_file", "")
    model_name = Path(yaml_file).stem.replace("yolo", "YOLO") or "Model"
    LOGGER.info(f"{model_name} summary{fused}: {n_l} layers, {n_p} parameters, {n_g} gradients{fs}")
    return n_l, n_p, n_g, flops
