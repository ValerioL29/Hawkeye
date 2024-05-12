import os
from pathlib import Path

import logging
from rich.logging import RichHandler

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("rich")

# Set up directories
# Directory paths
ROOT_DIR = Path(os.getcwd())

# Asset directory for storing images, videos, etc.
ASSETS_DIR = ROOT_DIR / "assets"
os.makedirs(ASSETS_DIR, exist_ok=True)

# Data directory for storing datasets
DATA_DIR = ROOT_DIR / "datasets"
os.makedirs(DATA_DIR, exist_ok=True)

# Checkpoints directory for storing model weights
MODELS_DIR = ROOT_DIR / "checkpoints"
os.makedirs(MODELS_DIR, exist_ok=True)

# Outputs directory for storing results, logs, etc.
OUTPUTS_DIR = ROOT_DIR / "outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)
