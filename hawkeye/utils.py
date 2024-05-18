import os
import json
import logging
from pathlib import Path
from ultralytics.utils import TQDM
from rich.logging import RichHandler

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

# Agent config directory
AGENT_CFG_DIR = ROOT_DIR / "cfg"
os.makedirs(AGENT_CFG_DIR, exist_ok=True)

# Load config file
try:
    with open(ROOT_DIR / "config.json", "r") as f:
        config = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError("Config file not found. Please create a 'config.json' file in the root directory.")

# Set up logging
logging.basicConfig(
    level=logging.DEBUG if config.get("log", "INFO") == "DEBUG" else logging.INFO,
    format="%(message)s",  # %(asctime)s [%(levelname)s] %(message)s
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("rich")
