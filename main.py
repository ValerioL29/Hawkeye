import cv2
import torch

from hawkeye.holoh import HOLOH
from hawkeye.utils import AGENT_CFG_DIR, MODELS_DIR, DATA_DIR

# Initialize the agent
device = torch.device("mps")
holoh_agent = HOLOH(config=AGENT_CFG_DIR, device=device)\
    .load(model_weights_paths=MODELS_DIR / "holoh")

# Perform tracking with the agent
holoh_agent.run_aio(
    video_or_stream=DATA_DIR / "test.mp4",
    is_blind_spot=True
)  # "demo" / "196_1716094434.mp4"
