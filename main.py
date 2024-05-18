import torch

from hawkeye.naive.model import HOLOH
from hawkeye.utils import AGENT_CFG_DIR, MODELS_DIR, DATA_DIR

# Initialize the agent
device = torch.device("mps")
holoh_agent = HOLOH(config=AGENT_CFG_DIR, device=device)\
    .load(model_weights_paths=MODELS_DIR / "holoh")

# Perform tracking with the agent
holoh_agent.run(video_or_stream=0)  # DATA_DIR / "test.mp4"

