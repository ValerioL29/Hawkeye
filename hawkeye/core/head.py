import torch
import torch.nn as nn
from ultralytics.nn.tasks import Detect, Segment


class MultiTaskHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.detect = Detect.forward
        self.segment = Segment.forward

    def forward(self, x):
        x = self.model(x)
        return self.detect(x), self.segment(x)
