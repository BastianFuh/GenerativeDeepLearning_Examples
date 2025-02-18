import torch.nn as nn
import torch.nn.functional as F


NUM_CLASSES = 10000


class GPT(nn.Module):
    """Example Module for an Generative pre-trained transformer Network"""

    def __init__(self, inference=True):
        super(GPT, self).__init__()

    def forward(self, x):
        return x
