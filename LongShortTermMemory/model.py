import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    """Example Module for an Long Short-Term Memory Network"""

    def __init__(self):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding()
        self.lstm = nn.LSTM()

    def forward(self, x):
        x = self.embedding(x)
        x = self.lstm(x)

        return x
