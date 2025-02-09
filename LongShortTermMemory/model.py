import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    """Example Module for an Long Short-Term Memory Network"""

    def __init__(self):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(10000, 100)
        self.lstm = nn.LSTM(100, 128, batch_first=True)
        self.output = nn.Linear(128, 10000)

    def forward(self, x):
        x = self.embedding(x.long())
        x, _ = self.lstm(x)
        x = self.output(x)

        return x
