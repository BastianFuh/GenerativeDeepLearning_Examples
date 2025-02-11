import torch.nn as nn
import torch.nn.functional as F


NUM_CLASSES = 10000


class LSTM(nn.Module):
    """Example Module for an Long Short-Term Memory Network"""

    def __init__(self, inference=True):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(NUM_CLASSES, 100)
        self.lstm = nn.LSTM(100, 128, batch_first=True, bias=False)
        self.output = nn.Linear(128, NUM_CLASSES)
        self.inference = inference

    def forward(self, x):
        x = self.embedding(x.long())
        x, _ = self.lstm(x)

        # If used in inference select last element in the sequence
        if self.inference:
            # Check if batch or not
            if len(x.shape) == 3:
                x = x[:, -1, :]
            else:
                x = x[-1, :]

        x = self.output(x)

        if self.inference:
            x = F.softmax(x, dim=-1)

        return x
