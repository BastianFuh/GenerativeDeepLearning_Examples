import torch.nn as nn
import torch.nn.functional as F
import numpy as np

PRE_EMBEDDING_SIZE = 4 * 4 * 128
EMBEDDING_SIZE = 2


class VariationalAutoencoderEncoder(nn.Module):
    """Example Module for an Encoder in a VariationalAutoencoder"""

    def __init__(self):
        super(VariationalAutoencoderEncoder, self).__init__()
        # ENCODER
        # 32x32
        self.conv1 = nn.Conv2d(1, 32, (3, 3), stride=2, padding=(1, 1))
        # 16x16
        self.conv2 = nn.Conv2d(32, 64, (3, 3), stride=2, padding=(1, 1))
        # 8x8
        self.conv3 = nn.Conv2d(64, 128, (3, 3), stride=2, padding=(1, 1))

        self.flat = nn.Flatten()
        self.embed = nn.Linear(PRE_EMBEDDING_SIZE, EMBEDDING_SIZE)

    def forward(self, x):
        # ENCODER
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flat(x)
        x = self.embed(x)

        return x


class VariationalAutoencoderDecoder(nn.Module):
    """Example Module for an Decoder in a VariationalAutoencoder"""

    def __init__(self):
        super(VariationalAutoencoderDecoder, self).__init__()
        self.linear2 = nn.Linear(EMBEDDING_SIZE, PRE_EMBEDDING_SIZE)

        # 4x4
        self.convtrans1 = nn.ConvTranspose2d(
            128, 128, (3, 3), stride=2, padding=1, output_padding=1
        )
        # 8x8
        self.convtrans2 = nn.ConvTranspose2d(
            128, 64, (3, 3), stride=2, padding=1, output_padding=1
        )
        # 16x16
        self.convtrans3 = nn.ConvTranspose2d(
            64, 32, (3, 3), stride=2, padding=1, output_padding=1
        )
        # 32x32
        self.conv4 = nn.Conv2d(32, 1, (3, 3), stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.linear2(x))
        x = x.view((-1, 128, 4, 4))
        x = F.relu(self.convtrans1(x))
        x = F.relu(self.convtrans2(x))
        x = F.relu(self.convtrans3(x))
        x = F.sigmoid(self.conv4(x))

        return x


class VariationalAutoencoder(nn.Module):
    """Example Module for an VariationalAutoencoder"""

    def __init__(self):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalAutoencoderEncoder()
        self.decoder = VariationalAutoencoderDecoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x
