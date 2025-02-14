import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.swa_utils as swa_utils

from funcs import cosine_diffusion_schedule, sinusoidal_embedding


class ResidualBlock:
    def __init__(self, input_width, width):
        super(ResidualBlock, self).__init__()

        self.scale_input = not input_width == width

        if not self.scale_input:
            self.scale = nn.Conv2d(input_width, width, 1)

        self.batch_norm = nn.BatchNorm2d(width, width, affine=False)

        self.conv_1 = nn.Conv2d(width, width, 3, padding="same")
        self.conv_2 = nn.Conv2d(width, width, 3, padding="same")

    def forward(self, x):
        if self.scale_input:
            residual = self.scale(x)
        else:
            residual = x

        x = self.batch_norm(x)
        x = F.silu(self.conv_1(x))
        x = self.conv_2(x)

        x = x + residual

        return x


class DownBlock:
    def __init__(self, input_width, width, block_depth):
        super(DownBlock, self).__init__()

        self.residuals = list()
        for i in range(block_depth):
            self.residuals.append(ResidualBlock(input_width, width))

        self.pool = nn.AvgPool2d(2)

    def forward(self, x, skips=list()):
        for residual in self.residuals:
            x = residual(x)
            skips.append(x)

        x = self.pool(x)

        return x


class UpBlock:
    def __init__(self, input_width, width, block_depth):
        super(UpBlock, self).__init__()

        self.residuals = list()
        for i in range(block_depth):
            self.residuals.append(ResidualBlock(input_width, width))

    def forward(self, x, skips=list()):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        for residual in self.residuals:
            x = torch.concat([x, skips.pop()])
            x = residual(x)

        return x


class UNet(nn.Module):
    """Example Module for an UNet Network"""

    def __init__(
        self, input_width, width, scale_factor=2, block_count=3, residual_count=2
    ):
        super(UNet, self).__init__()

        self.input_conv = nn.Conv2d(3, 32, 1)

        self.down_block_1 = DownBlock(32, 32, 2)
        self.down_block_2 = DownBlock(32, 64, 2)
        self.down_block_3 = DownBlock(64, 96, 2)

        self.residual_1 = ResidualBlock(96, 128)
        self.residual_2 = ResidualBlock(128, 128)

        self.up_block_1 = UpBlock(128, 96, 2)
        self.up_block_2 = UpBlock(96, 64, 2)
        self.up_block_3 = UpBlock(64, 32, 2)

        self.output_conv = nn.Conv2d(32, 3, 1)

        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def forward(self, x, noise_variances=None):
        x = self.input_conv(x)

        # This is mainly used so that the architecture of the model can be shown via
        # summary. For training and inference the noise_variance variable should be set
        if noise_variances == None:
            noise_variance = torch.Tensor(x.shape[0], 1, 1, 1)

        noise_embedding = sinusoidal_embedding(noise_variances)
        noise_embedding = F.interpolate(
            noise_embedding, scale_factor=64, mode="nearest", align_corners=False
        )

        x = torch.concat([x, noise_embedding])

        skips = list()

        x = self.down_block_1(x, skips)
        x = self.down_block_2(x, skips)
        x = self.down_block_3(x, skips)

        x = self.residual_1(x)
        x = self.residual_2(x)

        x = self.up_block_1(x, skips)
        x = self.up_block_2(x, skips)
        x = self.up_block_3(x, skips)

        x = self.output_conv(x)

        return x


class DiffusionModel(nn.Module):
    """Example Module for an Diffusion Network"""

    def __init__(self):
        super(DiffusionModel, self).__init__()
        self.network = UNet()
        self.inference = cosine_diffusion_schedule

        self.ema_network = swa_utils.AveragedModel(self)

    def update_ema(self):
        for param, ema_param in zip(self.parameters(), self.ema.parameters()):
            ema_param.data.mul_(0.99).add_(0.01 * param.data)

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        if training:
            network = self.network
        else:
            network = self.ema_network

        pred_noises = network([noisy_images, noise_rates**2])
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images

    def forward(self, x, training=False):
        x
        x = self.denoise

        return x
