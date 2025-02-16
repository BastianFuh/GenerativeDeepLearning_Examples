import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.swa_utils as swa_utils

from funcs import cosine_diffusion_schedule, sinusoidal_embedding


class ResidualBlock(nn.Module):
    def __init__(self, output_channels, scale_residual=False):
        super(ResidualBlock, self).__init__()

        self.scale_residual = scale_residual

        if self.scale_residual:
            self.scale = nn.LazyConv2d(output_channels, 1)

        self.batch_norm = nn.LazyBatchNorm2d(output_channels, affine=False)

        self.conv_1 = nn.LazyConv2d(output_channels, 3, padding="same")
        self.conv_2 = nn.Conv2d(output_channels, output_channels, 3, padding="same")

    def forward(self, x):
        if self.scale_residual:
            residual = self.scale(x)
        else:
            residual = x

        x = self.batch_norm(x)
        x = F.silu(self.conv_1(x))
        x = self.conv_2(x)

        x = x + residual

        return x


class DownBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(DownBlock, self).__init__()

        self.residual_1 = ResidualBlock(output_channels, True)
        self.residual_2 = ResidualBlock(output_channels, output_channels)

        self.pool = nn.AvgPool2d(2)

    def forward(self, x, skips=list()):
        x = self.residual_1(x)
        skips.append(x)
        x = self.residual_2(x)
        skips.append(x)

        x = self.pool(x)

        return x


class UpBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(UpBlock, self).__init__()

        self.residual_1 = ResidualBlock(output_channels, True)
        self.residual_2 = ResidualBlock(output_channels, True)

    def forward(self, x, skips=list()):
        # Upsampling
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        x = torch.concat([x, skips.pop()], dim=1)
        x = self.residual_1(x)

        x = torch.concat([x, skips.pop()], dim=1)
        x = self.residual_2(x)

        return x


class UNet(nn.Module):
    """Example Module for an UNet Network"""

    def __init__(self):
        super(UNet, self).__init__()

        self.input_conv = nn.Conv2d(3, 32, 1)

        # Inputs is doubled because of the concatination of the
        # noise values
        self.down_block_1 = DownBlock(64, 32)

        self.down_block_2 = DownBlock(32, 64)
        self.down_block_3 = DownBlock(64, 96)

        self.residual_1 = ResidualBlock(128, True)
        self.residual_2 = ResidualBlock(128)

        self.up_block_1 = UpBlock(128, 96)
        self.up_block_2 = UpBlock(96, 64)
        self.up_block_3 = UpBlock(64, 32)

        self.output_conv = nn.Conv2d(32, 3, 1)

        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def forward(self, x, noise_variances=None):
        x = self.input_conv(x)

        # This is mainly used so that the architecture of the model can be shown via
        # summary. For training and inference the noise_variance variable should be set
        if noise_variances is None:
            noise_variances = torch.Tensor(x.shape[0], 1, 1, 1).to(x.device)

        noise_embedding = sinusoidal_embedding(noise_variances)

        noise_embedding = F.interpolate(
            noise_embedding, scale_factor=64, mode="nearest"
        )

        x = torch.concat([x, noise_embedding], dim=1)

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

    def __init__(self, in_training=False):
        super(DiffusionModel, self).__init__()
        self.network = UNet()
        self.inference = cosine_diffusion_schedule
        self.ema_network = None
        self.in_training = in_training

    def create_ema(self):
        """Creates the ema network.
        This is not done in the constructor because the model uses lazy modules. Without
        running the model atleast once the lazy modules do not have an assosiated size
        and therfore can not be deep copied to a new model."""
        self.ema_network = swa_utils.AveragedModel(self.network)

    def update_ema(self):
        if self.ema_network is None:
            self.create_ema()

        for param, ema_param in zip(self.parameters(), self.ema_network.parameters()):
            ema_param.data.mul_(0.99).add_(0.01 * param.data)

    def denoise(self, noisy_images, noise_rates, signal_rates):
        if self.in_training:
            network = self.network
        else:
            network = self.ema_network

        pred_noises = network(noisy_images, noise_rates**2)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images

    def forward(self, x):
        if isinstance(x, list):
            x = torch.concat(x, dim=1)

        noises = torch.normal(0, 1, size=x.shape).to(x.device)
        batch_size = x.shape[0]

        diffusion_times = torch.rand(size=(batch_size, 1, 1, 1)).to(x.device)

        noise_rates, signal_rates = cosine_diffusion_schedule(diffusion_times)

        noisy_images = signal_rates * x + noise_rates * noises

        if self.in_training:
            return self.denoise(noisy_images, noise_rates, signal_rates), noises
        else:
            return self.denoise(x, noise_rates, signal_rates)
