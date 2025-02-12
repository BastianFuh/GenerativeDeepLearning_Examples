import torch.nn as nn
import torch.optim.swa_utils as swa_utils

from funcs import cosine_diffusion_schedule as cosine_diffusion_schedule


class UNet(nn.Module):
    """Example Module for an UNet Network"""

    def __init__(self):
        super(UNet, self).__init__()


class DiffusionModel(nn.Module):
    """Example Module for an Diffusion Network"""

    def __init__(self):
        super(DiffusionModel, self).__init__()
        self.normalizer = UNet()
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

    def forward(self, x):
        return x
