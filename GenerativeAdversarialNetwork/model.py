import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    """Example Module for an Discriminator. Its goal is it to detect the fake images
    of the generator."""

    def __init__(self):
        super(Discriminator, self).__init__()

        # 64 x 64
        self.conv1 = nn.Conv2d(1, 64, 4, stride=2, padding=1, bias=False)
        # 32 x 32
        self.conv2 = nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False)
        self.batch1 = nn.BatchNorm2d(128, momentum=0.9)
        # 16 x 16
        self.conv3 = nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False)
        self.batch2 = nn.BatchNorm2d(256, momentum=0.9)
        # 8 x 8
        self.conv4 = nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=False)
        self.batch3 = nn.BatchNorm2d(512, momentum=0.9)
        # 4 x 4
        self.conv5 = nn.Conv2d(512, 1, 4, stride=1, padding="valid")

        self.flat = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu_(x, 0.2)
        x = F.dropout(x, 0.3)

        x = self.conv2(x)
        x = self.batch1(x)
        x = F.leaky_relu_(x, 0.2)
        x = F.dropout(x, 0.3)

        x = self.conv3(x)
        x = self.batch2(x)
        x = F.leaky_relu_(x, 0.2)
        x = F.dropout(x, 0.3)

        x = self.conv4(x)
        x = self.batch3(x)
        x = F.leaky_relu_(x, 0.2)
        x = F.dropout(x, 0.3)

        x = self.conv5(x)
        x = F.sigmoid(x)

        x = self.flat(x)

        return x


class Generator(nn.Module):
    """Example Module for a Generator used in an GAN. It generates images
    which should be accepted as real by the descriminator."""

    def __init__(self):
        super(Generator, self).__init__()

        # 1x1
        self.convtrans1 = nn.ConvTranspose2d(
            100, 512, 4, 1, padding=0, output_padding=0, bias=False
        )
        self.batch1 = nn.BatchNorm2d(512, momentum=0.9)
        # 4x4
        self.convtrans2 = nn.ConvTranspose2d(
            512, 256, 4, 2, padding=1, output_padding=0, bias=False
        )
        self.batch2 = nn.BatchNorm2d(256, momentum=0.9)
        # 8x8
        self.convtrans3 = nn.ConvTranspose2d(
            256, 128, 4, 2, padding=1, output_padding=0, bias=False
        )
        self.batch3 = nn.BatchNorm2d(128, momentum=0.9)
        # 16x16
        self.convtrans4 = nn.ConvTranspose2d(
            128, 64, 4, 2, padding=1, output_padding=0, bias=False
        )
        self.batch4 = nn.BatchNorm2d(64, momentum=0.9)
        # 32x32
        self.convtrans5 = nn.ConvTranspose2d(
            64, 1, 4, 2, padding=1, output_padding=0, bias=False
        )
        # 64x64

    def forward(self, x):
        x = x.view((-1, 100, 1, 1))

        x = self.convtrans1(x)
        x = self.batch1(x)
        x = F.leaky_relu_(x, 0.2)

        x = self.convtrans2(x)
        x = self.batch2(x)
        x = F.leaky_relu_(x, 0.2)

        x = self.convtrans3(x)
        x = self.batch3(x)
        x = F.leaky_relu_(x, 0.2)

        x = self.convtrans4(x)
        x = self.batch4(x)
        x = F.leaky_relu_(x, 0.2)

        x = self.convtrans5(x)
        x = F.tanh(x)

        return x


class DCGAN(nn.Module):
    """Example Module for an Deep Convolutional Generative Adverserial Network"""

    def __init__(self):
        super(DCGAN, self).__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()

    def forward(self, x):
        x = self.generator(x)
        x = self.discriminator(x)

        return x
