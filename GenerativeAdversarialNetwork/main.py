"""
This example is based on a example given in
[Foster, D., & Friston, K. J. (2023). Generative Deep Learning: Teaching Machines to Paint, Write, Compose, and Play].

The author originally used Keras and Tensorflow however this example adepted the concept to pytorch.

!IMPORTANT!
This model requires a dataset from [1] to work. Download the dataset and copy the dataset folder into the
folder this script resides in.

[1] https://www.kaggle.com/datasets/joosthazelzet/lego-brick-images



This script trains a Deep Convolutional Generative Adversarial Network. The goal is it to pit
two models, a generator and a descriminator, against each other. The generator tries to generate
images that the discriminator detects as real. The discriminator then tries to detect all the fake
images.


"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms
import torch.nn as nn
import datasets
from torchsummary import summary
from tqdm import tqdm

from model import DCGAN

import os

from funcs import train, eval, collate_fn


if __name__ == "__main__":
    epoch = 100
    stop_threshold = 10

    model_path = "gan.pt"

    model = DCGAN()

    if os.path.exists(model_path):
        print("Loaded model")
        model.load_state_dict(torch.load(model_path, weights_only=True))

    loss_fn = nn.BCELoss()

    optimizer = {
        "discriminator": torch.optim.Adam(
            model.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)
        ),
        "generator": torch.optim.Adam(
            model.generator.parameters(), lr=0.0002, betas=(0.5, 0.999)
        ),
    }

    summary(model, (100, 1, 1), device="cpu")

    # Prepare data
    dataset = datasets.load_dataset("./GenerativeAdversarialNetwork/dataset")
    dataset = dataset["train"].train_test_split(0.1)

    data_transform = {
        "train": transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize((64, 64)),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize([0.5], [0.5], inplace=True),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize((64, 64)),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize([0.5], [0.5], inplace=True),
            ]
        ),
    }

    dataset["train"].set_transform(data_transform["train"])
    dataset["test"].set_transform(data_transform["test"])

    print(dataset)

    train_dataloader = DataLoader(
        dataset["train"],
        collate_fn=collate_fn,
        num_workers=4,
        batch_size=64,
        shuffle=True,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )
    test_dataloader = DataLoader(
        dataset["test"],
        collate_fn=collate_fn,
        num_workers=4,
        batch_size=16,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )

    progress = tqdm(range(len(train_dataloader) * epoch), position=0)

    model = model.to("cuda")

    no_improvement = 0
    for _ in range(epoch):
        train(model, train_dataloader, loss_fn, optimizer, progress)

        loss = eval(model, test_dataloader, loss_fn)

        torch.save(model.state_dict(), model_path)
