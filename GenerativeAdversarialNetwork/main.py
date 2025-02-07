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


def calc_loss(loss_fn, prediction, expectation, z_mean, z_log_var):
    if z_mean is None or z_log_var is None:
        loss = loss_fn(prediction)
    else:
        # 500 represent the beta loss of a beta-VAE
        reconstruction_loss = 500 * loss_fn(prediction, expectation)
        # kl = Lullback-Leibler
        kl_loss = torch.mean(
            -0.5
            * torch.sum(
                1 + z_log_var - torch.square(z_mean) - torch.exp(z_log_var), axis=1
            )
        )
        loss = reconstruction_loss + kl_loss
    return loss


def train(
    model: DCGAN,
    dataloader: DataLoader,
    loss_fn,
    optimizer,
    progress_bar: tqdm,
    is_variational=False,
    device="cuda",
):
    """Training function"""

    num_batches = len(dataloader)
    reporting_interval = num_batches / 5
    d_test_loss, g_test_loss = 0.0, 0.0

    # Prepare data

    model.train()
    for i, batch in enumerate(dataloader):
        data = batch["image"].to(device)
        random_latent_vectors = torch.randn(
            size=(data.size(0), 100, 1, 1), device=device
        )

        ##########
        # 1. Update Discriminator
        ##########

        model.discriminator.zero_grad()

        generated_images = model.generator(random_latent_vectors)

        real_predictions = model.discriminator(data)
        fake_predictions = model.discriminator(generated_images)

        real_labels = torch.ones_like(real_predictions, device=device)
        fake_labels = torch.zeros_like(fake_predictions, device=device)

        real_noisy_labels = real_labels - 0.1 * torch.rand_like(real_labels)
        fake_noisy_labels = fake_labels + 0.1 * torch.rand_like(fake_labels)

        d_real_loss = loss_fn(real_predictions, real_noisy_labels)
        d_fake_loss = loss_fn(fake_predictions, fake_noisy_labels)

        d_loss = (d_real_loss + d_fake_loss) / 2.0

        d_loss.backward(retain_graph=True)

        optimizer["discriminator"].step()

        ##########
        # 2. Update Generator
        ##########

        model.generator.zero_grad()

        # Again, because the model was just updated
        fake_predictions = model.discriminator(generated_images)

        # How many fake images were guesses real
        g_loss = loss_fn(fake_predictions, real_labels)

        g_loss.backward()
        optimizer["generator"].step()

        progress_bar.update(1)

        d_test_loss += d_loss.item()
        g_test_loss += g_loss.item()

        if i % reporting_interval == 0 and i != 0:
            d_print_loss = d_test_loss / reporting_interval
            g_print_loss = g_test_loss / reporting_interval
            tqdm.write(
                f"{i}/{num_batches} Current avg training loss descriminator {d_print_loss}"
            )
            tqdm.write(
                f"{i}/{num_batches} Current avg training loss generator     {g_print_loss}"
            )
            d_test_loss = 0.0
            g_test_loss = 0.0


def eval(model, dataloader: DataLoader, loss_fn, is_variational=False, device="cuda"):
    """Evaluation function."""
    num_batches = len(dataloader)

    d_test_loss, g_test_loss = 0.0, 0.0

    progress_bar = tqdm(range(num_batches), position=1)

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            data = batch["image"].to(device)
            random_latent_vectors = torch.randn(
                size=(data.size(0), 100, 1, 1), device=device
            )

            ##########
            # 1. Check Discriminator
            ##########

            generated_images = model.generator(random_latent_vectors)

            real_predictions = model.discriminator(data)
            fake_predictions = model.discriminator(generated_images)

            real_labels = torch.ones_like(real_predictions, device=device)
            fake_labels = torch.zeros_like(fake_predictions, device=device)

            d_real_loss = loss_fn(real_predictions, real_labels)
            d_fake_loss = loss_fn(fake_predictions, fake_labels)

            d_loss = (d_real_loss + d_fake_loss) / 2.0

            ##########
            # 2. Check Generator
            ##########

            # How many fake images were guesses real
            g_loss = loss_fn(fake_predictions, real_labels)

            progress_bar.update(1)
            d_test_loss += d_loss.item()
            g_test_loss += g_loss.item()

    d_print_loss = d_test_loss / num_batches
    g_print_loss = g_test_loss / num_batches

    tqdm.write(f"Avg training loss descriminator {d_print_loss}")
    tqdm.write(f"Avg training loss generator     {g_print_loss}")

    return g_print_loss


def collate_fn(batch):
    result = dict()

    result["image"] = torch.stack([x["image"] for x in batch])

    return result


def preprocess(imgs):
    imgs = (np.array(imgs["image"]).astype("float32") - 127.5) / 127.5

    return {"data": imgs}


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

    # current_best_loss = eval(model, test_dataloader, loss_fn)

    no_improvement = 0
    for _ in range(epoch):
        train(model, train_dataloader, loss_fn, optimizer, progress)

        loss = eval(model, test_dataloader, loss_fn)

        torch.save(model.state_dict(), model_path)

        # if loss < current_best_loss:
        #     tqdm.write(f"Saved new best model. {loss} vs {current_best_loss}")
        #     current_best_loss = loss
        #     no_improvement = 0
        #     torch.save(model.state_dict(), model_path)
        # else:
        #     no_improvement += 1
        #     if no_improvement == stop_threshold:
        #         tqdm.write(
        #             f"There were no improvements for {stop_threshold} iterations."
        #         )
        #         tqdm.write("Stopping Training")
        #         exit()
