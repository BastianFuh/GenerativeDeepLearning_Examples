"""
This example is based on a example given in
[Foster, D., & Friston, K. J. (2023). Generative Deep Learning: Teaching Machines to Paint, Write, Compose, and Play].

The author originally used Keras and Tensorflow. This example uses pytorch.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR
import torch.nn.functional as F
import torch.nn as nn
import datasets
from torchsummary import summary
from tqdm import tqdm

from model import VariationalAutoencoder

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
    model,
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
    test_loss = 0.0

    model.train()
    for i, batch in enumerate(dataloader):
        data = batch["data"].to(device)

        prediction, z_mean, z_log_var = model(data)
        loss = calc_loss(loss_fn, prediction, data, z_mean, z_log_var)

        test_loss += loss.item()

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        progress_bar.update(1)

        if i % reporting_interval == 0 and i != 0:
            print_loss = test_loss / reporting_interval
            tqdm.write(f"{i}/{num_batches} Current avg training loss {print_loss}")
            test_loss = 0.0


def eval(model, dataloader: DataLoader, loss_fn, is_variational=False, device="cuda"):
    """Evaluation function."""
    num_batches = len(dataloader)

    test_loss = 0.0

    progress_bar = tqdm(range(num_batches), position=1)

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            data = batch["data"].to(device)

            prediction, z_mean, z_log_var = model(data)
            loss = calc_loss(loss_fn, prediction, data, z_mean, z_log_var)

            test_loss += loss.item()

            progress_bar.update(1)

    test_loss /= num_batches
    tqdm.write(f"Avg evaluation loss: {test_loss:>8f}")

    return test_loss


def collate_fn(batch):
    result = dict()

    result["data"] = torch.tensor([[x["data"]] for x in batch], dtype=torch.float32)

    return result


def preprocess(imgs):
    imgs = np.array(imgs["image"]).astype("float32") / 255.0
    imgs = np.pad(imgs, ((2, 2), (2, 2)), constant_values=0.0)

    return {"data": imgs}


EMBEDDING_SIZE = 3
USE_MULTI_VARIATIONAL = True

if __name__ == "__main__":
    epoch = 100
    stop_threshold = 10

    if USE_MULTI_VARIATIONAL:
        model_path = f"vae_{EMBEDDING_SIZE}.pt"
    else:
        model_path = f"ae_{EMBEDDING_SIZE}.pt"

    model = VariationalAutoencoder(USE_MULTI_VARIATIONAL, embedding_size=EMBEDDING_SIZE)

    if os.path.exists(model_path):
        print("Loaded model")
        model.load_state_dict(torch.load(model_path, weights_only=True))

    mseloss = nn.MSELoss()

    # RMSE
    def loss_fn(x, y):
        return torch.sqrt(mseloss(x, y))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = LinearLR(optimizer, total_iters=epoch)

    summary(model, (1, 32, 32), device="cpu")

    # Prepare data
    dataset = datasets.load_dataset("fashion_mnist")

    dataset = dataset.remove_columns("label")
    dataset = dataset.map(preprocess)

    dataset = dataset.remove_columns("image")

    print(dataset)

    train_dataloader = DataLoader(
        dataset["train"],
        collate_fn=collate_fn,
        num_workers=4,
        batch_size=32,
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

    current_best_loss = eval(model, test_dataloader, loss_fn, USE_MULTI_VARIATIONAL)

    no_improvement = 0
    for _ in range(epoch):
        train(
            model, train_dataloader, loss_fn, optimizer, progress, USE_MULTI_VARIATIONAL
        )

        loss = eval(model, test_dataloader, loss_fn, USE_MULTI_VARIATIONAL)
        scheduler.step()

        if loss < current_best_loss:
            tqdm.write(f"Saved new best model. {loss} vs {current_best_loss}")
            current_best_loss = loss
            no_improvement = 0
            torch.save(model.state_dict(), model_path)
        else:
            no_improvement += 1
            if no_improvement == stop_threshold:
                tqdm.write(
                    f"There were no improvements for {stop_threshold} iterations."
                )
                tqdm.write("Stopping Training")
                exit()
