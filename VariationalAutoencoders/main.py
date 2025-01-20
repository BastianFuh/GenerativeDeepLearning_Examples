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
import tqdm

from model import VariationalAutoencoder

import os


def train(
    model,
    dataloader: DataLoader,
    loss_fn,
    optimizer,
    progress_bar: tqdm.tqdm,
    device="cuda",
):
    num_batches = len(dataloader)
    reporting_interval = num_batches / 5
    model.train()
    for i, batch in enumerate(dataloader):
        data = batch["data"].to(device)
        # target = batch["target"].to(device)

        prediction = model(data)
        loss = loss_fn(prediction, data)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        progress_bar.update(1)

        if i % reporting_interval == 0:
            print(f"\nCurrent loss {loss.item()}")


def eval(model, dataloader: DataLoader, loss_fn, device="cuda"):
    model.eval()

    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    test_loss = 0.0, 0.0

    progress_bar = tqdm.tqdm(range(num_batches))

    with torch.no_grad():
        for batch in dataloader:
            data = batch["data"].to(device)
            # target = batch["target"].to(device)

            prediction = model(data)
            loss = loss_fn(prediction, data)
            test_loss = loss.item()

            progress_bar.update(1)

    test_loss  # /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")


def collate_fn(batch):
    result = dict()

    # result["target"] = torch.tensor([x["target"] for x in batch], dtype=torch.float32)
    result["data"] = torch.tensor([[x["data"]] for x in batch], dtype=torch.float32)

    return result


def preprocess(imgs):
    imgs = np.array(imgs["image"]).astype("float32") / 255.0
    imgs = np.pad(imgs, ((2, 2), (2, 2)), constant_values=0.0)

    return {"data": imgs}


if __name__ == "__main__":
    epoch = 10

    model_path = "model.ph"

    model = VariationalAutoencoder()

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, weights_only=True))

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
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
        shuffle=True,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )

    progress = tqdm.tqdm(range(len(train_dataloader) * epoch))

    model = model.to("cuda")

    eval(model, test_dataloader, loss_fn)
    for _ in range(epoch):
        train(model, train_dataloader, loss_fn, optimizer, progress)

        eval(model, test_dataloader, loss_fn)
        scheduler.step()

    torch.save(model.state_dict(), model_path)
