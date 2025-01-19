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
        target = batch["target"].to(device)

        prediction = model(data)
        loss = loss_fn(prediction, target)

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

    test_loss, correct = 0.0, 0.0

    progress_bar = tqdm.tqdm(range(num_batches))

    with torch.no_grad():
        for batch in dataloader:
            data = batch["data"].to(device)
            target = batch["target"].to(device)

            prediction = model(data)
            loss = loss_fn(prediction, target)
            test_loss = loss.item()

            correct += (
                (
                    np.argmax(F.softmax(prediction, -1).cpu(), -1)
                    == np.argmax(target.cpu(), -1)
                )
                .type(torch.float)
                .sum()
                .item()
            )

            progress_bar.update(1)

    test_loss  # /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


def collate_fn(batch):
    result = dict()

    result["target"] = torch.tensor([x["target"] for x in batch], dtype=torch.float32)
    result["data"] = torch.tensor([[x["data"]] for x in batch], dtype=torch.float32)

    return result


if __name__ == "__main__":
    epoch = 10

    encoder = nn.Sequential(
        # 32x32
        nn.Conv2d(3, 32, (3, 3), stride=2, padding="same"),
        nn.ReLU(),
        # 16x16
        nn.Conv2d(32, 64, (3, 3), stride=2, padding="same"),
        nn.ReLU(),
        # 8x8
        nn.Conv2d(64, 128, (3, 3), stride=2, padding="same"),
        nn.ReLU(),
        # 4x4
        nn.Flatten(),
        nn.Linear(4 * 4 * 128, 2),
    )

    model = nn.Sequential(
        encoder,
    )

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = LinearLR(optimizer, total_iters=epoch)

    summary(model, (1, 28, 28), device="cpu")

    # Prepare data
    dataset_raw = datasets.load_dataset("MNIST")

    dataset_onehot = dataset_raw.map(
        lambda x: {"target": [0 if v != x["label"] else 1 for v in range(10)]}
    )

    dataset_onehot = dataset_onehot.map(
        lambda x: {"data": torch.from_numpy(np.array(x["image"]) / 255.0)}
    )

    dataset = dataset_onehot.remove_columns("image")
    dataset = dataset.remove_columns("label")

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
