"""
This example is based on a example given in
[Foster, D., & Friston, K. J. (2023). Generative Deep Learning: Teaching Machines to Paint, Write, Compose, and Play].

The author originally used Keras and Tensorflow however this example adepted the concept to pytorch.

!IMPORTANT!
This model requires a dataset from [1] to work. Download the dataset and copy the content of the dataset folder into a folder
called dataset.

[1] https://www.kaggle.com/datasets/nunenuh/pytorch-challange-flower-dataset



This script trains a Long Short-Term Memory(LSTM) Network. This network is a subtype of recurrent neural
networks(RNN). The unique aspect this model is that is does not only use a hidden state but also a cell state
which is continously updates with new data from the input.


"""

import torch
from torch.utils.data import DataLoader

from torchvision.transforms import v2 as transforms

import torch.nn as nn
import datasets
from torchinfo import summary
from tqdm import tqdm

import os

from funcs import train, eval, collate_fn, map_pil

from model import DiffusionModel


if __name__ == "__main__":
    epoch = 100
    stop_threshold = 10

    model_path = "diffusion.pt"

    model = DiffusionModel(in_training=True)

    if os.path.exists(model_path):
        print("Loaded model")
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.ema_network.load_state_dict(
            torch.load(f"ema-{model_path}", weights_only=True)
        )

    loss_fn = nn.L1Loss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    summary(
        model,
        [2, 3, 64, 64],
        device="cpu",
    )

    # Prepare data
    dataset = datasets.load_dataset("./Diffusion/dataset").cast_column(
        "image", datasets.Image(decode=True)
    )

    # Repeat every data point 5 times to increase dataset size
    for label in dataset:
        dataset[label] = datasets.Dataset.from_list(dataset[label].to_list() * 5)

    print(dataset)

    data_transform = {
        "train": transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize([0.5], [0.5], inplace=True),
            ]
        ),
        "validation": transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize([0.5], [0.5], inplace=True),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize([0.5], [0.5], inplace=True),
            ]
        ),
    }

    dataset = dataset.map(map_pil, batched=True)

    dataset = dataset.remove_columns("image")

    for label in dataset:
        print(label)
        dataset[label].set_transform(data_transform[label])

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

    eval_dataloader = DataLoader(
        dataset["validation"],
        collate_fn=collate_fn,
        num_workers=4,
        batch_size=16,
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

    current_best_loss = eval(model, eval_dataloader, loss_fn)

    tqdm.write(f"Initial Loss {current_best_loss}")

    no_improvement = 0

    for _ in range(epoch):
        train(model, train_dataloader, loss_fn, optimizer, progress)

        loss = eval(model, eval_dataloader, loss_fn)

        if loss < current_best_loss:
            tqdm.write(f"Saved new best model. {loss} vs {current_best_loss}")
            current_best_loss = loss
            no_improvement = 0
            torch.save(model.state_dict(), model_path)
            torch.save(model.ema_network.state_dict(), f"ema-{model_path}")
        else:
            no_improvement += 1
            if no_improvement == stop_threshold:
                tqdm.write(
                    f"There were no improvements for {stop_threshold} iterations."
                )
                tqdm.write("Stopping Training")
                exit()
