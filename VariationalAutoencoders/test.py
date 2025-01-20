import torch
from torch.utils.data import DataLoader
import datasets
from model import VariationalAutoencoder
import numpy as np
import os
import random
from matplotlib import pyplot as plt


def preprocess(imgs):
    imgs = np.array(imgs["image"]).astype("float32") / 255.0
    imgs = np.pad(imgs, ((2, 2), (2, 2)), constant_values=0.0)

    return {"data": imgs}


def collate_fn(batch):
    result = dict()

    # result["target"] = torch.tensor([x["target"] for x in batch], dtype=torch.float32)
    result["data"] = torch.tensor([[x["data"]] for x in batch], dtype=torch.float32)

    return result


if __name__ == "__main__":
    model = VariationalAutoencoder()

    model_path = "model.ph"
    device = "cuda"

    if not os.path.exists(model_path):
        print("Pleasure ensure that a model exist")
        exit()

    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    model = model.to(device)

    dataset = datasets.load_dataset("fashion_mnist")

    dataset = dataset.remove_columns("label")
    dataset = dataset.map(preprocess)

    dataset = dataset.remove_columns("image")

    print(dataset)

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

    data = next(iter(test_dataloader))

    data = data["data"].to(device)

    output = model(data)

    f, ax = plt.subplots(4, 8)

    for i in range(0, 16):
        row = 2 * int(i / 8)
        column = int(i % 8)
        ax[row, column].imshow(data.cpu()[i].squeeze(), cmap="gray")
        ax[row + 1, column].imshow(
            output.cpu().detach().numpy()[i].squeeze(), cmap="gray"
        )

    plt.show()
