import torch
from torch.utils.data import DataLoader
import datasets
from model import VariationalAutoencoder
import numpy as np
import os
import random
from matplotlib import pyplot as plt
from matplotlib import cm, colors


def preprocess(imgs):
    imgs = np.array(imgs["image"]).astype("float32") / 255.0
    imgs = np.pad(imgs, ((2, 2), (2, 2)), constant_values=0.0)

    return {"data": imgs}


def collate_fn(batch):
    result = dict()

    result["label"] = torch.tensor([x["label"] for x in batch], dtype=torch.float32)
    result["data"] = torch.tensor([[x["data"]] for x in batch], dtype=torch.float32)

    return result


EMBEDDING_SIZE = 2
USE_MULTI_VARIATIONAL = True

if USE_MULTI_VARIATIONAL:
    MODEL_PATH = f"vae_{EMBEDDING_SIZE}.pt"
else:
    MODEL_PATH = f"ae_{EMBEDDING_SIZE}.pt"

DEVICE = "cuda"

if __name__ == "__main__":
    model = VariationalAutoencoder(USE_MULTI_VARIATIONAL, embedding_size=EMBEDDING_SIZE)

    if not os.path.exists(MODEL_PATH):
        print("Pleasure ensure that a model exists")
        exit()

    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    model = model.to(DEVICE)

    dataset = datasets.load_dataset("fashion_mnist")

    # dataset = dataset.remove_columns("label")
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

    # Shows a grid of processed test images
    data = next(iter(test_dataloader))

    data = data["data"].to(DEVICE)

    output, _, _ = model(data)
    fig = plt.figure()
    ax = fig.subplots(4, 8)

    for i in range(0, 16):
        row = 2 * int(i / 8)
        column = int(i % 8)
        ax[row, column].imshow(data.cpu()[i].squeeze(), cmap="gray")
        ax[row + 1, column].imshow(
            output.cpu().detach().numpy()[i].squeeze(), cmap="gray"
        )

    # Shows a scatter plot of the embedding space
    fig = plt.figure()
    if EMBEDDING_SIZE == 2 or EMBEDDING_SIZE == 3:
        if EMBEDDING_SIZE == 3:
            ax = fig.add_subplot(projection="3d")
        else:
            ax = fig.add_subplot()
        total_embeddings = None
        labels = list()
        for data in test_dataloader:
            x = data["data"].to(DEVICE)
            labels.append(data["label"])
            embedding, mean, _ = model.encoder(x)
            if total_embeddings is None:
                total_embeddings = mean
            else:
                total_embeddings = torch.cat((total_embeddings, mean), 0)

        total_embeddings = total_embeddings.cpu().detach()

        if EMBEDDING_SIZE == 3:
            sc = ax.scatter(
                total_embeddings[:, 0],
                total_embeddings[:, 1],
                total_embeddings[:, 2],
                c=labels,
                alpha=0.5,
                s=3,
                cmap="gist_rainbow",
            )
        else:
            sc = ax.scatter(
                total_embeddings[:, 0],
                total_embeddings[:, 1],
                c=labels,
                alpha=0.5,
                s=3,
                cmap="gist_rainbow",
            )
        fig.colorbar(sc)
    plt.show()
