import torch
from torch.utils.data import DataLoader
import datasets
from model import VariationalAutoencoder, EMBEDDING_SIZE
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


if __name__ == "__main__":
    model = VariationalAutoencoder()

    model_path = "vae_3.pt"
    device = "cuda"

    if not os.path.exists(model_path):
        print("Pleasure ensure that a model exist")
        exit()

    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    model = model.to(device)

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

    # If false, show scatter plot
    SHOW_IMAGES = False

    if SHOW_IMAGES:
        # Shows a grid of processed test images
        data = next(iter(test_dataloader))

        data = data["data"].to(device)

        output = model.encoder(data)
        f, ax = plt.subplots(4, 8)

        for i in range(0, 16):
            row = 2 * int(i / 8)
            column = int(i % 8)
            ax[row, column].imshow(data.cpu()[i].squeeze(), cmap="gray")
            ax[row + 1, column].imshow(
                output.cpu().detach().numpy()[i].squeeze(), cmap="gray"
            )

        plt.show()
    else:
        # Shows a scatter plot of the embedding space

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        total_embeddings = None
        labels = list()
        for data in test_dataloader:
            x = data["data"].to(device)
            labels.append(data["label"])
            embedding = model.encoder(x)
            if total_embeddings is None:
                total_embeddings = embedding
            else:
                total_embeddings = torch.cat((total_embeddings, embedding), 0)

        total_embeddings = total_embeddings.cpu().detach()

        sc = ax.scatter(
            total_embeddings[:, 0],
            total_embeddings[:, 1],
            total_embeddings[:, 2],
            c=labels,
            alpha=0.5,
            s=3,
            cmap="gist_rainbow",
        )
        fig.colorbar(sc)
        plt.show()
