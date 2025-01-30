import torch
from model import DCGAN
import numpy as np
import os
from matplotlib import pyplot as plt


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


MODEL_PATH = "gan.pt"

DEVICE = "cuda"

if __name__ == "__main__":
    model = DCGAN()

    if not os.path.exists(MODEL_PATH):
        print("Pleasure ensure that a model exists")
        exit()

    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    model = model.to(DEVICE)

    # Shows a grid of processed test images
    data = torch.randn(size=(16, 100, 1, 1), device=DEVICE)

    data = data.to(DEVICE)

    output = model.generator(data)
    fig = plt.figure()
    ax = fig.subplots(2, 8)

    for i in range(0, 16):
        row = int(i / 8)
        column = int(i % 8)
        ax[row, column].imshow(output.cpu().detach().numpy()[i].squeeze(), cmap="gray")

    plt.show()
