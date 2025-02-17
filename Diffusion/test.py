import torch

import os

from matplotlib import pyplot as plt

from model import DiffusionModel


MODEL_PATH = "diffusion.pt"

DEVICE = "cuda"


if __name__ == "__main__":
    model = DiffusionModel(use_ema=False)
    if os.path.exists(MODEL_PATH):
        print("Loaded model")
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True), strict=False)
        model.create_ema()
        model.ema_network.load_state_dict(
            torch.load(f"ema-{MODEL_PATH}", weights_only=True)
        )

    model.eval()

    num_images = 10
    steps = 20

    img, initial_noise = model.generate(num_images, steps)

    fig = plt.figure()
    ax = fig.subplots(1, num_images)

    for index, img in enumerate(img):
        ax[index].imshow(img.transpose(0, 2).detach().numpy())
        ax[index].axis("off")

    plt.show()
