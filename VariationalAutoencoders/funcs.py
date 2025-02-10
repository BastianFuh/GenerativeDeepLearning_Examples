import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch


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
