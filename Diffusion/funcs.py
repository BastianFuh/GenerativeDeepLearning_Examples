import torch
import PIL
import PIL.Image

from torch import Tensor
from torch.optim import Optimizer
from torch.nn import Module
from torch.utils.data import DataLoader
from datasets import Dataset
from tqdm import tqdm


def process(batch: Tensor, loss_fn: Module, model: Module, device: str) -> Tensor:
    data = batch["data"].to(device)

    pred, noises = model(data)

    pred_noises = pred[0]

    loss = loss_fn(pred_noises, noises)

    return loss


def train(
    model: Module,
    dataloader: DataLoader,
    loss_fn: Module,
    optimizer: Optimizer,
    progress_bar: tqdm,
    device: str = "cuda",
) -> None:
    """Training function"""

    num_batches = len(dataloader)
    reporting_interval = int(num_batches / 5)
    test_loss = 0.0

    # Prepare data

    model.train()
    for i, batch in enumerate(dataloader):
        loss = process(batch, loss_fn, model, device)
        test_loss += loss.item()

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        model.update_ema()
        progress_bar.update(1)

        if i % reporting_interval == 0 and i != 0:
            print_loss = test_loss / reporting_interval
            tqdm.write(f"{i}/{num_batches} Current avg training loss {print_loss}")
            test_loss = 0.0


def eval(
    model: Module, dataloader: DataLoader, loss_fn: Module, device: str = "cuda"
) -> str:
    """Evaluation function."""
    num_batches = len(dataloader)

    test_loss = 0.0

    progress_bar = tqdm(range(num_batches), position=1)

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            loss = process(batch, loss_fn, model, device)
            test_loss += loss.item()

            progress_bar.update(1)

    test_loss /= num_batches
    tqdm.write(f"Avg evaluation loss: {test_loss:>8f}")

    return test_loss


def collate_fn(batch: Tensor) -> dict[str, Tensor]:
    result = dict()

    result["data"] = torch.stack([x["image_data"] for x in batch])

    return result


def map_pil(x: Dataset) -> dict:
    image_data = [PIL.Image.open(entry["path"]) for entry in x["image"]]

    return {"image_data": image_data}


def to_tensor(obj: int) -> Tensor:
    if not isinstance(obj, torch.Tensor):
        return torch.tensor(obj)
    else:
        return obj


def linear_diffusion_schedule(diffusion_times: Tensor | int) -> tuple[Tensor, Tensor]:
    diffusion_times = to_tensor(diffusion_times)

    min_rate = 0.0001
    max_rate = 0.02

    betas = min_rate + diffusion_times * (max_rate - min_rate)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas)
    signal_rates = alpha_bars
    noise_rates = 1 - alpha_bars

    return noise_rates, signal_rates


def cosine_diffusion_schedule(diffusion_times: Tensor | int) -> tuple[Tensor, Tensor]:
    diffusion_times = to_tensor(diffusion_times)

    signal_rates = torch.cos(diffusion_times * torch.pi / 2)
    noise_rates = torch.sin(diffusion_times * torch.pi / 2)

    return noise_rates, signal_rates


def offset_cosine_diffusion_schedule(
    diffusion_times: Tensor | int,
) -> tuple[Tensor, Tensor]:
    diffusion_times = to_tensor(diffusion_times)

    min_signal_rate = 0.02
    max_singal_rate = 0.95

    start_angle = torch.acos(torch.tensor(max_singal_rate))
    end_angle = torch.acos(torch.tensor(min_signal_rate))

    diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

    signal_rates = torch.cos(diffusion_angles)
    noise_rates = torch.sin(diffusion_angles)

    return noise_rates, signal_rates


def sinusoidal_embedding(x: Tensor) -> Tensor:
    frequecies = torch.exp(
        torch.linspace(torch.log(torch.tensor(1)), torch.log(torch.tensor(1000)), 16)
    ).to(x.device)

    angular_speed = 2.0 * torch.pi * frequecies

    embeddings = torch.concat(
        [torch.sin(angular_speed * x), torch.cos(angular_speed * x)], dim=3
    )

    # Create the correct shape [N, C, H, W], batch, channels, height, width
    embeddings = torch.transpose(embeddings, 1, 3)

    return embeddings
