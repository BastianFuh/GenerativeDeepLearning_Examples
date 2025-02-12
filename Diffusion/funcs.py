from torch.utils.data import DataLoader
from tqdm import tqdm
import torch


def train(
    model,
    dataloader: DataLoader,
    loss_fn,
    optimizer,
    progress_bar: tqdm,
    device="cuda",
):
    """Training function"""

    num_batches = len(dataloader)
    reporting_interval = int(num_batches / 5)
    test_loss = 0.0

    # Prepare data

    model.train()
    for i, batch in enumerate(dataloader):
        data = batch["tokens"].to(device)
        target = batch["labels"].to(device)

        prediction = model(data)

        prediction = prediction.permute([0, 2, 1])

        loss = loss_fn(prediction, target)

        test_loss += loss.item()

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        progress_bar.update(1)

        if i % reporting_interval == 0 and i != 0:
            print_loss = test_loss / reporting_interval
            tqdm.write(f"{i}/{num_batches} Current avg training loss {print_loss}")
            test_loss = 0.0


def eval(model, dataloader: DataLoader, loss_fn, device="cuda"):
    """Evaluation function."""
    num_batches = len(dataloader)

    test_loss = 0.0

    progress_bar = tqdm(range(num_batches), position=1)

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            data = batch["tokens"].to(device)
            target = batch["labels"].to(device)

            prediction = model(data)

            prediction = prediction.permute([0, 2, 1])

            loss = loss_fn(prediction, target)

            test_loss += loss.item()

            progress_bar.update(1)

    test_loss /= num_batches
    tqdm.write(f"Avg evaluation loss: {test_loss:>8f}")

    return test_loss


def collate_fn(batch):
    result = dict()

    result["image"] = torch.stack([x["image"] for x in batch])
    result["label"] = torch.stack([x["label"] for x in batch])

    return result


def to_tensor(obj):
    if not isinstance(obj, torch.Tensor):
        return torch.tensor(obj)
    else:
        return obj


def linear_diffusion_schedule(diffusion_times):
    diffusion_times = to_tensor(diffusion_times)

    min_rate = 0.0001
    max_rate = 0.02

    betas = min_rate + diffusion_times * (max_rate - min_rate)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas)
    signal_rates = alpha_bars
    noise_rates = 1 - alpha_bars

    return noise_rates, signal_rates


def cosine_diffusion_schedule(diffusion_times):
    diffusion_times = to_tensor(diffusion_times)

    signal_rates = torch.cos(diffusion_times * torch.pi / 2)
    noise_rates = torch.sin(diffusion_times * torch.pi / 2)

    return noise_rates, signal_rates


def offset_cosine_diffusion_schedule(diffusion_times):
    diffusion_times = to_tensor(diffusion_times)

    min_signal_rate = 0.02
    max_singal_rate = 0.95

    start_angle = torch.acos(max_singal_rate)
    end_angle = torch.acos(min_signal_rate)

    diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

    signal_rates = torch.cos(diffusion_angles)
    noise_rates = torch.sin(diffusion_angles)

    return noise_rates, signal_rates
