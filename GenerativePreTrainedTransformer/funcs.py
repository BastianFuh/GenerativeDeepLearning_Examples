import re
import string
import torch

from torch import dtype, Tensor, device
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from tqdm import tqdm

from const import SEQ_LENGTH

tokenizer = None


def process(batch: Tensor, loss_fn: Module, model: Module, device: str) -> Tensor:
    data = batch["tokens"].to(device)
    target = batch["labels"].to(device)

    prediction, _ = model(data)

    # Switch sequence and classes, because we need the shape (N, C, S)
    # where N = Batch size, C = Classes, and S = Sequence
    # normally the output would be (N, S, C)
    prediction = prediction.permute([0, 2, 1])

    loss = loss_fn(prediction, target)

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
        progress_bar.update(1)

        if i % reporting_interval == 0 and i != 0:
            print_loss = test_loss / reporting_interval
            tqdm.write(f"{i}/{num_batches} Current avg training loss {print_loss}")
            test_loss = 0.0


def eval(
    model: Module, dataloader: DataLoader, loss_fn: Module, device: str = "cuda"
) -> Tensor:
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


def collate_fn(batch: dict) -> dict:
    result = dict()

    result["tokens"] = torch.tensor([x["tokens"] for x in batch], dtype=torch.int64)

    result["labels"] = torch.tensor([x["labels"] for x in batch]).to(torch.int64)

    return result


def preprocess(x: dict) -> dict:
    trainings_text = f"wine review : {x['country']} : {x['province']} : {x['variety']} : {x['description']}"

    return {"data": trainings_text}


def pad_punctuation(s: dict) -> dict:
    padded_text = re.sub(f"([{string.punctuation}])", r" \1 ", s["data"])
    padded_text = re.sub(" +", " ", padded_text)
    padded_text = padded_text.lower()

    return {"padded_text": padded_text}


def yield_text(dataset: dict):
    for x in dataset["train"]["padded_text"]:
        yield x


def tokenize_and_pad(text: dict) -> dict:
    encoding = tokenizer.encode(text["padded_text"])
    token_ids = encoding.ids[:SEQ_LENGTH]  # Truncate if longer than max_seq_length

    # Pad if shorter
    input_length = len(token_ids)
    pad_length = max(0, SEQ_LENGTH - input_length)
    token_ids.extend([tokenizer.token_to_id("<pad>")] * pad_length)
    return {"tokens": token_ids, "input_length": input_length}


def prepare_inputs(text: dict) -> dict:
    labels = text["tokens"][1:] + [tokenizer.token_to_id("<pad>")]
    return {"labels": labels}


def causal_attention_mask(
    batch_size: int,
    n_dest: int,
    n_src: int,
    num_heads: int,
    dtype: dtype,
    device: device,
):
    i = torch.arange(n_dest).to(device)[:, None]
    j = torch.arange(n_src).to(device)
    m = i > j - n_src + n_dest
    mask = m.to(dtype)
    mask = torch.reshape(mask, [1, n_dest, n_src])
    mult = torch.concat(
        [
            torch.unsqueeze(torch.tensor(batch_size * num_heads), -1),
            torch.tensor([1, 1], dtype=torch.int32),
        ],
        0,
    )
    return torch.tile(mask, mult.numpy().tolist())
