import re
import string
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tqdm import tqdm
import torch

from model import NUM_CLASSES

tokenizer = None


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

    result["tokens"] = torch.tensor([x["tokens"] for x in batch], dtype=torch.int64)

    result["labels"] = torch.tensor([x["labels"] for x in batch]).to(torch.int64)

    return result


def preprocess(x):
    trainings_text = f"Recipe for {x['title']} | {' '.join(x['directions'])}"

    return {"data": trainings_text}


def pad_punctuation(s):
    padded_text = re.sub(f"([{string.punctuation}])", r" \1 ", s["data"])
    padded_text = re.sub(" +", " ", padded_text)
    padded_text = padded_text.lower()

    return {"padded_text": padded_text}


def yield_text(dataset):
    for x in dataset["train"]["padded_text"]:
        yield x


def tokenize_and_pad(text):
    encoding = tokenizer.encode(text["padded_text"])
    token_ids = encoding.ids[:201]  # Truncate if longer than max_seq_length

    # Pad if shorter
    input_length = len(token_ids)
    pad_length = max(0, 201 - input_length)
    token_ids.extend([tokenizer.token_to_id("<pad>")] * pad_length)
    return {"tokens": token_ids, "input_length": input_length}


def prepare_inputs(text):
    labels = text["tokens"][1:] + [tokenizer.token_to_id("<pad>")]
    return {"labels": labels}
