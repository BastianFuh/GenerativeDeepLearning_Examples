"""
This example is based on a example given in
[Foster, D., & Friston, K. J. (2023). Generative Deep Learning: Teaching Machines to Paint, Write, Compose, and Play].

The author originally used Keras and Tensorflow however this example adepted the concept to pytorch.

!IMPORTANT!
This model requires a dataset from [1] to work. Download the dataset and copy the dataset folder into the
folder this script resides in.

[1] https://www.kaggle.com/datasets/joosthazelzet/lego-brick-images



This script trains a Deep Convolutional Generative Adversarial Network. The goal is it to pit
two models, a generator and a descriminator, against each other. The generator tries to generate
images that the discriminator detects as real. The discriminator then tries to detect all the fake
images.


"""

import numpy as np
import re
import string
import torch
from torch.utils.data import DataLoader
from tokenizers import trainers, Tokenizer, models, pre_tokenizers

import torch.nn as nn
from torch.nn import functional as F
import datasets
from torchinfo import summary
from tqdm import tqdm


from model import LSTM

import os


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
    print(num_batches)
    reporting_interval = int(num_batches / 5)
    test_loss = 0.0

    # Prepare data

    model.train()
    for i, batch in enumerate(dataloader):
        data = batch["tokens"].to(device)
        target = batch["labels"].to(device)

        prediction = model(data)

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

            loss = loss_fn(prediction, target)

            test_loss += loss.item()

            progress_bar.update(1)

    test_loss /= num_batches
    tqdm.write(f"Avg evaluation loss: {test_loss:>8f}")

    return test_loss


def collate_fn(batch):
    result = dict()

    result["tokens"] = torch.tensor([x["tokens"] for x in batch], dtype=torch.int64)
    result["labels"] = F.one_hot(
        torch.tensor([x["labels"] for x in batch], dtype=torch.int64), num_classes=10000
    ).to(torch.float32)

    return result


def preprocess(x):
    trainings_text = f"Recipe for {x['title']}  |  {x['directions']}"

    return {"data": trainings_text}


def pad_punctuation(s):
    padded_text = re.sub(f"([{string.punctuation}])", r" \1 ", s["data"])
    padded_text = re.sub(" +", " ", padded_text)
    padded_text = padded_text.lower()
    return {"padded_text": padded_text}


def yield_text(dataset):
    for x in dataset["train"]["padded_text"]:
        yield x


def tokenize_and_pad(tokenizer, text):
    encoding = tokenizer.encode(text)
    token_ids = encoding.ids[:201]  # Truncate if longer than max_seq_length

    # Pad if shorter
    pad_length = max(0, 201 - len(token_ids))
    token_ids.extend([tokenizer.token_to_id("<pad>")] * pad_length)

    return token_ids


def prepare_inputs(text):
    labels = text["tokens"][1:] + [tokenizer.token_to_id("<pad>")]
    return {"labels": labels}


if __name__ == "__main__":
    epoch = 100
    stop_threshold = 10

    model_path = "lstm.pt"

    model = LSTM()

    if os.path.exists(model_path):
        print("Loaded model")
        model.load_state_dict(torch.load(model_path, weights_only=True))

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

    summary(
        model,
        [201],
        device="cpu",
    )

    # Prepare data
    dataset = datasets.load_dataset("./LongShortTermMemory/dataset/")

    print(dataset)

    # Get column names for later
    sections = dataset["train"].column_names

    # Filter out malformed datapoints
    dataset = dataset.filter(lambda x: x["title"] is not None)
    dataset = dataset.filter(lambda x: x["directions"] is not None)

    # Generate training labels
    dataset = dataset.map(preprocess)

    # Remove all other columns which are not needed anymore
    dataset = dataset.remove_columns(sections)

    dataset = dataset.map(pad_punctuation)

    tokenizer = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.WordLevelTrainer(
        vocab_size=10000, special_tokens=["<pad>", "<unk>"]
    )
    tokenizer.train_from_iterator(yield_text(dataset), trainer=trainer)

    dataset = dataset.map(
        lambda x: {"tokens": tokenize_and_pad(tokenizer, x["padded_text"])}
    )

    dataset = dataset.map(prepare_inputs)

    dataset = dataset.remove_columns(("data", "padded_text"))

    # Create train-test split
    dataset = dataset["train"].train_test_split(0.1)

    print(dataset)

    train_dataloader = DataLoader(
        dataset["train"],
        collate_fn=collate_fn,
        num_workers=4,
        batch_size=32,
        shuffle=True,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )
    test_dataloader = DataLoader(
        dataset["test"],
        collate_fn=collate_fn,
        num_workers=4,
        batch_size=32,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )

    progress = tqdm(range(len(train_dataloader) * epoch), position=0)

    model = model.to("cuda")

    current_best_loss = eval(model, test_dataloader, loss_fn)

    no_improvement = 0
    for _ in range(epoch):
        train(model, train_dataloader, loss_fn, optimizer, progress)

        loss = eval(model, test_dataloader, loss_fn)

        if loss < current_best_loss:
            tqdm.write(f"Saved new best model. {loss} vs {current_best_loss}")
            current_best_loss = loss
            no_improvement = 0
            torch.save(model.state_dict(), model_path)
        else:
            no_improvement += 1
            if no_improvement == stop_threshold:
                tqdm.write(
                    f"There were no improvements for {stop_threshold} iterations."
                )
                tqdm.write("Stopping Training")
                exit()
