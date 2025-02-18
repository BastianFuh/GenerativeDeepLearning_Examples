"""
This example is based on a example given in
[Foster, D., & Friston, K. J. (2023). Generative Deep Learning: Teaching Machines to Paint, Write, Compose, and Play].

The author originally used Keras and Tensorflow however this example adepted the concept to pytorch.

!IMPORTANT!
This model requires a dataset from [1] to work. Download the dataset and copy the json file into a folder
called dataset.

[1] https://www.kaggle.com/datasets/hugodarwood/epirecipes



This script trains a Long Short-Term Memory(LSTM) Network. This network is a subtype of recurrent neural
networks(RNN). The unique aspect this model is that is does not only use a hidden state but also a cell state
which is continously updates with new data from the input.


"""

import torch
from torch.utils.data import DataLoader
from tokenizers import trainers, Tokenizer, models, pre_tokenizers

import torch.nn as nn
import datasets
from torchinfo import summary
from tqdm import tqdm

import os

import funcs
from funcs import (
    train,
    eval,
    collate_fn,
    preprocess,
    prepare_inputs,
    pad_punctuation,
    yield_text,
    tokenize_and_pad,
)

exit()

from model import GPT, NUM_CLASSES


if __name__ == "__main__":
    epoch = 100
    stop_threshold = 10

    model_path = "gpt.pt"
    tokenizer_path = "tokenizer_gpt.pt"

    model = GPT(inference=False)

    if os.path.exists(model_path):
        print("Loaded model")
        model.load_state_dict(torch.load(model_path, weights_only=True))

    loss_fn = nn.CrossEntropyLoss()

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

    summary(
        model,
        [2, 201],
        device="cpu",
    )

    # Prepare data
    dataset = datasets.load_dataset("./GenerativePreTrainedTransformer/dataset/")

    # Get column names for later
    sections = dataset["train"].column_names

    # Generate training labels
    dataset = dataset.map(preprocess)

    # Remove all other columns which are not needed anymore
    dataset = dataset.remove_columns(sections)

    dataset = dataset.map(pad_punctuation)

    tokenizer: Tokenizer = None

    if os.path.exists(tokenizer_path):
        tokenizer = Tokenizer.from_file(tokenizer_path)
    else:
        tokenizer = Tokenizer(models.WordLevel(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        trainer = trainers.WordLevelTrainer(
            vocab_size=NUM_CLASSES, special_tokens=["<pad>", "<unk>"]
        )
        tokenizer.train_from_iterator(yield_text(dataset), trainer=trainer)

        tokenizer.save("tokenizer_lstm.pt")

    funcs.tokenizer = tokenizer

    dataset = dataset.map(tokenize_and_pad)

    dataset = dataset.map(prepare_inputs)

    dataset = dataset.remove_columns(("data", "padded_text"))

    # Create train-test split
    dataset = dataset["train"].train_test_split(0.1)

    print(dataset)

    exit()

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
        batch_size=16,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )

    progress = tqdm(range(len(train_dataloader) * epoch), position=0)

    model = model.to("cuda")

    current_best_loss = eval(model, test_dataloader, loss_fn)

    tqdm.write(f"Initial Loss {current_best_loss}")

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
