import torch
from torch.nn import functional as F
from model import LSTM

import numpy as np
from tokenizers import Tokenizer, Encoding
import os
from matplotlib import pyplot as plt


MODEL_PATH = "lstm.pt"
TOKENIZER_PATH = "tokenizer_lstm.pt"

DEVICE = "cuda"


class TextGenerator:
    def __init__(self, top_k=10):
        self.top_k = top_k

        self.model = LSTM()
        if not os.path.exists(MODEL_PATH):
            print("Pleasure ensure that a model exists")
            exit()

        self.model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
        self.model.eval()
        self.model = self.model.to(DEVICE)

        self.tokenizer: Tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

        self.random_generator = np.random.Generator(np.random.PCG64())

    def sample_from(self, probs, temperature):
        probs = probs ** (1 / temperature)
        probs = probs / torch.sum(probs, dim=-1)
        probs = probs.cpu().detach().numpy().flatten()
        return self.random_generator.choice(len(probs), p=probs), probs

    def generate(self, start_prompt, max_tokens, temperature):
        prompt = start_prompt
        sample_token = None

        iteration = 0
        while iteration < max_tokens and sample_token != 0:
            iteration += 1
            input_tokens: Encoding = self.tokenizer.encode(prompt)

            x = torch.tensor([input_tokens.ids], dtype=torch.long).to(DEVICE)

            y = self.model(x)

            out = self.sample_from(y, temperature)

            prompt += f" {self.tokenizer.decode([out[0]])}"

            print(f"{iteration}: {prompt}")

        print("\n", prompt)


if __name__ == "__main__":
    generator = TextGenerator()

    generator.generate("recipe for a vegetarian", 201, 0.5)
