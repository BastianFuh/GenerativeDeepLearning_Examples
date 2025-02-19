import torch

import torch.nn as nn
import torch.nn.functional as F


from torch import Tensor


from funcs import causal_attention_mask
from const import NUM_CLASSES, SEQ_LENGTH


class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, max_len: int, vocab_size: int, embed_dim: int):
        super(TokenAndPositionEmbedding, self).__init__()

        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_len, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        max_len = x.shape[-1]
        positions = torch.arange(0, max_len, 1).to(x.device)
        positions_embedding = self.pos_emb(positions.to(torch.long))

        token_embedding = self.token_emb(x.to(torch.long))

        return token_embedding + positions_embedding


class TransformerBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        key_dim: int,
        embeded_dim: int,
        ff_dim: int,
        dropout_rate: int = 0.1,
        return_attn_weights=False,
    ):
        super(TransformerBlock, self).__init__()

        self.num_heads = num_heads
        self.return_attn_weights = return_attn_weights

        self.attn = nn.MultiheadAttention(
            embeded_dim,
            self.num_heads,
            kdim=key_dim,
            batch_first=True,
        )

        self.dropout_1 = nn.Dropout(dropout_rate)

        self.ln_1 = nn.LayerNorm(embeded_dim, eps=1e-6)

        self.ffn_1 = nn.Linear(embeded_dim, ff_dim)
        self.ffn_2 = nn.Linear(ff_dim, embeded_dim)

        self.dropout_2 = nn.Dropout(dropout_rate)
        self.ln_2 = nn.LayerNorm(embeded_dim, eps=1e-6)

    def forward(self, x: Tensor) -> Tensor:
        input_shape = x.shape
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = causal_attention_mask(
            batch_size, seq_len, seq_len, self.num_heads, torch.bool, x.device
        )

        attention_output, attention_scores = self.attn(
            x,
            x,
            x,
            attn_mask=causal_mask,
            need_weights=self.return_attn_weights,
        )

        attention_output = self.dropout_1(attention_output)
        out1 = self.ln_1(x + attention_output)

        x = F.relu(self.ffn_1(out1))
        x = self.ffn_2(x)

        x = self.dropout_2(x)
        x = self.ln_2(out1 + x)

        return (x, attention_scores)


class GPT(nn.Module):
    """Example Module for an Generative pre-trained transformer Network"""

    def __init__(
        self,
        num_head: int = 2,
        key_dim: int = 256,
        embeded_dim: int = 256,
        ff_dim: int = 256,
        max_len: int = SEQ_LENGTH,
        dropout_rate: int = 0.1,
    ):
        super(GPT, self).__init__()

        self.transformer = TransformerBlock(
            num_head,
            key_dim,
            embeded_dim,
            ff_dim,
            dropout_rate,
        )

        self.embedding = TokenAndPositionEmbedding(max_len, NUM_CLASSES, embeded_dim)

        self.head = nn.Linear(embeded_dim, NUM_CLASSES)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)
        x, attention_scores = self.transformer(x)
        x = F.softmax(self.head(x), dim=-1)
        return x, attention_scores
