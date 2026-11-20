"""Transformer block and feed-forward network."""

from __future__ import annotations

import torch.nn as nn

from .attention import CausalSelfAttention
from .config import ModelConfig


class FeedForward(nn.Module):
    """Position-wise feed-forward network with GELU."""

    def __init__(self, d_model: int, ffn_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, ffn_dim, bias=True)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(ffn_dim, d_model, bias=True)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, d_model]
        x = self.fc1(x)          # [B, T, ffn_dim]
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)          # [B, T, d_model]
        x = self.drop2(x)
        return x


class TransformerBlock(nn.Module):
    """Pre-LayerNorm transformer block."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.attn = CausalSelfAttention(config.d_model, config.n_heads, config.dropout)
        self.ln2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.ffn = FeedForward(config.d_model, config.ffn_dim, config.dropout)

    def forward(self, x, attn_mask=None):
        # x: [B, T, d_model]
        x = x + self.attn(self.ln1(x), attn_mask=attn_mask)
        x = x + self.ffn(self.ln2(x))
        return x
