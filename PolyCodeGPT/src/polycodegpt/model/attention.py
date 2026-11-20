"""Causal multi-head self-attention implemented from scratch."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Standard causal (masked) multi-head self-attention.

    Input/output shape: [B, T, d_model].
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: [B, T, d_model]
        b, t, _ = x.shape

        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        # q,k,v: [B, n_heads, T, head_dim]

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, n_heads, T, T]

        # causal mask: block future positions
        causal = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal[None, None, :, :], float("-inf"))

        if attn_mask is not None:
            # attn_mask: [B, T] or [B, 1, 1, T] of allowed keys (True/1 = allowed)
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.view(b, 1, 1, t)
            scores = scores.masked_fill(~attn_mask.to(torch.bool), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)  # [B, n_heads, T, head_dim]
        out = out.transpose(1, 2).contiguous().view(b, t, self.d_model)
        out = self.out_proj(out)
        out = self.resid_dropout(out)
        return out
