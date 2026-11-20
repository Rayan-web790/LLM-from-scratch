"""Decoder-only GPT model."""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .layers import TransformerBlock


class GPT(nn.Module):
    """Decoder-only Transformer language model.

    Token + learned absolute positional embeddings -> N pre-LayerNorm blocks ->
    final LayerNorm -> vocabulary projection, with the projection weight tied to
    the token embedding matrix.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        self.embed_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)
        # residual projections scaled init
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("fc2.weight"):
                nn.init.normal_(p, mean=0.0, std=config.init_std / math.sqrt(2 * config.n_layers))

        # re-tie after apply (apply may have overwritten? weights stay shared reference)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def _init_weights(self, module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        # input_ids: [B, T]
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be 2D [B,T], got shape {tuple(input_ids.shape)}")
        b, t = input_ids.shape
        if t > self.config.context_length:
            raise ValueError(
                f"sequence length {t} exceeds context_length {self.config.context_length}"
            )
        if t < 1:
            raise ValueError("sequence length must be >= 1")

        device = input_ids.device
        positions = torch.arange(t, device=device).unsqueeze(0)  # [1, T]

        tok = self.token_embedding(input_ids)  # [B, T, d_model]
        pos = self.position_embedding(positions)  # [1, T, d_model]
        x = self.embed_dropout(tok + pos)

        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)

        x = self.ln_f(x)  # [B, T, d_model]
        logits = self.lm_head(x)  # [B, T, vocab_size]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    def initialization_metadata(self) -> dict:
        """Record of the random initialization procedure (no pretrained weights)."""
        return {
            "scheme": "normal(0, init_std) for Linear/Embedding, zeros for biases, ones for LayerNorm",
            "init_std": self.config.init_std,
            "residual_projections": "out_proj/fc2 weights scaled by init_std / sqrt(2 * n_layers)",
            "residual_scale": self.config.init_std / math.sqrt(2 * self.config.n_layers),
            "layer_norm_eps": self.config.layer_norm_eps,
            "pretrained": False,
            "tie_embeddings": self.config.tie_embeddings,
            "seed_note": "initialized from the ambient torch RNG state; caller seeds it",
        }

    def parameter_breakdown(self) -> dict[str, Any]:
        """Trainable parameter counts grouped by architectural component."""
        d = self.config.d_model
        v = self.config.vocab_size
        ctx = self.config.context_length
        attn = sum(p.numel() for blk in self.blocks for p in blk.attn.parameters())
        ffn = sum(p.numel() for blk in self.blocks for p in blk.ffn.parameters())
        ln = sum(p.numel() for blk in self.blocks for p in list(blk.ln1.parameters()) + list(blk.ln2.parameters()))
        return {
            "token_embeddings": v * d,
            "positional_embeddings": ctx * d,
            "attention_projections": attn,
            "feed_forward": ffn,
            "layer_norms": ln + sum(p.numel() for p in self.ln_f.parameters()),
            "final_vocab_projection": 0 if self.config.tie_embeddings else v * d,
            "weight_tying": f"lm_head shares token_embedding ({v * d} params counted once)" if self.config.tie_embeddings else "disabled",
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be [B, T]")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")

        # inference mode without permanently changing the caller's train/eval state
        was_training = self.training
        self.eval()
        try:
            return self._generate_loop(input_ids, max_new_tokens, temperature, top_k, eos_token_id)
        finally:
            self.train(was_training)

    def _generate_loop(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: Optional[int],
        eos_token_id: Optional[int],
    ) -> torch.Tensor:
        out = input_ids.clone()
        for _ in range(max_new_tokens):
            ctx = out[:, -self.config.context_length:]
            logits, _ = self.forward(ctx)
            logits = logits[:, -1, :]  # [B, vocab]
            if temperature == 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                if temperature != 1.0:
                    logits = logits / temperature
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            out = torch.cat([out, next_token], dim=1)
            if eos_token_id is not None and bool((next_token == eos_token_id).all()):
                break
        return out
