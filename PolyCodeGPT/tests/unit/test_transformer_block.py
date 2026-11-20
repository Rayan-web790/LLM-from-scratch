from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from polycodegpt.model.config import ModelConfig
from polycodegpt.model.layers import TransformerBlock


def test_block_residual_shape():
    cfg = ModelConfig.small()
    block = TransformerBlock(cfg)
    x = torch.randn(2, 5, cfg.d_model)
    out = block(x)
    assert out.shape == x.shape


def test_block_finite():
    cfg = ModelConfig.small(dropout=0.1)
    block = TransformerBlock(cfg)
    block.train()
    x = torch.randn(2, 5, cfg.d_model)
    out = block(x)
    assert torch.isfinite(out).all()


def test_ffn_shape():
    from polycodegpt.model.layers import FeedForward
    cfg = ModelConfig.small()
    ffn = FeedForward(cfg.d_model, cfg.ffn_dim, 0.0)
    x = torch.randn(2, 5, cfg.d_model)
    out = ffn(x)
    assert out.shape == x.shape

