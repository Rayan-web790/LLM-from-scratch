from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from polycodegpt.model.attention import CausalSelfAttention
from polycodegpt.model.config import ModelConfig


def test_attention_output_shape():
    cfg = ModelConfig.small()
    attn = CausalSelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
    x = torch.randn(2, 5, cfg.d_model)
    out = attn(x)
    assert out.shape == (2, 5, cfg.d_model)


def test_causal_mask_blocks_future():
    torch.manual_seed(0)
    cfg = ModelConfig.small(dropout=0.0)
    attn = CausalSelfAttention(cfg.d_model, cfg.n_heads, 0.0)
    attn.eval()
    x = torch.randn(1, 4, cfg.d_model)
    # modify last token; earlier outputs must be identical
    x2 = x.clone()
    x2[:, -1, :] += 10.0
    with torch.no_grad():
        o1 = attn(x)
        o2 = attn(x2)
    assert torch.allclose(o1[:, :3, :], o2[:, :3, :], atol=1e-5)


def test_attention_no_nan():
    cfg = ModelConfig.small(dropout=0.1)
    attn = CausalSelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
    attn.train()
    x = torch.randn(2, 6, cfg.d_model)
    out = attn(x)
    assert torch.isfinite(out).all()


def test_attention_determinism_under_seed():
    cfg = ModelConfig.small(dropout=0.0)
    x = torch.randn(1, 4, cfg.d_model)

    torch.manual_seed(1234)
    a = CausalSelfAttention(cfg.d_model, cfg.n_heads, 0.0).eval()
    torch.manual_seed(1234)
    b = CausalSelfAttention(cfg.d_model, cfg.n_heads, 0.0).eval()

    with torch.no_grad():
        assert torch.equal(a(x), b(x))

