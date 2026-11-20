from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from polycodegpt.model.config import ModelConfig
from polycodegpt.model.gpt import GPT
from polycodegpt.model.checkpoint import save_checkpoint, load_checkpoint


def _small_cfg(**kw):
    return ModelConfig.small(**kw)


def test_forward_shape():
    cfg = _small_cfg()
    m = GPT(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 6))
    logits, loss = m(ids)
    assert logits.shape == (2, 6, cfg.vocab_size)
    assert loss is None


def test_loss_finite():
    cfg = _small_cfg()
    m = GPT(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 6))
    targets = ids.clone()
    logits, loss = m(ids, targets=targets)
    assert loss is not None
    assert torch.isfinite(loss)


def test_loss_ignore_index():
    cfg = _small_cfg()
    m = GPT(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 6))
    targets = torch.full_like(ids, -100)
    targets[0, 0] = ids[0, 0]
    _, loss = m(ids, targets=targets)
    assert torch.isfinite(loss)


def test_backward_grads():
    cfg = _small_cfg()
    m = GPT(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 6))
    targets = ids.clone()
    _, loss = m(ids, targets=targets)
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


def test_weight_tying_identity():
    cfg = _small_cfg()
    m = GPT(cfg)
    assert m.lm_head.weight is m.token_embedding.weight


def test_seq_too_long_rejected():
    cfg = _small_cfg(context_length=4)
    m = GPT(cfg)
    ids = torch.randint(0, cfg.vocab_size, (1, 5))
    import pytest
    with pytest.raises(ValueError):
        m(ids)


def test_generate_shape():
    cfg = _small_cfg()
    m = GPT(cfg)
    m.eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 3))
    out = m.generate(ids, max_new_tokens=4, temperature=0.0)
    assert out.shape == (2, 7)


def test_greedy_deterministic():
    torch.manual_seed(0)
    cfg = _small_cfg()
    m = GPT(cfg)
    m.eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 3))
    with torch.no_grad():
        a = m.generate(ids.clone(), max_new_tokens=5, temperature=0.0)
        b = m.generate(ids.clone(), max_new_tokens=5, temperature=0.0)
    assert torch.equal(a, b)


def test_eos_stopping():
    cfg = _small_cfg()
    m = GPT(cfg)
    vocab = cfg.vocab_size
    eos = 2

    def forced_forward(input_ids, targets=None, attn_mask=None):
        b, t = input_ids.shape
        logits = torch.full((b, t, vocab), -10.0)
        logits[:, :, eos] = 10.0  # eos is always argmax
        return logits, None

    m.forward = forced_forward  # type: ignore[method-assign]
    ids = torch.tensor([[1, 2, 3]])
    out = m.generate(ids, max_new_tokens=6, temperature=0.0, eos_token_id=eos)
    # stops right after emitting the first eos, not after max_new_tokens
    assert out.shape[1] == ids.shape[1] + 1
    assert int(out[0, -1]) == eos


def test_generation_crops_long_context():
    cfg = _small_cfg(context_length=4)
    m = GPT(cfg)
    m.eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 10))  # longer than context
    out = m.generate(ids, max_new_tokens=2, temperature=0.0)
    assert out.shape == (1, 12)


def test_future_token_perturbation():
    cfg = _small_cfg()
    m = GPT(cfg)
    m.eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 5))
    ids2 = ids.clone()
    ids2[0, -1] = (ids[0, -1] + 1) % cfg.vocab_size
    with torch.no_grad():
        l1, _ = m(ids)
        l2, _ = m(ids2)
    assert torch.allclose(l1[:, :4, :], l2[:, :4, :], atol=1e-5)


def test_checkpoint_roundtrip(tmp_path):
    cfg = _small_cfg()
    m = GPT(cfg)
    m.eval()
    ckpt = tmp_path / "m.pt"
    save_checkpoint(ckpt, m, cfg, tokenizer_version="1.1.0")
    sd, cfg2, meta = load_checkpoint(ckpt)
    m2 = GPT(cfg2)
    m2.load_state_dict(sd)
    m2.eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 4))
    with torch.no_grad():
        a, _ = m(ids)
        b, _ = m2(ids)
    assert torch.allclose(a, b)
    assert meta["param_count"] == sum(p.numel() for p in m.parameters())
    assert m2.lm_head.weight is m2.token_embedding.weight
    assert meta["init_metadata"]["pretrained"] is False


def test_configured_model_forward_smoke():
    """Integration: the real 16K-vocab, 6-layer configuration runs a forward pass."""
    cfg = ModelConfig.from_yaml(PROJECT_ROOT / "configs" / "model.yaml")
    m = GPT(cfg)
    m.eval()
    n = sum(p.numel() for p in m.parameters())
    assert n == 16_988_160, f"unexpected parameter count {n:,}"
    assert n == cfg.estimate_parameters()
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    with torch.no_grad():
        logits, _ = m(ids)
    assert logits.shape == (1, 16, cfg.vocab_size)
    assert torch.isfinite(logits).all()


def test_no_pretrained_deps():
    """Model sources must not use nn.MultiheadAttention/nn.Transformer or HF packages."""
    import ast

    model_dir = PROJECT_ROOT / "src" / "polycodegpt" / "model"
    banned_attrs = {"MultiheadAttention", "Transformer", "TransformerEncoder", "TransformerEncoderLayer"}
    banned_mods = {"transformers", "torchvision", "timm", "sentence_transformers"}
    for path in model_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in banned_mods, f"{path.name} imports {alias.name}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in banned_mods, f"{path.name} imports {node.module}"
            elif isinstance(node, ast.Attribute):
                assert node.attr not in banned_attrs, f"{path.name} uses banned attribute {node.attr}"


def test_param_breakdown_sums_to_total():
    cfg = _small_cfg()
    m = GPT(cfg)
    b = m.parameter_breakdown()
    numeric = sum(v for v in b.values() if isinstance(v, int))
    assert numeric == sum(p.numel() for p in m.parameters())
    assert numeric == cfg.estimate_parameters()


def test_generation_restores_train_mode():
    cfg = _small_cfg()
    m = GPT(cfg)
    m.train()
    ids = torch.randint(0, cfg.vocab_size, (1, 3))
    m.generate(ids, max_new_tokens=2, temperature=0.0)
    assert m.training is True


def test_generation_does_not_change_params():
    cfg = _small_cfg()
    m = GPT(cfg)
    before = [p.detach().clone() for p in m.parameters()]
    ids = torch.randint(0, cfg.vocab_size, (1, 3))
    m.generate(ids, max_new_tokens=3, temperature=1.0, top_k=5)
    for b, p in zip(before, m.parameters()):
        assert torch.equal(b, p)

