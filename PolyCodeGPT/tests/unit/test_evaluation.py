"""Unit tests for Phase 6 evaluation helpers.

All tests use synthetic fixtures or tiny tensors. The real test.bin is never read.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from polycodegpt.data.dataset import TokenBlockDataset, build_dataloader
from polycodegpt.data.stream import TokenStreamWriter
from polycodegpt.data.tokenizer_io import SPECIAL_TOKEN_IDS, verify_special_tokens, load_tokenizer
from polycodegpt.model import GPT, ModelConfig
from polycodegpt.training import (
    evaluate_loss,
    load_checkpoint,
    perplexity,
    rebuild_model,
    weights_are_tied,
)
from polycodegpt.training.checkpoint import save_checkpoint as save_training_checkpoint

VOCAB = 32


def make_model(config, seed=1234):
    torch.manual_seed(seed)
    return GPT(config)


def write_stream(path, tokens=400, vocab=VOCAB, seed=0):
    ids = np.random.default_rng(seed).integers(0, vocab, size=tokens)
    with TokenStreamWriter(path, vocab) as w:
        w.write(ids)
    return path


# --------------------------------------------------------------------------- #
# token-weighted loss + perplexity
# --------------------------------------------------------------------------- #
def test_full_split_token_weighted_loss():
    """Token-weighted mean over all blocks, not mean of batch means."""
    # Two batches: batch0 loss 2.0 over 4 tokens, batch1 loss 4.0 over 2 tokens
    class Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._losses = iter([2.0, 4.0])
            self.anchor = torch.nn.Parameter(torch.zeros(1))

        def forward(self, input_ids, targets=None, attn_mask=None):
            return None, torch.tensor(next(self._losses), requires_grad=False)

        def eval(self):
            return self

        def train(self, mode=True):
            return self

    model = Stub()
    targets_a = torch.tensor([[1, 2, 3, 4]])
    targets_b = torch.tensor([[1, 2, -100, -100]])
    batches = [
        (torch.zeros_like(targets_a), targets_a),
        (torch.zeros_like(targets_b), targets_b),
    ]
    result = evaluate_loss(model, batches, "cpu")
    weighted = (2.0 * 4 + 4.0 * 2) / 6
    assert result["loss"] == pytest.approx(weighted)
    assert result["tokens"] == 6
    assert result["perplexity"] == pytest.approx(math.exp(weighted))


def test_perplexity_calculation():
    assert perplexity(0.0) == pytest.approx(1.0)
    assert perplexity(math.log(10)) == pytest.approx(10.0)
    assert perplexity(1000.0) == float("inf")


def test_final_incomplete_tail_handled():
    """build_dataloader with drop_last=False keeps the short final batch."""
    import tempfile, os
    tmp = Path(tempfile.mkdtemp())
    stream = write_stream(tmp / "s.bin", tokens=20, vocab=VOCAB, seed=1)
    ds = TokenBlockDataset(stream, context_length=8)  # 20 tokens -> 2 blocks? (20-9)//8=1 -> actually 1 block? let's just check loader
    # Use a stream large enough for multiple blocks + tail
    stream2 = write_stream(tmp / "s2.bin", tokens=60, vocab=VOCAB, seed=2)
    ds2 = TokenBlockDataset(stream2, context_length=8)
    # 60 tokens: blocks = 1 + (60-9)//8 = 1+6 = 7, tokens_in_blocks = 6*8+9=57, dropped 3
    assert ds2.dropped_trailing_tokens == 3
    loader = build_dataloader(ds2, batch_size=3, shuffle=False, drop_last=False)
    batches = list(loader)
    # 7 blocks / 3 = 3 batches (3,3,1) -> last batch has 1 block (incomplete tail batch)
    assert len(batches) == 3
    assert batches[-1][0].shape[0] == 1
    # evaluate_loss handles the short batch via token weighting
    cfg = ModelConfig.small(vocab_size=VOCAB, context_length=8, dropout=0.0)
    model = make_model(cfg)
    result = evaluate_loss(model, batches, "cpu")
    assert result["tokens"] == 7 * 8
    assert result["sequences"] == 7


# --------------------------------------------------------------------------- #
# checkpoint loading + tied embeddings
# --------------------------------------------------------------------------- #
def test_checkpoint_loading_and_tied_embeddings(tmp_path):
    cfg = ModelConfig.small(vocab_size=VOCAB, context_length=8, dropout=0.0)
    model = make_model(cfg)
    dataset = TokenBlockDataset(write_stream(tmp_path / "s.bin", 200), context_length=8)
    settings = None
    # use a minimal trainer-less save: save via training checkpoint helper
    import torch
    from polycodegpt.training import TrainSettings
    ts = TrainSettings(micro_batch_size=2, gradient_accumulation_steps=2, max_steps=1,
                       total_steps=10, precision="fp32", context_length=8, initial_validation=False)
    from polycodegpt.training import Trainer
    loader = build_dataloader(dataset, batch_size=2, shuffle=True, seed=7, pin_memory=False)
    trainer = Trainer(model, loader, ts, device="cpu", log_fn=lambda *a, **k: None)
    path = tmp_path / "ckpt.pt"
    trainer.save(path)

    payload = load_checkpoint(path, map_location="cpu")
    restored = rebuild_model(payload, device="cpu")
    assert weights_are_tied(restored)
    assert restored.lm_head.weight is restored.token_embedding.weight
    # weights match
    for key, value in payload["model_state"].items():
        assert torch.equal(restored.state_dict()[key], value), key
    # finite
    assert all(torch.isfinite(p).all().item() for p in restored.parameters())


def test_no_gradients_or_optimizer_updates_during_eval(tmp_path):
    cfg = ModelConfig.small(vocab_size=VOCAB, context_length=8, dropout=0.0)
    model = make_model(cfg)
    before = [p.detach().clone() for p in model.parameters()]
    stream = write_stream(tmp_path / "s.bin", 200)
    ds = TokenBlockDataset(stream, context_length=8)
    loader = build_dataloader(ds, batch_size=2, shuffle=False)
    result = evaluate_loss(model, loader, "cpu")
    assert all(p.grad is None for p in model.parameters())
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))
    assert math.isfinite(result["loss"])


# --------------------------------------------------------------------------- #
# deterministic greedy generation
# --------------------------------------------------------------------------- #
def test_deterministic_greedy_generation():
    cfg = ModelConfig.small(vocab_size=VOCAB, context_length=8, dropout=0.0)
    model = make_model(cfg).eval()
    ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    out1 = model.generate(ids.clone(), max_new_tokens=5, temperature=0)
    out2 = model.generate(ids.clone(), max_new_tokens=5, temperature=0)
    assert torch.equal(out1, out2)
    # greedy => argmax path; output length = input + 5 (or stopped by eos none)
    assert out1.shape[1] == 3 + 5


# --------------------------------------------------------------------------- #
# special-token handling
# --------------------------------------------------------------------------- #
def test_special_token_handling():
    # verify_special_tokens needs a real tokenizer; use the actual artifact if present
    from pathlib import Path
    tok_path = PROJECT_ROOT / "tokenizer" / "artifacts" / "polycodegpt_tokenizer_v1_1.json"
    if not tok_path.is_file():
        pytest.skip("tokenizer artifact not available")
    from polycodegpt.data.tokenizer_io import load_tokenizer
    tok = load_tokenizer(tok_path, vocab_size=16000)
    verify_special_tokens(tok)
    # eos id
    assert tok.encode("<|eos|>", add_special_tokens=False).ids == [SPECIAL_TOKEN_IDS["<|eos|>"]]


# --------------------------------------------------------------------------- #
# failure on incompatible config / tokenizer
# --------------------------------------------------------------------------- #
def test_failure_on_incompatible_model_config():
    cfg = ModelConfig.small(vocab_size=VOCAB, context_length=8, dropout=0.0)
    model = make_model(cfg)
    # mismatched vocab
    with pytest.raises(ValueError):
        ModelConfig.small(vocab_size=VOCAB + 1, context_length=8, dropout=0.0).validate_tokenizer_vocab(VOCAB)


def test_failure_on_incompatible_checkpoint_config(tmp_path):
    cfg = ModelConfig.small(vocab_size=VOCAB, context_length=8, dropout=0.0)
    model = make_model(cfg)
    stream = write_stream(tmp_path / "s.bin", 200)
    ds = TokenBlockDataset(stream, context_length=8)
    from polycodegpt.training import TrainSettings, Trainer
    ts = TrainSettings(micro_batch_size=2, gradient_accumulation_steps=2, max_steps=1,
                       total_steps=10, precision="fp32", context_length=8, initial_validation=False)
    loader = build_dataloader(ds, batch_size=2, shuffle=True, seed=7, pin_memory=False)
    trainer = Trainer(model, loader, ts, device="cpu", log_fn=lambda *a, **k: None)
    path = tmp_path / "ckpt.pt"
    trainer.save(path)
    payload = load_checkpoint(path, map_location="cpu")
    # tamper model_config vocab (incompatible)
    payload["model_config"]["vocab_size"] = VOCAB + 1
    # rebuild must fail because weights no longer match the architecture
    with pytest.raises(RuntimeError):
        rebuild_model(payload, device="cpu")


# --------------------------------------------------------------------------- #
# tokenizer return type and tensor conversion (generation script)
# --------------------------------------------------------------------------- #
def test_tokenizer_encode_returns_encoding_with_ids():
    """tokenizer.encode() returns an Encoding; .ids gives ints usable by torch."""
    from tokenizers import Encoding
    tok_path = PROJECT_ROOT / "tokenizer" / "artifacts" / "polycodegpt_tokenizer_v1_1.json"
    if not tok_path.is_file():
        pytest.skip("tokenizer artifact not available")
    tok = load_tokenizer(tok_path, vocab_size=16000)
    prompt = "<|lang:python|><|docstring|>Compute factorial.\n<|code|>def factorial(n):\n    "
    enc = tok.encode(prompt, add_special_tokens=False)
    assert isinstance(enc, Encoding), type(enc)
    ids = enc.ids
    assert isinstance(ids, list)
    assert all(isinstance(x, int) for x in ids)
    assert len(ids) > 0
    # mirror the generation script conversion
    input_ids = torch.tensor([ids], dtype=torch.long)
    assert input_ids.dim() == 2
    assert input_ids.dtype == torch.long
    assert input_ids[0].tolist() == ids
    # round-trip decode
    assert tok.decode(ids, skip_special_tokens=False) == prompt