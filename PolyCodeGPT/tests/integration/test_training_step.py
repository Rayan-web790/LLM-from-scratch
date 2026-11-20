"""Integration test: a real (tiny) training run over real token streams.

The streams are generated with the project's own writer, the model is a small
``ModelConfig``, and the full Trainer path runs on CPU: data loading, gradient
accumulation, schedule, fixed-subset validation, metrics CSV and checkpoints.
A CUDA variant exercises the actual mixed-precision path when a GPU is present.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from polycodegpt.data.dataset import TokenBlockDataset, build_dataloader
from polycodegpt.data.stream import TokenStreamWriter
from polycodegpt.model import GPT, ModelConfig
from polycodegpt.training import (
    METRIC_COLUMNS,
    Trainer,
    TrainSettings,
    build_validation_loader,
    fixed_validation_indices,
    load_checkpoint,
    rebuild_model,
    weights_are_tied,
)

VOCAB = 32


def write_stream(path: Path, tokens: int, seed: int) -> Path:
    ids = np.random.default_rng(seed).integers(0, VOCAB, size=tokens)
    with TokenStreamWriter(path, VOCAB) as writer:
        writer.write(ids)
    return path


@pytest.fixture
def data(tmp_path):
    train = TokenBlockDataset(write_stream(tmp_path / "train.bin", 800, 1), context_length=8)
    valid = TokenBlockDataset(write_stream(tmp_path / "valid.bin", 200, 2), context_length=8)
    return train, valid


@pytest.fixture
def model_config():
    return ModelConfig.small(vocab_size=VOCAB, context_length=8, dropout=0.1)


def build_trainer(model_config, data, checkpoint_dir, max_steps=3, device="cpu", precision="fp32"):
    train_ds, valid_ds = data
    eval_steps = tuple(step for step in (2, 3) if step <= max_steps)
    checkpoint_steps = tuple(step for step in (max_steps - 1, max_steps) if step >= 1)
    settings = TrainSettings(
        micro_batch_size=2,
        gradient_accumulation_steps=2,
        max_steps=max_steps,
        total_steps=20,
        warmup_fraction=0.1,
        learning_rate=1e-3,
        precision=precision,
        seed=7,
        context_length=8,
        eval_steps=eval_steps,
        checkpoint_steps=checkpoint_steps,
        initial_validation=True,
        log_every=100,
    )
    torch.manual_seed(7)
    model = GPT(model_config)
    loader = build_dataloader(train_ds, batch_size=2, shuffle=True, seed=7)
    indices = fixed_validation_indices(len(valid_ds), 6, seed=7)
    val_loader = build_validation_loader(valid_ds, indices, batch_size=2)
    trainer = Trainer(
        model,
        loader,
        settings,
        device=device,
        val_loader=val_loader,
        train_tokens=train_ds.token_count,
        checkpoint_dir=checkpoint_dir,
        model_config=model_config,
        tokenizer_sha256="tokenizer-sha",
        data_manifest_sha256="manifest-sha",
        log_fn=lambda *args, **kwargs: None,
    )
    return trainer


def test_full_training_step_pipeline(model_config, data, tmp_path):
    train_ds, _ = data
    checkpoint_dir = tmp_path / "ckpts"
    trainer = build_trainer(model_config, data, checkpoint_dir, max_steps=3)

    before = [p.detach().clone() for p in trainer.model.parameters()]
    result = trainer.run()

    # optimizer steps, not micro-batches
    assert result["steps"] == 3
    assert trainer.step_index == 3
    assert trainer.micro_batches == 3 * 2
    assert trainer.tokens_seen == 3 * 2 * 2 * 8
    assert result["tokens_per_step"] == 32
    assert trainer.passes == 0  # one pass needs ~49 blocks; 12 micro-batches suffice

    # losses and gradients are finite throughout
    assert len(trainer.history) == 3
    assert all(np.isfinite(row["training_loss"]) for row in trainer.history)
    assert all(np.isfinite(row["gradient_norm"]) for row in trainer.history)
    assert all(row["optimizer_step"] == index + 1 for index, row in enumerate(trainer.history))
    assert trainer.skipped_steps == 0

    # schedule advanced once per successful update and warmup is in effect
    assert trainer.scheduler.last_step == 3
    assert trainer.history[0]["learning_rate"] < trainer.history[-1]["learning_rate"]

    # parameters actually changed
    after = list(trainer.model.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))
    assert all(torch.isfinite(p).all() for p in after)
    assert weights_are_tied(trainer.model)

    # validation: initial (step 0), step 2 and step 3, all on the same subset
    steps = [record["step"] for record in trainer.validation_history]
    assert steps == [0, 2, 3]
    token_counts = {record["tokens"] for record in trainer.validation_history}
    assert token_counts == {6 * 8}
    assert all(np.isfinite(r["loss"]) for r in trainer.validation_history)

    # checkpoints at the configured steps
    for name in ("step_000002.pt", "step_000003.pt", "latest.pt", "best.pt"):
        assert (checkpoint_dir / name).is_file(), name
    assert not list(checkpoint_dir.glob("*.partial"))

    payload = load_checkpoint(checkpoint_dir / "latest.pt", map_location="cpu")
    assert payload["step"] == 3
    assert payload["tokens_seen"] == trainer.tokens_seen
    assert payload["tokenizer_sha256"] == "tokenizer-sha"
    assert payload["data_manifest_sha256"] == "manifest-sha"
    assert payload["weights_tied"] is True
    assert payload["model_config"]["vocab_size"] == VOCAB

    restored = rebuild_model(payload, device="cpu")
    assert restored.lm_head.weight is restored.token_embedding.weight
    for key, value in payload["model_state"].items():
        assert torch.equal(restored.state_dict()[key], value), key

    # best checkpoint is the lowest validation loss
    best_payload = load_checkpoint(checkpoint_dir / "best.pt", map_location="cpu")
    losses = {r["step"]: r["loss"] for r in trainer.validation_history}
    assert best_payload["best_val_loss"] == pytest.approx(min(losses.values()))


def test_metrics_csv_matches_history(model_config, data, tmp_path):
    trainer = build_trainer(model_config, data, tmp_path / "ckpts", max_steps=2)
    trainer.run()

    path = tmp_path / "metrics.csv"
    from polycodegpt.training import write_metrics_csv

    write_metrics_csv(path, trainer.history)
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == METRIC_COLUMNS
        rows = list(reader)

    assert len(rows) == 2
    assert rows[-1]["optimizer_step"] == "2"
    assert rows[-1]["precision"] == "fp32"
    assert float(rows[-1]["tokens_seen"]) == trainer.tokens_seen
    assert rows[-1]["validation_loss"] != ""  # evaluated at step 2
    assert float(rows[0]["epoch_or_pass_fraction"]) > 0.0


def test_validation_uses_the_same_blocks_every_time(model_config, data, tmp_path):
    _, valid_ds = data
    indices = fixed_validation_indices(len(valid_ds), 6, seed=7)
    loader = build_validation_loader(valid_ds, indices, batch_size=2)

    seen_before = [int(batch[0][0, 0]) for batch in loader]
    trainer = build_trainer(model_config, data, tmp_path / "ckpts", max_steps=2)
    trainer.run()
    seen_after = [int(batch[0][0, 0]) for batch in loader]

    assert seen_before == seen_after
    assert all(r["tokens"] == 6 * 8 for r in trainer.validation_history)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_mixed_precision_step_updates_parameters(model_config, data, tmp_path):
    trainer = build_trainer(
        model_config, data, tmp_path / "ckpts_cuda", max_steps=1, device="cuda", precision="auto"
    )
    assert trainer.device.type == "cuda"
    assert trainer.plan.precision in ("bf16", "fp16")

    before = [p.detach().clone() for p in trainer.model.parameters()]
    result = trainer.run()

    assert result["steps"] == 1
    assert trainer.model.token_embedding.weight.device.type == "cuda"
    after = list(trainer.model.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))
    assert all(torch.isfinite(p).all() for p in after)
    assert (tmp_path / "ckpts_cuda" / "latest.pt").is_file()
    assert torch.cuda.max_memory_allocated() > 0
