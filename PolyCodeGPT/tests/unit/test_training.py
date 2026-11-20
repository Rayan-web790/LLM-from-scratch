"""Unit tests for the Phase 5B training engine.

Everything here runs on CPU with a tiny model and synthetic token streams; the
real corpus, tokenizer and GPU are not required. Each test targets one piece of
the contract: parameter grouping, accumulation arithmetic, the schedule,
token-weighted validation, and checkpoint save/restore/resume.
"""

from __future__ import annotations

import csv
import json
import math
import random
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
    OptimizerConfig,
    Trainer,
    TrainSettings,
    WarmupCosineScheduler,
    build_optimizer,
    build_parameter_groups,
    build_validation_loader,
    checkpoint_metadata,
    evaluate_loss,
    fixed_validation_indices,
    load_checkpoint,
    load_weights_into,
    lr_factor,
    parameter_group_report,
    perplexity,
    rebuild_model,
    require_cuda,
    resolve_precision,
    should_decay,
    warmup_steps_for,
    weights_are_tied,
    write_metrics_csv,
)

VOCAB = 32


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #
@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig.small(vocab_size=VOCAB, context_length=8, dropout=0.0)


def make_model(config: ModelConfig, seed: int = 1234) -> GPT:
    """Same weights on every call, so two runs can be compared exactly."""
    torch.manual_seed(seed)
    return GPT(config)


def write_stream(path: Path, tokens: int = 400, vocab: int = VOCAB, seed: int = 0) -> Path:
    ids = np.random.default_rng(seed).integers(0, vocab, size=tokens)
    with TokenStreamWriter(path, vocab) as writer:
        writer.write(ids)
    return path


def batch_of(config: ModelConfig, size: int = 2, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, config.vocab_size, (size, config.context_length), generator=generator)
    return input_ids, input_ids.clone()


def make_settings(**overrides) -> TrainSettings:
    base = dict(
        micro_batch_size=2,
        gradient_accumulation_steps=2,
        max_steps=1,
        total_steps=50,
        warmup_fraction=0.1,
        learning_rate=1e-3,
        weight_decay=0.1,
        gradient_clip=1.0,
        precision="fp32",
        seed=1234,
        context_length=8,
        eval_steps=(),
        checkpoint_steps=(),
        initial_validation=False,
        log_every=100,
    )
    base.update(overrides)
    return TrainSettings(**base)


def make_trainer(config: ModelConfig, dataset, settings: TrainSettings, **kwargs) -> Trainer:
    loader = kwargs.pop(
        "train_loader",
        build_dataloader(dataset, batch_size=settings.micro_batch_size, shuffle=True, seed=7, pin_memory=False),
    )
    model = kwargs.pop("model", make_model(config))
    return Trainer(
        model,
        loader,
        settings,
        device="cpu",
        log_fn=lambda *args, **kw: None,
        **kwargs,
    )


class StubModel(torch.nn.Module):
    """Returns a preset loss per batch (used to test token weighting)."""

    def __init__(self, losses: list[float]) -> None:
        super().__init__()
        self._losses = iter(losses)
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, targets=None, attn_mask=None):
        return None, torch.tensor(next(self._losses), requires_grad=True)


# --------------------------------------------------------------------------- #
# 1-2. optimizer grouping and weight decay exclusion
# --------------------------------------------------------------------------- #
def test_parameter_groups_split_weights_and_biases(model_config):
    model = make_model(model_config)
    groups = build_parameter_groups(model, weight_decay=0.1)

    assert [g["group_name"] for g in groups] == ["decay", "no_decay"]
    assert groups[0]["weight_decay"] == pytest.approx(0.1)
    assert groups[1]["weight_decay"] == 0.0

    decay_ids = {id(p) for p in groups[0]["params"]}
    no_decay_ids = {id(p) for p in groups[1]["params"]}
    assert decay_ids.isdisjoint(no_decay_ids)

    model_ids = {id(p) for p in model.parameters()}
    assert decay_ids | no_decay_ids == model_ids
    # the tied output projection is the embedding matrix: counted once
    assert len(decay_ids) + len(no_decay_ids) == len(model_ids)

    assert all(p.dim() >= 2 for p in groups[0]["params"])
    assert all(p.dim() < 2 for p in groups[1]["params"])


def test_weight_decay_excludes_biases_and_layer_norms(model_config):
    model = make_model(model_config)
    named = dict(model.named_parameters())

    assert should_decay("token_embedding.weight", named["token_embedding.weight"]) is True
    assert should_decay("position_embedding.weight", named["position_embedding.weight"]) is True
    assert should_decay("blocks.0.attn.q_proj.weight", named["blocks.0.attn.q_proj.weight"]) is True
    assert should_decay("blocks.0.attn.q_proj.bias", named["blocks.0.attn.q_proj.bias"]) is False
    assert should_decay("blocks.0.ln1.weight", named["blocks.0.ln1.weight"]) is False
    assert should_decay("ln_f.bias", named["ln_f.bias"]) is False

    optimizer = build_optimizer(model, OptimizerConfig(learning_rate=3e-4, weight_decay=0.1))
    decay_group, no_decay_group = optimizer.param_groups
    assert decay_group["weight_decay"] == pytest.approx(0.1)
    assert no_decay_group["weight_decay"] == 0.0
    assert no_decay_group["lr"] == pytest.approx(3e-4)

    report = parameter_group_report(optimizer)
    assert report[0]["parameters"] == sum(p.numel() for p in decay_group["params"])
    assert report[1]["parameters"] == sum(p.numel() for p in no_decay_group["params"])
    assert report[0]["parameters"] + report[1]["parameters"] == sum(
        p.numel() for p in model.parameters()
    )


# --------------------------------------------------------------------------- #
# 3-4. gradient accumulation arithmetic and loss scaling
# --------------------------------------------------------------------------- #
def test_accumulation_gradients_match_the_large_batch(model_config):
    model_full = make_model(model_config)
    model_accum = make_model(model_config)

    input_ids, targets = batch_of(model_config, size=4, seed=1)

    _, loss_full = model_full(input_ids, targets=targets)
    loss_full.backward()

    accum = 2
    for start in range(0, 4, 2):
        _, loss = model_accum(input_ids[start : start + 2], targets=targets[start : start + 2])
        (loss / accum).backward()

    for (name, full), (_, accumulated) in zip(
        model_full.named_parameters(), model_accum.named_parameters()
    ):
        assert full.grad is not None and accumulated.grad is not None, name
        # sum of two half-batch means equals the mean over the full batch
        assert torch.allclose(full.grad, accumulated.grad, atol=1e-6), name


def test_step_reports_mean_micro_batch_loss(model_config, tmp_path):
    dataset = TokenBlockDataset(write_stream(tmp_path / "s.bin", 400), context_length=8)
    settings = make_settings(micro_batch_size=2, gradient_accumulation_steps=2)
    reference = make_model(model_config)

    batches = [batch_of(model_config, size=2, seed=i) for i in range(2)]
    with torch.no_grad():
        expected = float(
            np.mean([reference(x, y)[1].item() for x, y in batches])
        )

    trainer = make_trainer(model_config, dataset, settings)
    result = trainer.train_step(batches)

    assert result["stepped"] is True
    assert result["micro_batches"] == 2
    assert result["loss"] == pytest.approx(expected, rel=1e-5)
    assert trainer.step_index == 1
    assert trainer.tokens_seen == 2 * 2 * settings.context_length
    assert trainer.micro_batches == 2


# --------------------------------------------------------------------------- #
# 5. gradient clipping
# --------------------------------------------------------------------------- #
def test_gradient_clipping_limits_the_total_norm():
    params = [torch.nn.Parameter(torch.full((4,), 10.0)), torch.nn.Parameter(torch.full((4,), -10.0))]
    for p in params:
        p.grad = torch.full_like(p, 5.0)

    norm_before = torch.nn.utils.clip_grad_norm_(params, 1.0)
    assert norm_before.item() == pytest.approx(math.sqrt(8 * 25.0))

    norm_after = torch.sqrt(sum((p.grad**2).sum() for p in params))
    assert norm_after.item() == pytest.approx(1.0, abs=1e-5)


# --------------------------------------------------------------------------- #
# 6-7. schedule
# --------------------------------------------------------------------------- #
def test_scheduler_warmup_is_linear(model_config):
    optimizer = build_optimizer(make_model(model_config), OptimizerConfig(learning_rate=1e-3))
    scheduler = WarmupCosineScheduler(optimizer, total_steps=100, warmup_steps=10)

    lrs = [scheduler.lr_at(step)[0] for step in range(10)]
    assert lrs[0] == pytest.approx(1e-3 / 10)
    assert all(later > earlier for earlier, later in zip(lrs, lrs[1:]))
    assert lrs[-1] == pytest.approx(1e-3)
    # the scheduled rate at construction is the first update's rate
    assert scheduler.get_last_lr()[0] == pytest.approx(1e-4)


def test_scheduler_cosine_decay_is_monotonic(model_config):
    optimizer = build_optimizer(make_model(model_config), OptimizerConfig(learning_rate=1e-3))
    scheduler = WarmupCosineScheduler(
        optimizer, total_steps=100, warmup_steps=10, min_lr_ratio=0.1
    )
    decay = [scheduler.lr_at(step)[0] for step in range(10, 101)]
    assert all(later <= earlier + 1e-12 for earlier, later in zip(decay, decay[1:]))
    # the first decay step sits just below the base rate (cosine starts at 1)
    assert decay[0] == pytest.approx(1e-3, rel=1e-3)
    assert decay[-1] == pytest.approx(1e-4, rel=1e-6)
    # midway the cosine is at half the distance between base and floor
    mid = scheduler.lr_at(55)[0]
    assert 1e-4 < mid < 1e-3


def test_scheduler_only_advances_on_step(model_config):
    optimizer = build_optimizer(make_model(model_config), OptimizerConfig(learning_rate=1e-3))
    scheduler = WarmupCosineScheduler(optimizer, total_steps=10, warmup_steps=2)

    assert scheduler.last_step == 0
    scheduler.step()
    assert scheduler.last_step == 1
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)

    for _ in range(20):  # never past the planned length
        scheduler.step()
    assert scheduler.last_step == 10
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)


def test_schedule_configuration_is_respected():
    assert warmup_steps_for(8129, 0.02) == 163
    assert warmup_steps_for(100, 0.0) == 0
    assert lr_factor(0, 100, 10, 0.1) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        warmup_steps_for(0, 0.02)
    with pytest.raises(ValueError):
        lr_factor(-1, 100, 10, 0.1)


# --------------------------------------------------------------------------- #
# 8-10. validation
# --------------------------------------------------------------------------- #
def test_validation_runs_without_grad(model_config, tmp_path):
    dataset = TokenBlockDataset(write_stream(tmp_path / "s.bin", 200), context_length=8)
    model = make_model(model_config)
    loader = build_validation_loader(dataset, list(range(4)), batch_size=2)
    before = [p.detach().clone() for p in model.parameters()]

    model.train()
    result = evaluate_loss(model, loader, "cpu")

    assert model.training is True  # previous mode restored
    assert all(p.grad is None for p in model.parameters())
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))
    assert result["tokens"] == 4 * 8
    assert math.isfinite(result["loss"])
    assert result["perplexity"] == pytest.approx(math.exp(result["loss"]))


def test_validation_is_deterministic(model_config, tmp_path):
    dataset = TokenBlockDataset(write_stream(tmp_path / "s.bin", 200), context_length=8)
    model = make_model(model_config)
    loader = build_validation_loader(dataset, [0, 1, 2, 3], batch_size=2)

    first = evaluate_loss(model, loader, "cpu")
    second = evaluate_loss(model, loader, "cpu")
    assert first["loss"] == pytest.approx(second["loss"], rel=0, abs=0)
    assert first["tokens"] == second["tokens"]


def test_fixed_validation_subset_is_stable():
    a = fixed_validation_indices(100, 10, seed=42)
    b = fixed_validation_indices(100, 10, seed=42)
    assert a == b and len(a) == 10 and a == sorted(a)
    assert fixed_validation_indices(100, 10, seed=43) != a
    assert fixed_validation_indices(5, 10, seed=0) == [0, 1, 2, 3, 4]


def test_validation_loss_is_token_weighted():
    model = StubModel([2.0, 4.0])
    # batch 0: 4 valid targets; batch 1: only 2 valid targets of 4
    targets_a = torch.tensor([[1, 2, 3, 4]])
    targets_b = torch.tensor([[1, 2, -100, -100]])
    batches = [
        (torch.zeros_like(targets_a), targets_a),
        (torch.zeros_like(targets_b), targets_b),
    ]

    result = evaluate_loss(model, batches, "cpu")
    weighted = (2.0 * 4 + 4.0 * 2) / 6
    simple_mean = (2.0 + 4.0) / 2

    assert result["loss"] == pytest.approx(weighted)
    assert result["loss"] != pytest.approx(simple_mean)
    assert result["tokens"] == 6


# --------------------------------------------------------------------------- #
# 11-15. checkpoints
# --------------------------------------------------------------------------- #
def test_checkpoint_roundtrip_is_weights_only(model_config, tmp_path):
    dataset = TokenBlockDataset(write_stream(tmp_path / "s.bin", 200), context_length=8)
    trainer = make_trainer(model_config, dataset, make_settings())
    path = tmp_path / "ckpt.pt"
    trainer.save(path)

    assert not (tmp_path / "ckpt.pt.partial").exists()
    payload = load_checkpoint(path, map_location="cpu")  # weights_only=True inside

    for key in (
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "step",
        "tokens_seen",
        "model_config",
        "tokenizer_sha256",
        "data_manifest_sha256",
        "settings",
        "rng_state",
        "weights_tied",
    ):
        assert key in payload, key

    meta = checkpoint_metadata(payload)
    assert meta["step"] == 0
    assert meta["parameter_count"] == sum(p.numel() for p in trainer.model.parameters())
    assert meta["weights_tied"] is True


def test_optimizer_state_is_restored(model_config, tmp_path):
    dataset = TokenBlockDataset(write_stream(tmp_path / "s.bin", 400), context_length=8)
    settings = make_settings(micro_batch_size=2, gradient_accumulation_steps=2, max_steps=2)
    trainer = make_trainer(model_config, dataset, settings)

    for _ in range(2):
        trainer.train_step([batch_of(model_config, size=2, seed=i) for i in range(2)])

    path = tmp_path / "step2.pt"
    trainer.save(path)
    payload = load_checkpoint(path, map_location="cpu")

    fresh = make_trainer(model_config, dataset, settings)
    assert fresh.optimizer.state_dict()["state"] == {}
    fresh.optimizer.load_state_dict(payload["optimizer_state"])

    restored_state = fresh.optimizer.state_dict()["state"]
    assert restored_state, "optimizer state should not be empty after restoration"
    for index, saved in payload["optimizer_state"]["state"].items():
        for key, value in saved.items():
            if torch.is_tensor(value):
                assert torch.equal(restored_state[index][key], value)


def test_scheduler_state_is_restored(model_config, tmp_path):
    dataset = TokenBlockDataset(write_stream(tmp_path / "s.bin", 400), context_length=8)
    settings = make_settings(max_steps=3, total_steps=20, warmup_fraction=0.2)
    trainer = make_trainer(model_config, dataset, settings)
    for _ in range(3):
        trainer.train_step([batch_of(model_config, size=2, seed=i) for i in range(2)])

    path = tmp_path / "step3.pt"
    trainer.save(path)
    payload = load_checkpoint(path, map_location="cpu")

    fresh = make_trainer(model_config, dataset, settings)
    fresh.scheduler.load_state_dict(payload["scheduler_state"])

    assert fresh.scheduler.last_step == 3
    assert fresh.scheduler.get_last_lr()[0] == pytest.approx(trainer.scheduler.get_last_lr()[0])


def test_weight_tying_survives_checkpoint_restoration(model_config, tmp_path):
    dataset = TokenBlockDataset(write_stream(tmp_path / "s.bin", 200), context_length=8)
    trainer = make_trainer(model_config, dataset, make_settings())
    path = tmp_path / "ckpt.pt"
    trainer.save(path)

    payload = load_checkpoint(path, map_location="cpu")
    restored = rebuild_model(payload, device="cpu")

    assert weights_are_tied(restored)
    assert restored.lm_head.weight is restored.token_embedding.weight
    for key, value in payload["model_state"].items():
        assert torch.equal(restored.state_dict()[key], value), key


def test_resume_matches_an_uninterrupted_run(model_config, tmp_path):
    stream = write_stream(tmp_path / "s.bin", 600, seed=3)
    settings = make_settings(micro_batch_size=2, gradient_accumulation_steps=2, max_steps=6)

    # uninterrupted reference run
    dataset_a = TokenBlockDataset(stream, context_length=8)
    reference = make_trainer(model_config, dataset_a, settings, model=make_model(model_config))
    reference.run()

    # identical run, interrupted after 3 steps and resumed from the checkpoint
    dataset_b = TokenBlockDataset(stream, context_length=8)
    first = make_trainer(model_config, dataset_b, settings, model=make_model(model_config))
    first.settings = make_settings(
        micro_batch_size=2, gradient_accumulation_steps=2, max_steps=3
    )
    first.run()
    path = tmp_path / "resume.pt"
    first.save(path)
    payload = load_checkpoint(path, map_location="cpu")
    assert payload["step"] == 3

    dataset_c = TokenBlockDataset(stream, context_length=8)
    resumed = make_trainer(model_config, dataset_c, settings, model=make_model(model_config))
    resumed.restore(payload)
    assert resumed.step_index == 3
    assert resumed.scheduler.last_step == 3
    result = resumed.run()

    assert result["steps"] == 6
    assert resumed.tokens_seen == reference.tokens_seen
    assert resumed.step_index == reference.step_index
    for (name, expected), actual in zip(reference.model.named_parameters(), resumed.model.parameters()):
        assert torch.allclose(expected, actual, atol=1e-7), name


def test_rng_state_is_restored(model_config, tmp_path):
    dataset = TokenBlockDataset(write_stream(tmp_path / "s.bin", 400), context_length=8)
    trainer = make_trainer(model_config, dataset, make_settings(max_steps=2))
    trainer.train_step([batch_of(model_config, size=2, seed=i) for i in range(2)])

    path = tmp_path / "rng.pt"
    trainer.save(path)
    payload = load_checkpoint(path, map_location="cpu")

    fresh = make_trainer(model_config, dataset, make_settings(max_steps=2))
    fresh.restore(payload)

    assert torch.equal(torch.get_rng_state(), payload["rng_state"]["torch"])
    assert fresh._loader_generator.get_state().equal(payload["rng_state"]["loader_generator"])

    # the python and numpy streams continue identically too
    expected_py = random.random()
    expected_np = float(np.random.rand())
    fresh.restore(payload)
    assert random.random() == expected_py
    assert float(np.random.rand()) == expected_np


# --------------------------------------------------------------------------- #
# 17-18. precision and effective batch
# --------------------------------------------------------------------------- #
def test_precision_selection_follows_gpu_capability():
    bf16 = resolve_precision("auto", torch.device("cuda"), bf16_supported=True)
    assert bf16.precision == "bf16"
    assert bf16.autocast_dtype is torch.bfloat16
    assert bf16.use_grad_scaler is False

    fp16 = resolve_precision("auto", torch.device("cuda"), bf16_supported=False)
    assert fp16.precision == "fp16"
    assert fp16.autocast_dtype is torch.float16
    assert fp16.use_grad_scaler is True

    fp32 = resolve_precision("fp32", torch.device("cuda"), bf16_supported=True)
    assert fp32.precision == "fp32" and fp32.autocast_dtype is None

    cpu = resolve_precision("auto", torch.device("cpu"))
    assert cpu.precision == "fp32" and cpu.device_type == "cpu"

    with pytest.raises(ValueError):
        resolve_precision("bf16", torch.device("cuda"), bf16_supported=False)
    with pytest.raises(ValueError):
        resolve_precision("fp16", torch.device("cpu"))
    with pytest.raises(ValueError):
        resolve_precision("int8", torch.device("cpu"))


def test_training_refuses_cpu_fallback():
    with pytest.raises(RuntimeError):
        require_cuda("cpu")


def test_effective_batch_arithmetic():
    settings = TrainSettings(micro_batch_size=4, gradient_accumulation_steps=8, context_length=512)
    assert settings.effective_batch_size == 32
    assert settings.tokens_per_step == 32 * 512 == 16_384

    fallback = TrainSettings(micro_batch_size=2, gradient_accumulation_steps=16, context_length=512)
    assert fallback.effective_batch_size == 32

    with pytest.raises(ValueError):
        TrainSettings(micro_batch_size=0)


def test_perplexity_and_loss_combination():
    assert perplexity(0.0) == pytest.approx(1.0)
    assert perplexity(math.log(10)) == pytest.approx(10.0)
    assert perplexity(1000.0) == float("inf")


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def test_metrics_csv_has_the_required_columns(tmp_path):
    rows = [
        {
            "optimizer_step": 1,
            "training_loss": 9.5,
            "validation_loss": "",
            "learning_rate": 1e-6,
            "gradient_norm": 0.8,
            "tokens_seen": 16384,
            "epoch_or_pass_fraction": 0.0001,
            "elapsed_seconds": 1.2,
            "allocated_vram": 100.0,
            "peak_allocated_vram": 200.0,
        }
    ]
    path = write_metrics_csv(tmp_path / "metrics.csv", rows)
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == METRIC_COLUMNS
        written = list(reader)

    assert len(written) == 1
    assert written[0]["tokens_seen"] == "16384"
    for column in (
        "optimizer_step",
        "epoch_or_pass_fraction",
        "training_loss",
        "validation_loss",
        "learning_rate",
        "gradient_norm",
        "tokens_seen",
        "elapsed_seconds",
        "allocated_vram",
        "peak_allocated_vram",
    ):
        assert column in METRIC_COLUMNS


# --------------------------------------------------------------------------- #
# final partial update ordering and throughput accounting (Phase 12C audit)
# --------------------------------------------------------------------------- #
def test_final_partial_update_is_applied_before_validation_and_checkpoint(tmp_path):
    """The final partial group must update weights before validation/checkpoint save.

    Regression for the Phase 12C audit: a 7-block stream yields one full update
    and one partial update of 3 sequences. The checkpoint written for the final
    step must contain the post-update parameters and optimizer state, and final
    validation must be recorded at the final step.
    """
    stream = tmp_path / "tiny.bin"
    ids = np.random.default_rng(3).integers(0, VOCAB, size=7 * 8 + 1)
    with TokenStreamWriter(stream, VOCAB) as writer:
        writer.write(ids)

    dataset = TokenBlockDataset(stream, context_length=8, vocab_size=VOCAB)
    assert dataset.num_blocks == 7

    config = ModelConfig.small(vocab_size=VOCAB, context_length=8, dropout=0.0)
    torch.manual_seed(1)
    model = GPT(config)

    settings = TrainSettings(
        micro_batch_size=2,
        gradient_accumulation_steps=2,
        max_steps=2,
        total_steps=2,
        warmup_fraction=0.5,
        learning_rate=1e-3,
        precision="fp32",
        seed=7,
        context_length=8,
        eval_steps=(),
        checkpoint_steps=(),
        initial_validation=False,
        log_every=100,
        complete_pass=True,
        partial_final_update=True,
        initial_global_step=0,
        additional_steps=2,
        fresh_schedule=False,
    )

    loader = build_dataloader(dataset, batch_size=2, shuffle=False, seed=7)
    indices = fixed_validation_indices(len(dataset), 4, seed=7)
    val_loader = build_validation_loader(dataset, indices, batch_size=2)

    ckpt_dir = tmp_path / "ckpts"
    trainer = Trainer(
        model,
        loader,
        settings,
        device="cpu",
        val_loader=val_loader,
        train_tokens=dataset.token_count,
        checkpoint_dir=ckpt_dir,
        model_config=config,
        tokenizer_sha256="tok",
        data_manifest_sha256="man",
        log_fn=lambda *args, **kwargs: None,
    )

    result = trainer.run()

    # both updates happened, including the partial one
    assert result["steps"] == 2
    assert trainer.step_index == 2
    assert trainer.stage_step == 2
    assert result["skipped_steps"] == 0
    assert trainer.scheduler.last_step == 2
    # the second group was the partial one (3 sequences, not 4)
    assert trainer.history[-1]["effective_batch_size"] == 3

    # final validation recorded at the final step (after the update)
    assert trainer.validation_history, "expected a final validation record"
    assert trainer.validation_history[-1]["step"] == trainer.step_index == 2

    # final checkpoint exists and is post-update
    final_path = ckpt_dir / "step_000002.pt"
    assert final_path.is_file()
    payload = load_checkpoint(final_path, map_location="cpu")
    assert payload["step"] == 2
    assert payload["stage_step"] == 2
    optimizer_steps = {int(v["step"]) for v in payload["optimizer_state"]["state"].values()}
    assert optimizer_steps == {2}, optimizer_steps
    for key, value in payload["model_state"].items():
        assert torch.equal(value, trainer.model.state_dict()[key]), key


def test_throughput_uses_stage_delta_not_cumulative_tokens(tmp_path):
    """tokens_per_second must divide the stage delta by elapsed time."""
    stream = tmp_path / "tiny.bin"
    ids = np.random.default_rng(5).integers(0, VOCAB, size=400)
    with TokenStreamWriter(stream, VOCAB) as writer:
        writer.write(ids)

    dataset = TokenBlockDataset(stream, context_length=8, vocab_size=VOCAB)
    config = ModelConfig.small(vocab_size=VOCAB, context_length=8, dropout=0.0)
    torch.manual_seed(2)
    model = GPT(config)

    settings = TrainSettings(
        micro_batch_size=2,
        gradient_accumulation_steps=2,
        max_steps=2,
        total_steps=2,
        warmup_fraction=0.5,
        learning_rate=1e-3,
        precision="fp32",
        seed=9,
        context_length=8,
        eval_steps=(),
        checkpoint_steps=(),
        initial_validation=False,
        log_every=100,
        complete_pass=False,
        partial_final_update=False,
        initial_global_step=0,
        additional_steps=2,
        fresh_schedule=False,
    )
    loader = build_dataloader(dataset, batch_size=2, shuffle=False, seed=9)
    trainer = Trainer(
        model,
        loader,
        settings,
        device="cpu",
        train_tokens=dataset.token_count,
        log_fn=lambda *args, **kwargs: None,
    )
    # simulate a restored continuation baseline: 50 tokens already seen
    trainer.initial_tokens_seen = 50

    result = trainer.run()

    assert result["tokens_seen"] > 50
    assert result["stage_tokens_seen"] == result["tokens_seen"] - 50
    expected = result["stage_tokens_seen"] / result["elapsed_seconds"]
    assert result["tokens_per_second"] == pytest.approx(expected, rel=1e-9)
    # cumulative figure is still reported separately
    assert result["initial_tokens_seen"] == 50
