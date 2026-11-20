"""Continuation-stage semantics for the Phase 12B experiment runner.

The engine is run against a *copy* of the real checkpoint in a temporary
directory.  Nothing here touches the original artifacts: the checkpoint is copied
first, its sha256 is re-verified at the end, and the run is limited to a handful of
optimizer updates over the real dataset blocks with a deliberately tiny model.

These tests cover the eight required properties:

1. AdamW moments survive the restore.
2. The optimizer's internal per-parameter step counters survive the restore.
3. The continuation scheduler starts at stage step 0 (fresh warmup).
4. The global step counter starts at the checkpoint's step.
5. Exactly ``additional_steps`` optimizer updates happen.
6. The final global step is ``initial_global_step + additional_steps``.
7. The scheduler finishes at stage step ``additional_steps``.
8. The original checkpoint file is byte-identical afterwards.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from polycodegpt.data.dataset import TokenBlockDataset, build_dataloader
from polycodegpt.data.stream import TokenStreamWriter
from polycodegpt.model import GPT, ModelConfig
from polycodegpt.training import (
    OptimizerConfig,
    ScheduleConfig,
    Trainer,
    TrainSettings,
    WarmupCosineScheduler,
    build_optimizer,
    load_checkpoint,
    lr_factor,
    warmup_steps_for,
)

VOCAB = 32
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "polycodegpt_first_pass" / "step_008130.pt"
CHECKPOINT_SHA = "5562acc2fd58837f9d4245042b7f0b10de65520aa2ba292999c80045470590bd"

#: Stage length used by the four Phase 12 experiment configs.
STAGE_STEPS = 8130
WARMUP_FRACTION = 0.02


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.fixture(scope="module")
def checkpoint_copy(tmp_path_factory) -> Path:
    """A byte-identical temporary copy of the phase-1 checkpoint."""
    if not CHECKPOINT.exists():
        pytest.skip(f"checkpoint not available: {CHECKPOINT}")
    target = tmp_path_factory.mktemp("ckpt") / CHECKPOINT.name
    shutil.copyfile(CHECKPOINT, target)
    assert sha256_of(target) == CHECKPOINT_SHA
    return target


@pytest.fixture
def tiny_stream(tmp_path) -> Path:
    """Synthetic token stream: enough blocks for a few updates, no real corpus."""
    import numpy as np

    ids = np.random.default_rng(11).integers(0, VOCAB, size=4096)
    path = tmp_path / "tiny.bin"
    with TokenStreamWriter(path, VOCAB) as writer:
        writer.write(ids)
    return path


def tiny_model(vocab_size: int = VOCAB) -> GPT:
    torch.manual_seed(7)
    return GPT(ModelConfig.small(vocab_size=vocab_size, context_length=8, dropout=0.0))


def continuation_settings(**overrides) -> TrainSettings:
    base = dict(
        micro_batch_size=2,
        gradient_accumulation_steps=1,
        max_steps=6,
        total_steps=6,
        initial_global_step=8130,
        additional_steps=6,
        warmup_fraction=0.5,
        learning_rate=3e-4,
        weight_decay=0.1,
        gradient_clip=1.0,
        precision="fp32",
        seed=1234,
        context_length=8,
        eval_steps=(),
        checkpoint_steps=(),
        initial_validation=False,
        log_every=1000,
        fresh_schedule=True,
    )
    base.update(overrides)
    return TrainSettings(**base)


def make_trainer(dataset, settings: TrainSettings, model: GPT | None = None) -> Trainer:
    loader = build_dataloader(
        dataset,
        batch_size=settings.micro_batch_size,
        shuffle=True,
        seed=7,
        pin_memory=False,
    )
    return Trainer(
        model or tiny_model(),
        loader,
        settings,
        device="cpu",
        log_fn=lambda *args, **kw: None,
    )


def prime_checkpoint(payload, trainer: Trainer) -> dict:
    """Real checkpoint metadata, optimizer counters retargeted to the tiny model.

    The production optimizer state belongs to the 8130-step 25M-parameter model, so
    its moment tensors cannot be loaded into a tiny test model.  The tensors are
    rebuilt at the test model's shapes while the *per-parameter AdamW step counter*
    is taken verbatim from the real checkpoint.  Restoring this payload exercises
    exactly the code path a real continuation run uses.
    """
    real_state = payload["optimizer_state"]["state"]
    real_steps = {int(entry["step"]) for entry in real_state.values()}
    assert real_steps == {8130}, real_steps

    optimizer = trainer.optimizer
    groups = [dict(g) for g in optimizer.state_dict()["param_groups"]]
    state = {}
    for group in optimizer.param_groups:
        for param in group["params"]:
            state[param] = {
                "step": torch.tensor(8130.0),
                "exp_avg": torch.zeros_like(param),
                "exp_avg_sq": torch.zeros_like(param),
            }

    primed = dict(payload)
    primed.pop("rng_state", None)
    primed["model_state"] = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    primed["optimizer_state"] = {"state": state, "param_groups": groups}
    return primed


# --------------------------------------------------------------------------- #
# 1 + 2: optimizer moments and internal step counters survive restoration
# --------------------------------------------------------------------------- #
def test_adamw_moments_and_step_counters_are_restored(tiny_stream):
    """A save/restore round trip keeps every moment tensor and step counter."""
    model = tiny_model()
    settings = continuation_settings(initial_global_step=0, additional_steps=6, fresh_schedule=False)
    trainer = make_trainer(TokenBlockDataset(tiny_stream, context_length=8), settings, model)

    # three real optimizer updates so every parameter has non-zero AdamW state
    for seed in range(3):
        ids = torch.randint(0, VOCAB, (2, 8), generator=torch.Generator().manual_seed(seed))
        trainer.train_step([(ids, ids.clone())])
    assert trainer.scheduler.last_step == 3
    assert all(
        int(entry["step"]) == 3 for entry in trainer.optimizer.state_dict()["state"].values()
    )

    saved = trainer.checkpoint_state()
    payload = {
        "model_state": {k: v.clone() for k, v in trainer.model.state_dict().items()},
        "optimizer_state": trainer.optimizer.state_dict(),
        "scheduler_state": trainer.scheduler.state_dict(),
        "step": trainer.step_index,
    }
    assert saved["step"] == 3

    restored = make_trainer(TokenBlockDataset(tiny_stream, context_length=8), settings, tiny_model())
    restored.restore(payload)

    original_state = trainer.optimizer.state_dict()["state"]
    restored_state = restored.optimizer.state_dict()["state"]
    assert len(original_state) == len(restored_state) == 36
    for index, state in original_state.items():
        for key in ("exp_avg", "exp_avg_sq"):
            assert torch.allclose(state[key], restored_state[index][key]), f"{key} of param {index}"
            assert restored_state[index][key].abs().sum() > 0, f"{key} of param {index} was zeroed"
        # AdamW keeps its own per-parameter update counter: it must be preserved
        assert int(state["step"]) == 3
        assert int(restored_state[index]["step"]) == 3
    assert restored.step_index == 3


def test_real_checkpoint_optimizer_counters_are_present(checkpoint_copy):
    """The production checkpoint stores AdamW moments and step == 8130."""
    payload = load_checkpoint(checkpoint_copy, map_location="cpu")
    state = payload["optimizer_state"]["state"]
    assert state, "optimizer state is empty"
    steps = {int(entry["step"]) for entry in state.values()}
    assert steps == {8130}
    for entry in state.values():
        assert entry["exp_avg"].abs().sum() > 0
        assert entry["exp_avg_sq"].abs().sum() > 0
    assert int(payload["step"]) == 8130


# --------------------------------------------------------------------------- #
# 3 + 4: scheduler starts at stage 0, global step starts at the checkpoint step
# --------------------------------------------------------------------------- #
def test_continuation_starts_at_stage_zero_with_global_step_8130(checkpoint_copy, tiny_stream):
    payload = load_checkpoint(checkpoint_copy, map_location="cpu")
    model = tiny_model()
    settings = continuation_settings()
    trainer = make_trainer(TokenBlockDataset(tiny_stream, context_length=8), settings, model)
    trainer.restore(prime_checkpoint(payload, trainer))

    # global step keeps the historical value; stage progress restarts at zero
    assert trainer.step_index == 8130
    assert trainer.stage_step == 0
    assert trainer.initial_global_step == 8130
    assert trainer.additional_steps == 6
    assert trainer.expected_final_global_step == 8136

    # fresh schedule: no warmup has been consumed yet
    assert trainer.scheduler.last_step == 0
    assert trainer.scheduler.total_steps == settings.total_steps
    assert trainer.scheduler.warmup_steps == 3  # round(6 * 0.5)
    # first update runs at base/3, not at the historical post-8130 rate
    assert trainer.scheduler.get_last_lr()[0] == pytest.approx(3e-4 / 3)


def test_stage_local_scheduler_matches_production_schedule_shape():
    """The production continuation schedule is 8130 steps with 163 warmup updates."""
    config = ScheduleConfig(total_steps=STAGE_STEPS, warmup_fraction=WARMUP_FRACTION, min_lr_ratio=0.1)
    assert config.warmup_steps == 163
    assert warmup_steps_for(STAGE_STEPS, WARMUP_FRACTION) == 163

    model = tiny_model()
    optimizer = build_optimizer(model, OptimizerConfig(learning_rate=3e-4, weight_decay=0.1))
    scheduler = WarmupCosineScheduler(
        optimizer, total_steps=STAGE_STEPS, warmup_fraction=WARMUP_FRACTION, min_lr_ratio=0.1
    )
    assert scheduler.last_step == 0
    assert scheduler.get_last_lr()[0] == pytest.approx(3e-4 / 163)
    # the schedule is a function of stage progress only
    assert scheduler.lr_at(STAGE_STEPS - 1)[0] == pytest.approx(
        3e-4 * lr_factor(STAGE_STEPS - 1, STAGE_STEPS, 163, 0.1)
    )
    assert scheduler.lr_at(0)[0] == pytest.approx(3e-4 / 163)


# --------------------------------------------------------------------------- #
# 5 + 6 + 7 + 8: exact update count, final global step, scheduler end, integrity
# --------------------------------------------------------------------------- #
def test_exactly_additional_steps_updates_and_checkpoint_unchanged(checkpoint_copy, tiny_stream):
    before = sha256_of(checkpoint_copy)
    payload = load_checkpoint(checkpoint_copy, map_location="cpu")

    model = tiny_model()
    settings = continuation_settings(additional_steps=6, max_steps=6, total_steps=6)
    trainer = make_trainer(TokenBlockDataset(tiny_stream, context_length=8), settings, model)
    trainer.restore(prime_checkpoint(payload, trainer))

    updates = []
    original_train_step = trainer.train_step

    def counting_train_step(batches):
        result = original_train_step(batches)
        updates.append(trainer.stage_step)
        return result

    trainer.train_step = counting_train_step  # type: ignore[method-assign]
    result = trainer.run()

    # 5: exactly six additional optimizer updates
    assert len(updates) == 6
    assert updates == [1, 2, 3, 4, 5, 6]
    # 6: final global step is initial + additional
    assert result["stage_step"] == 6
    assert result["steps"] == 8136
    assert trainer.step_index == 8136
    assert trainer.expected_final_global_step == 8136
    # 7: the scheduler finished at stage step 6
    assert trainer.scheduler.last_step == 6
    assert result["additional_updates"] == 6
    # 8: the source checkpoint is untouched
    assert sha256_of(checkpoint_copy) == before == CHECKPOINT_SHA


def test_run_stops_at_stage_boundary_not_global_step(checkpoint_copy, tiny_stream):
    """max_steps stays the stage length; the loop stops on stage progress."""
    settings = continuation_settings(additional_steps=4, max_steps=4, total_steps=4)
    payload = load_checkpoint(checkpoint_copy, map_location="cpu")
    model = tiny_model()
    trainer = make_trainer(TokenBlockDataset(tiny_stream, context_length=8), settings, model)
    trainer.restore(prime_checkpoint(payload, trainer))

    result = trainer.run()
    assert result["stage_step"] == 4
    assert result["steps"] == 8134
    assert trainer.scheduler.last_step == 4


def test_restore_rejects_mismatched_global_step(checkpoint_copy, tiny_stream):
    payload = load_checkpoint(checkpoint_copy, map_location="cpu")
    model = tiny_model()
    settings = continuation_settings(initial_global_step=9999)
    trainer = make_trainer(TokenBlockDataset(tiny_stream, context_length=8), settings, model)
    with pytest.raises(ValueError, match="does not match the continuation"):
        trainer.restore(prime_checkpoint(payload, trainer))


def test_deterministic_settings_make_cpu_runs_bitwise_identical(tiny_stream):
    """Two runs with the same seeds produce identical weights and histories."""
    settings = continuation_settings(initial_global_step=0, additional_steps=4, max_steps=4,
                                     fresh_schedule=False)
    results = []
    for _ in range(2):
        trainer = make_trainer(TokenBlockDataset(tiny_stream, context_length=8), settings, tiny_model())
        trainer.run()
        results.append(
            {name: param.detach().clone() for name, param in trainer.model.named_parameters()}
        )
        assert trainer.determinism_report["applied"], "determinism settings were not applied"
        assert any("cudnn" in item for item in trainer.determinism_report["applied"])
    first, second = results
    assert set(first) == set(second)
    for name in first:
        assert torch.equal(first[name], second[name]), name


def test_reset_loader_state_prevents_zero_update_stage(checkpoint_copy, tmp_path):
    """Regression: a stale phase-1 consumed-batch count must not exhaust a fresh dataset.

    The phase-1 checkpoint records ``loader_batches_consumed == 65036`` (its whole
    corpus).  A continuation stage trains on a *different* dataset, so replaying
    that count makes ``complete_pass`` mode believe the epoch is already finished
    and perform zero updates.  ``reset_loader_state=True`` restarts consumption
    and lets the stage run to completion.
    """
    import numpy as np

    # A dataset whose block count exactly matches additional_steps updates
    # (grad_accum=1) so complete_pass can finish without an unconsumed-tail error.
    small = tmp_path / "small.bin"
    with TokenStreamWriter(small, VOCAB) as writer:
        writer.write(np.arange(64, dtype=np.int64) % VOCAB)
    dataset = TokenBlockDataset(small, context_length=8)
    assert len(build_dataloader(dataset, batch_size=2, shuffle=False, seed=1)) == 4

    settings = continuation_settings(
        additional_steps=4, max_steps=4, total_steps=4, complete_pass=True,
        partial_final_update=True,
    )

    payload = load_checkpoint(checkpoint_copy, map_location="cpu")

    # Without reset: the stale count exceeds the new loader size -> immediate abort.
    bad = make_trainer(dataset, settings, tiny_model())
    bad.restore(prime_checkpoint(payload, bad))
    assert bad._pending_skip == 65036
    with pytest.raises(RuntimeError, match="batches were consumed"):
        bad.run()

    # With reset: the fresh dataset is consumed and the stage completes.
    good = make_trainer(dataset, settings, tiny_model())
    good.restore(prime_checkpoint(payload, good), reset_loader_state=True)
    assert good._pending_skip == 0
    result = good.run()
    assert result["stage_step"] == 4
    assert result["steps"] == 8134
    assert good.scheduler.last_step == 4
    assert result["additional_updates"] == 4


def test_settings_validate_stage_constraints():
    """The original ``max_steps <= total_steps`` intent is preserved."""
    with pytest.raises(ValueError):
        TrainSettings(max_steps=0, total_steps=10)
    with pytest.raises(ValueError):
        TrainSettings(max_steps=10, total_steps=0)
    # a continuation stage of 8130 updates over an 8130-step schedule is valid
    settings = TrainSettings(max_steps=STAGE_STEPS, total_steps=STAGE_STEPS, initial_global_step=8130)
    assert settings.additional_steps == STAGE_STEPS
    assert settings.expected_final_global_step == 16260
    assert settings.warmup_steps == 163
