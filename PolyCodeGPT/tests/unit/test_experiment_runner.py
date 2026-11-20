"""Tests for the Phase 12B experiment runner (scripts/run_phase12_experiment.py).

The runner is exercised end to end at the level that matters for approval: YAML
resolution, fail-closed behaviour, and the fact that the *existing* engine is
constructed with continuation semantics over the *prepared* dataset.  ``run()`` is
never called on the real model - no Phase 12C training happens here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

RUNNER_PATH = PROJECT_ROOT / "scripts" / "run_phase12_experiment.py"
spec = importlib.util.spec_from_file_location("phase12_runner", RUNNER_PATH)
runner = importlib.util.module_from_spec(spec)
sys.modules["phase12_runner"] = runner
spec.loader.exec_module(runner)  # type: ignore[union-attr]

ALL = ["control", "exp_a", "exp_b", "exp_c"]


# --------------------------------------------------------------------------- #
# resolution
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("experiment", ALL)
def test_every_config_resolves_to_the_frozen_stage_plan(experiment):
    plan = runner.resolve(runner.EXPERIMENT_FILES[experiment])
    assert plan.stage_steps == 8130
    assert plan.base_step == 8130
    assert plan.schedule_total_steps == 8130
    # the engine constraint is satisfied by construction, not bypassed
    assert plan.stage_steps <= plan.schedule_total_steps
    assert plan.config_max_steps == 16260 == plan.base_step + plan.stage_steps
    assert plan.warmup_steps == 163
    assert plan.effective_batch_size == 32
    assert plan.tokens_per_step == 16384
    assert plan.python_tokens + plan.javascript_tokens + plan.english_tokens == plan.total_tokens


@pytest.mark.parametrize("experiment", ALL)
def test_learning_rate_schedule_is_a_fresh_warmup_then_cosine(experiment):
    plan = runner.resolve(runner.EXPERIMENT_FILES[experiment])
    first = runner.lr_at(0, plan.learning_rate, plan.warmup_steps, plan.schedule_total_steps,
                         plan.min_lr_ratio)
    assert first == pytest.approx(plan.learning_rate / plan.warmup_steps)
    last = runner.lr_at(plan.schedule_total_steps - 1, plan.learning_rate, plan.warmup_steps,
                        plan.schedule_total_steps, plan.min_lr_ratio)
    assert last == pytest.approx(plan.learning_rate * plan.min_lr_ratio, rel=1e-3)
    # monotonically rising during warmup
    rates = [runner.lr_at(s, plan.learning_rate, plan.warmup_steps, plan.schedule_total_steps,
                          plan.min_lr_ratio) for s in range(plan.warmup_steps)]
    assert all(b > a for a, b in zip(rates, rates[1:]))


def test_max_steps_mismatch_is_rejected(tmp_path):
    original = yaml.safe_load(runner.EXPERIMENT_FILES["control"].read_text(encoding="utf-8"))
    original["training"]["max_steps"] = 9999
    broken = tmp_path / "broken.yaml"
    broken.write_text(yaml.safe_dump(original), encoding="utf-8")
    with pytest.raises(SystemExit, match="max_steps"):
        runner.resolve(broken)


def test_schedule_length_mismatch_is_rejected(tmp_path):
    original = yaml.safe_load(runner.EXPERIMENT_FILES["control"].read_text(encoding="utf-8"))
    original["schedule"]["total_steps"] = 16260
    broken = tmp_path / "broken.yaml"
    broken.write_text(yaml.safe_dump(original), encoding="utf-8")
    with pytest.raises(SystemExit, match="total_steps"):
        runner.resolve(broken)


# --------------------------------------------------------------------------- #
# fail-closed behaviour
# --------------------------------------------------------------------------- #
def test_train_without_confirmation_exits(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_phase12_experiment.py", "--experiment", "control", "--train"])
    with pytest.raises(SystemExit, match="confirm-training"):
        runner.main()


def test_unprepared_dataset_blocks_training(tmp_path, monkeypatch):
    plan = runner.resolve(runner.EXPERIMENT_FILES["control"])
    monkeypatch.setattr(type(plan), "dataset_manifest_path",
                        property(lambda self: tmp_path / "missing.json"), raising=False)
    monkeypatch.setattr(type(plan), "train_bin",
                        property(lambda self: tmp_path / "missing.bin"), raising=False)
    with pytest.raises(SystemExit, match="dataset not prepared"):
        runner.verify_prepared_dataset(plan)


# --------------------------------------------------------------------------- #
# one-update smoke mode
# --------------------------------------------------------------------------- #
def test_smoke_requires_its_own_confirmation(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["run_phase12_experiment.py", "--experiment", "control", "--smoke"]
    )
    with pytest.raises(SystemExit, match="confirm-smoke"):
        runner.main()


def test_smoke_and_train_cannot_be_combined(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_phase12_experiment.py", "--experiment", "control", "--smoke",
         "--confirm-smoke", "--train"],
    )
    with pytest.raises(SystemExit, match="either --smoke or --train"):
        runner.main()


def test_smoke_refuses_a_non_control_dataset():
    plan = runner.resolve(runner.EXPERIMENT_FILES["exp_b"])
    with pytest.raises(SystemExit, match="control training data"):
        runner.run_smoke_test(plan, device="cpu", log_fn=lambda *a, **k: None)


@pytest.mark.skipif(not runner.BASE_CHECKPOINT.is_file(), reason="checkpoint not available")
def test_smoke_settings_are_single_update_and_cannot_write():
    plan = runner.resolve(runner.EXPERIMENT_FILES["control"])
    if not plan.dataset_manifest_path.is_file() or not plan.train_bin.is_file():
        pytest.skip("control dataset not prepared")

    trainer, _ = runner.build_trainer(plan, device="cpu", log_fn=lambda *a, **k: None, smoke=True)

    assert trainer.additional_steps == 1
    assert trainer.settings.max_steps == 1
    assert trainer.settings.max_steps <= trainer.settings.total_steps == 8130
    assert trainer.settings.complete_pass is False  # one update cannot finish a pass
    assert trainer.settings.partial_final_update is True
    assert trainer.checkpoint_dir is None  # nothing can be saved
    assert trainer.val_loader is None
    assert trainer.step_index == 8130 and trainer.stage_step == 0
    assert trainer.scheduler.last_step == 0
    assert trainer.scheduler.get_last_lr()[0] == pytest.approx(3e-4 / 163)
    assert trainer.settings.micro_batch_size == 4
    assert trainer.settings.gradient_accumulation_steps == 8
    assert runner._optimizer_step_counters(trainer.optimizer) == [8130]


@pytest.mark.skipif(not runner.BASE_CHECKPOINT.is_file(), reason="checkpoint not available")
def test_smoke_run_on_cpu_performs_exactly_one_verified_update():
    plan = runner.resolve(runner.EXPERIMENT_FILES["control"])
    if not plan.dataset_manifest_path.is_file() or not plan.train_bin.is_file():
        pytest.skip("control dataset not prepared")

    report = runner.run_smoke_test(plan, device="cpu", log_fn=lambda *a, **k: None)

    assert report["passed"] is True, report["failed_checks"]
    assert all(report["checks"].values())
    assert report["checkpoint_sha256_before"] == report["checkpoint_sha256_after"]
    assert report["output_dir_unchanged"] is True
    assert report["global_step_after"] == 8131 and report["stage_step_after"] == 1
    assert report["optimizer_step_before"] == [8130]
    assert report["optimizer_step_after"] == [8131]
    assert report["micro_batches_this_update"] == 8
    assert report["sequences_this_update"] == 32
    assert report["tokens_this_update"] == 16384
    assert report["parameters_changed"] == report["parameters_total"]
    assert report["max_parameter_delta"] > 0
    # smoke never writes a checkpoint; the directory may already hold the real
    # control run's *.pt files (preserved evidence) and that is acceptable.


def _primed_payload(checkpoint_payload, trainer):
    """Real counters, tiny-model optimizer tensors (same shape-adaptation as the engine tests)."""
    import torch

    state = {}
    for group in trainer.optimizer.param_groups:
        for param in group["params"]:
            state[param] = {
                "step": torch.tensor(float(runner.BASE_STEP)),
                "exp_avg": torch.zeros_like(param),
                "exp_avg_sq": torch.zeros_like(param),
            }
    payload = dict(checkpoint_payload)
    payload.pop("rng_state", None)
    payload["model_state"] = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    payload["optimizer_state"] = {
        "state": state,
        "param_groups": trainerdict(trainer),
    }
    return payload


def trainerdict(trainer):
    import copy

    return [copy.deepcopy(group) for group in trainer.optimizer.state_dict()["param_groups"]]


def test_production_train_command_performs_updates_and_reports_them(tmp_path, monkeypatch):
    """End-to-end `main() --train --confirm-training` on a 3-update stage.

    Only the heavy components are substituted (tiny model, synthetic blocks, primed
    checkpoint payload); the command path, the reset-loader wiring, ``run()`` and the
    written summary are the production ones.  This is the regression that the
    smoke-only helper could not catch.
    """
    import dataclasses
    import json
    import shutil

    import numpy as np
    import torch

    from polycodegpt.data import TokenBlockDataset, build_dataloader
    from polycodegpt.data.stream import TokenStreamWriter
    from polycodegpt.model import GPT, ModelConfig
    from polycodegpt.training import Trainer, TrainSettings, load_checkpoint

    base = runner.resolve(runner.EXPERIMENT_FILES["control"])
    if not runner.BASE_CHECKPOINT.is_file():
        pytest.skip("checkpoint not available")

    updates = 3
    stage_plan = dataclasses.replace(
        base, stage_steps=updates, schedule_total_steps=updates, warmup_steps=1,
        config_max_steps=base.base_step + updates, eval_steps=(), checkpoint_steps=(),
    )
    out_dir = tmp_path / "control_run"
    monkeypatch.setattr(type(stage_plan), "output_dir", property(lambda self: out_dir), raising=False)

    # 96 blocks = 3 updates x 8 micro-batches x 4 sequences (context 8)
    stream = tmp_path / "tiny.bin"
    with TokenStreamWriter(stream, 32) as writer:
        writer.write(np.random.default_rng(5).integers(0, 32, size=96 * 8 + 1))
    dataset = TokenBlockDataset(stream, context_length=8, vocab_size=32)
    assert dataset.num_blocks == 96

    torch.manual_seed(7)
    model = GPT(ModelConfig.small(vocab_size=32, context_length=8, dropout=0.0))
    settings = TrainSettings(
        micro_batch_size=4, gradient_accumulation_steps=8, max_steps=updates,
        total_steps=updates, warmup_fraction=1 / updates, learning_rate=3e-4,
        weight_decay=0.1, gradient_clip=1.0, precision="fp32", seed=42, context_length=8,
        eval_steps=(), checkpoint_steps=(), initial_validation=False, log_every=1,
        complete_pass=True, partial_final_update=True, initial_global_step=runner.BASE_STEP,
        additional_steps=updates, fresh_schedule=True, deterministic=True,
    )
    loader = build_dataloader(dataset, batch_size=4, shuffle=True, seed=42, pin_memory=False)
    trainer = Trainer(model, loader, settings, device="cpu", train_tokens=dataset.token_count,
                      log_fn=lambda *a, **k: None)

    payload = load_checkpoint(runner.BASE_CHECKPOINT, map_location="cpu")
    assert int(payload["loader_batches_consumed"]) > len(loader)  # stale phase-1 count

    def fake_build(plan, *, device=None, log_fn=print, smoke=False, reset_loader_state=False):
        assert smoke is False
        assert reset_loader_state is True, "--train must reset the loader resume state"
        trainer.restore(_primed_payload(payload, trainer), reset_loader_state=reset_loader_state)
        return trainer, payload

    monkeypatch.setattr(runner, "build_trainer", fake_build)
    monkeypatch.setattr(runner, "resolve", lambda _path: stage_plan)
    monkeypatch.setattr(
        sys, "argv",
        ["run_phase12_experiment.py", "--experiment", "control", "--train",
         "--confirm-training", "--device", "cpu"],
    )

    runner.main()

    summary = json.loads((out_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["stage_step"] == updates
    assert summary["additional_updates"] == updates
    assert summary["steps"] == runner.BASE_STEP + updates
    assert summary["initial_global_step"] == runner.BASE_STEP
    assert summary["expected_final_global_step"] == runner.BASE_STEP + updates
    assert summary["skipped_steps"] == 0
    assert summary["elapsed_seconds"] > 0
    assert summary["micro_batches"] == 65036 + updates * 8
    assert summary["sequences"] == 260142 + updates * 32
    # tiny model context is 8, so each update consumed 32 sequences x 8 tokens
    assert summary["tokens_seen"] == 133192704 + updates * 32 * 8
    assert trainer.step_index == runner.BASE_STEP + updates
    assert trainer.scheduler.last_step == updates
    assert len(trainer.history) == 8130 + updates


def test_train_path_resets_loader_state_before_running():
    """The production --train path must not replay phase-1's consumed-batch count."""
    plan = runner.resolve(runner.EXPERIMENT_FILES["control"])
    if not plan.dataset_manifest_path.is_file() or not plan.train_bin.is_file():
        pytest.skip("control dataset not prepared")

    # build like --train does
    trainer, _ = runner.build_trainer(
        plan, device="cpu", log_fn=lambda *a, **k: None, reset_loader_state=True
    )
    # loader resume state must be cleared so complete_pass can consume the new dataset
    assert trainer._pending_skip == 0
    assert trainer._epoch_batches_consumed == 0
    assert trainer._epoch_start_state is None
    assert trainer.settings.complete_pass is True
    assert trainer.additional_steps == 8130
    # a run() call is never issued here; the wiring alone is what --train relies on.
    # (The output dir may already contain the real control run's artifacts,
    # including *.pt checkpoints and run_summary.json; that evidence is preserved.)


def test_training_output_may_not_overlap_protected_paths(monkeypatch):
    plan = runner.resolve(runner.EXPERIMENT_FILES["control"])
    monkeypatch.setattr(type(plan), "output_dir",
                        property(lambda self: PROJECT_ROOT / "tokenizer" / "runs"), raising=False)
    with pytest.raises(SystemExit, match="protected path"):
        runner.assert_output_paths_are_safe(plan)


def test_training_output_default_is_the_experiment_run_directory():
    for experiment in ALL:
        plan = runner.resolve(runner.EXPERIMENT_FILES[experiment])
        runner.assert_output_paths_are_safe(plan)
        assert plan.output_dir.parts[-1] == experiment
        assert plan.output_dir != plan.dataset_dir


# --------------------------------------------------------------------------- #
# engine wiring (no training)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not runner.BASE_CHECKPOINT.is_file(),
    reason="checkpoint not available",
)
def test_build_trainer_uses_the_existing_engine_with_continuation_state():
    plan = runner.resolve(runner.EXPERIMENT_FILES["control"])
    if not plan.dataset_manifest_path.is_file() or not plan.train_bin.is_file():
        pytest.skip("control dataset not prepared")

    trainer, payload = runner.build_trainer(plan, device="cpu", log_fn=lambda *a, **k: None)

    # the project's own trainer, not a second loop
    assert type(trainer).__module__ == "polycodegpt.training.engine"
    assert trainer.settings.complete_pass is True
    assert trainer.settings.partial_final_update is True

    # continuation accounting
    assert trainer.step_index == 8130
    assert trainer.stage_step == 0
    assert trainer.initial_global_step == 8130
    assert trainer.additional_steps == 8130
    assert trainer.expected_final_global_step == 16260

    # fresh stage scheduler: step 0 of a 8,130-step / 163-warmup cosine
    assert trainer.scheduler.last_step == 0
    assert trainer.scheduler.total_steps == 8130
    assert trainer.scheduler.warmup_steps == 163
    assert trainer.scheduler.get_last_lr()[0] == pytest.approx(3e-4 / 163)

    # AdamW moments and internal counters came from the checkpoint
    steps = {int(entry["step"]) for entry in trainer.optimizer.state_dict()["state"].values()}
    assert steps == {8130}
    for entry in list(trainer.optimizer.state_dict()["state"].values())[:5]:
        assert entry["exp_avg"].abs().sum() > 0
        assert entry["exp_avg_sq"].abs().sum() > 0

    # the dataset is the prepared experiment artifact, and one pass is 8,130 updates
    dataset = trainer.train_loader.dataset
    assert Path(dataset.bin_path) == plan.train_bin
    assert dataset.num_blocks == 260142
    assert dataset.num_blocks // 32 == 8129
    assert dataset.num_blocks % 32 == 14

    # weights really are the checkpoint's
    import torch

    for key, value in list(payload["model_state"].items())[:5]:
        assert torch.equal(trainer.model.state_dict()[key].cpu(), value.cpu()), key        # building the trainer never wrote a checkpoint (the dir may already hold
        # the real control run's preserved *.pt artifacts)
