#!/usr/bin/env python
"""PolyCodeGPT Phase 20 continuation trainer for Scenario B.

Starts from the verified Phase 19 checkpoint, preserves AdamW moments, and
uses a fresh warmup/cosine schedule. Phase 19 artifacts are never overwritten.

Usage (from project root):
  .venv_phase19\\Scripts\\python.exe scripts\\train_phase20.py --preflight-only
  .venv_phase19\\Scripts\\python.exe scripts\\train_phase20.py --smoke-test
  .venv_phase19\\Scripts\\python.exe scripts\\train_phase20.py
  .venv_phase19\\Scripts\\python.exe scripts\\train_phase20.py --resume-phase20
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

# Reuse Phase 19's audited dataset/tokenizer preflight and data helpers.
# Running this file as `python scripts/train_phase20.py` places `scripts/`
# on sys.path, so this import resolves without changing the legacy script.
import train_phase19 as phase19  # noqa: E402

from polycodegpt.training import (  # noqa: E402
    Trainer,
    TrainSettings,
    build_validation_loader,
    fixed_validation_indices,
    load_checkpoint,
    require_cuda,
    weights_are_tied,
    write_metrics_csv,
)

ROOT = phase19.ROOT
DATA_DIR = phase19.DATA_DIR
PHASE19_CHECKPOINT = phase19.CHECKPOINT_DIR / "latest.pt"
CONFIG_PATH = ROOT / "configs" / "phase20_training.yaml"
PHASE20_CHECKPOINT_DIR = ROOT / "checkpoints" / "polycodegpt_phase20_scenario_b"
PHASE20_LOG_DIR = ROOT / "logs" / "phase20_scenario_b"
SMOKE_CHECKPOINT_DIR = ROOT / "checkpoints" / "polycodegpt_phase20_smoke"
SMOKE_LOG_DIR = ROOT / "logs" / "phase20_smoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PolyCodeGPT Phase 20 Scenario B continuation")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true", help="verify artifacts/config without training")
    mode.add_argument("--smoke-test", action="store_true", help="restore Phase 19 and perform two isolated updates")
    mode.add_argument("--resume-phase20", action="store_true", help="resume an interrupted Phase 20 run")
    parser.add_argument("--device", default="cuda", help="training device (default: cuda)")
    return parser.parse_args()


def read_config() -> dict[str, Any]:
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Phase 20 config missing: {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict) or not isinstance(value.get("training"), dict):
        raise ValueError("Phase 20 configuration must contain a training mapping")
    return value


def make_settings(config: dict[str, Any], *, smoke: bool, resume: bool) -> TrainSettings:
    cfg = config["training"]
    eval_cfg = cfg.get("evaluation", {})

    if smoke:
        additional_steps = 2
        total_steps = 2
        warmup_fraction = 0.0
        min_lr_ratio = 1.0
        complete_pass = False
        partial_final_update = False
        eval_every = 0
        checkpoint_every = 0
    else:
        additional_steps = int(cfg["additional_steps"])
        total_steps = int(cfg["total_steps"])
        warmup_fraction = float(cfg.get("warmup_fraction", 0.02))
        min_lr_ratio = float(cfg.get("min_lr_ratio", 0.1))
        complete_pass = bool(cfg.get("complete_pass", True))
        partial_final_update = bool(cfg.get("partial_final_update", True))
        eval_every = int(eval_cfg.get("every_steps", 500))
        checkpoint_every = int(cfg.get("checkpoint_every", 1000))

    return TrainSettings(
        micro_batch_size=int(cfg.get("micro_batch_size", 4)),
        gradient_accumulation_steps=(
            int(cfg.get("effective_batch_size", 32)) // int(cfg.get("micro_batch_size", 4))
        ),
        max_steps=additional_steps,
        additional_steps=additional_steps,
        total_steps=total_steps,
        warmup_fraction=warmup_fraction,
        min_lr_ratio=min_lr_ratio,
        learning_rate=float(cfg.get("learning_rate", 1e-4)),
        weight_decay=float(cfg.get("weight_decay", 0.1)),
        gradient_clip=float(cfg.get("gradient_clip", 1.0)),
        precision=str(cfg.get("precision", "auto")),
        seed=int(cfg.get("seed", 42)),
        context_length=int(cfg.get("context_length", 512)),
        log_every=int(cfg.get("log_every", 100 if not smoke else 1)),
        eval_steps=(),
        checkpoint_steps=(),
        eval_every=eval_every,
        checkpoint_every=checkpoint_every,
        initial_validation=True,
        max_skipped_steps=int(cfg.get("max_skipped_steps", 10)),
        complete_pass=complete_pass,
        partial_final_update=partial_final_update,
        initial_global_step=int(cfg.get("initial_global_step", 200)),
        fresh_schedule=not resume,
        deterministic=bool(cfg.get("deterministic", True)),
    )


def verify_checkpoint_identity(
    checkpoint: dict[str, Any], tokenizer_sha: str, manifest_sha: str, *, expected_step: int | None
) -> None:
    if checkpoint.get("tokenizer_sha256") != tokenizer_sha:
        raise RuntimeError("checkpoint tokenizer SHA-256 does not match the current v2 tokenizer")
    if checkpoint.get("data_manifest_sha256") != manifest_sha:
        raise RuntimeError("checkpoint dataset-manifest SHA-256 does not match Scenario B")
    if expected_step is not None and int(checkpoint.get("step", -1)) != expected_step:
        raise RuntimeError(
            f"expected checkpoint global step {expected_step}, got {checkpoint.get('step')}"
        )
    if not checkpoint.get("optimizer_state", {}).get("state"):
        raise RuntimeError("checkpoint has no populated optimizer state; refusing continuation")
    if not checkpoint.get("weights_tied", False):
        raise RuntimeError("source checkpoint does not record tied embeddings")
    required = ("model_state", "optimizer_state", "scheduler_state", "rng_state", "model_config")
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise RuntimeError(f"checkpoint is missing required restoration fields: {missing}")


def compare_model_weights(model: torch.nn.Module, checkpoint: dict[str, Any]) -> bool:
    actual = model.state_dict()
    expected = checkpoint["model_state"]
    if actual.keys() != expected.keys():
        return False
    return all(torch.equal(actual[key].detach().cpu(), expected[key].detach().cpu()) for key in actual)


def run() -> int:
    args = parse_args()
    preflight_report, model_config, _manifest, tokenizer_sha, manifest_sha = phase19.preflight()
    config = read_config()
    cfg = config["training"]

    print(json.dumps(preflight_report, indent=2))
    if preflight_report.get("test_partition_opened", False):
        raise RuntimeError("preflight indicates the test partition was opened; refusing to continue")

    if args.preflight_only:
        blocks = int(preflight_report["train"]["blocks"])
        micro = int(cfg.get("micro_batch_size", 4))
        effective = int(cfg.get("effective_batch_size", 32))
        accumulation = effective // micro
        loader_batches = math.ceil(blocks / micro)
        full_updates, leftover_batches = divmod(loader_batches, accumulation)
        expected_updates = full_updates + (1 if leftover_batches else 0)
        print("\nPHASE 20 PREFLIGHT PASSED")
        print(f"train blocks: {blocks:,}")
        print(f"DataLoader batches: {loader_batches:,}")
        print(f"full optimizer updates: {full_updates:,}")
        print(f"leftover micro-batches: {leftover_batches}")
        print(f"expected optimizer updates including partial: {expected_updates:,}")
        print(f"configured additional updates: {cfg.get('additional_steps')}")
        print("test partition opened: no")
        return 0

    device = require_cuda(args.device)
    smoke = bool(args.smoke_test)
    resume = bool(args.resume_phase20)

    if smoke and resume:
        raise ValueError("--smoke-test and --resume-phase20 cannot be combined")

    source_checkpoint = (
        PHASE20_CHECKPOINT_DIR / "latest.pt" if resume else PHASE19_CHECKPOINT
    )
    if not source_checkpoint.is_file():
        raise FileNotFoundError(f"starting checkpoint not found: {source_checkpoint}")

    checkpoint = load_checkpoint(source_checkpoint, map_location="cpu")
    verify_checkpoint_identity(
        checkpoint,
        tokenizer_sha,
        manifest_sha,
        expected_step=None if resume else 200,
    )
    if checkpoint.get("model_config") != model_config.to_dict():
        raise RuntimeError("checkpoint model configuration does not match Phase 19 model configuration")
    if resume:
        saved_settings = checkpoint.get("settings", {})
        if int(saved_settings.get("initial_global_step", -1)) != int(cfg.get("initial_global_step", 200)):
            raise RuntimeError("Phase 20 checkpoint has an unexpected initial_global_step")
        if int(saved_settings.get("additional_steps", -1)) != int(cfg.get("additional_steps", -2)):
            raise RuntimeError("Phase 20 checkpoint training horizon differs from current config")

    settings = make_settings(config, smoke=smoke, resume=resume)
    if int(checkpoint["step"]) != settings.initial_global_step and not resume:
        raise RuntimeError("Phase 19 checkpoint step does not match configured initial_global_step")

    train_ds = phase19.TokenBlockDataset(
        DATA_DIR / "train.bin",
        context_length=settings.context_length,
        vocab_size=model_config.vocab_size,
        stride=phase19.WINDOW_STRIDE,
    )
    valid_ds = phase19.TokenBlockDataset(
        DATA_DIR / "valid.bin",
        context_length=settings.context_length,
        vocab_size=model_config.vocab_size,
        stride=phase19.WINDOW_STRIDE,
    )

    micro = settings.micro_batch_size
    if settings.effective_batch_size != int(cfg.get("effective_batch_size", 32)):
        raise RuntimeError("effective batch size does not match the Phase 20 configuration")
    train_loader = phase19.build_dataloader(
        train_ds,
        batch_size=micro,
        shuffle=True,
        seed=settings.seed,
        pin_memory=True,
        drop_last=False,
    )

    eval_cfg = cfg.get("evaluation", {})
    val_count = min(int(eval_cfg.get("val_blocks", 256)), len(valid_ds))
    val_indices = fixed_validation_indices(len(valid_ds), val_count, seed=settings.seed)
    val_loader = build_validation_loader(
        valid_ds,
        val_indices,
        batch_size=min(int(eval_cfg.get("val_batch_size", 4)), micro),
    )

    loader_batches = len(train_loader)
    expected_updates = math.ceil(loader_batches / settings.gradient_accumulation_steps)
    if not smoke and not resume and settings.complete_pass:
        if settings.additional_steps != expected_updates:
            raise RuntimeError(
                f"configured updates {settings.additional_steps} do not match one-pass requirement "
                f"{expected_updates}; refusing to train"
            )
        if not settings.partial_final_update and loader_batches % settings.gradient_accumulation_steps:
            raise RuntimeError("one-pass loader has a remainder but partial_final_update is disabled")

    phase19.set_seeds(settings.seed)
    model = phase19.GPT(model_config).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != model_config.estimate_parameters():
        raise RuntimeError("model parameter count differs from analytical estimate")

    checkpoint_dir = SMOKE_CHECKPOINT_DIR if smoke else PHASE20_CHECKPOINT_DIR
    log_dir = SMOKE_LOG_DIR if smoke else PHASE20_LOG_DIR
    if not smoke and not resume:
        free_gb = shutil.disk_usage(ROOT).free / (1024 ** 3)
        # Full checkpoints include model + AdamW state; 12 GiB leaves headroom
        # for periodic snapshots, logs, and temporary atomic-save files.
        if free_gb < 12.0:
            raise RuntimeError(
                f"only {free_gb:.2f} GiB free on the project drive; "
                "at least 12 GiB is required before starting Phase 20"
            )
        print(f"Project-drive free space: {free_gb:.2f} GiB")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model,
        train_loader,
        settings,
        device=device,
        val_loader=val_loader,
        train_tokens=train_ds.token_count,
        checkpoint_dir=checkpoint_dir,
        model_config=model_config,
        tokenizer_sha256=tokenizer_sha,
        data_manifest_sha256=manifest_sha,
    )

    trainer.restore(checkpoint, reset_loader_state=not resume)
    if not compare_model_weights(trainer.model, checkpoint):
        raise RuntimeError("restored model weights do not match the source checkpoint")
    if len(trainer.optimizer.state) != len(checkpoint["optimizer_state"]["state"]):
        raise RuntimeError("restored optimizer state-entry count does not match checkpoint")
    if not weights_are_tied(trainer.model):
        raise RuntimeError("restored model embeddings are not tied")

    if not resume:
        # Phase 19's initial_tokens_seen is zero. Start Phase 20 stage accounting
        # at the inherited cumulative token count, without resetting that count.
        trainer.initial_tokens_seen = trainer.tokens_seen
        if trainer.step_index != 200 or trainer.stage_step != 0:
            raise RuntimeError("unexpected starting counters after Phase 19 restoration")
        trainer.save(checkpoint_dir / "phase20_start.pt")
        trainer.save(checkpoint_dir / "latest.pt")
    else:
        if trainer.stage_step >= settings.additional_steps:
            print("Phase 20 checkpoint already reached its configured final stage step.")
            return 0

    print("\n========== PHASE 20 RESTORATION VERIFIED ==========")
    print(f"source checkpoint: {source_checkpoint}")
    print(f"device: {device} ({torch.cuda.get_device_name(device)})")
    print(f"global starting step: {trainer.step_index}")
    print(f"stage starting step: {trainer.stage_step}")
    print(f"restored optimizer entries: {len(trainer.optimizer.state)}")
    print(f"fresh schedule: {settings.fresh_schedule}")
    print(f"schedule total steps: {trainer.scheduler.total_steps}")
    print(f"next learning rate: {trainer.scheduler.get_last_lr()[0]:.10g}")
    print(f"training loader batches: {loader_batches:,}")
    print(f"expected updates for one pass: {expected_updates:,}")
    print(f"complete pass mode: {settings.complete_pass}")
    print(f"partial final update: {settings.partial_final_update}")
    print("test partition opened: no")

    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    result = trainer.run()
    elapsed = time.perf_counter() - started

    if not smoke and settings.complete_pass:
        if result["stage_step"] != expected_updates:
            raise RuntimeError(
                f"completed {result['stage_step']} updates, expected {expected_updates} for one pass"
            )
        expected_sequences = len(train_ds)
        if result["sequences"] - result["initial_tokens_seen"] // settings.context_length != expected_sequences:
            # The trainer's `sequences` counter is cumulative while initial token
            # accounting is cumulative too; use stage token count as the decisive check.
            if result["stage_tokens_seen"] != expected_sequences * settings.context_length:
                raise RuntimeError("Phase 20 consumed-token count does not match one complete pass")
        if result["stage_tokens_seen"] != expected_sequences * settings.context_length:
            raise RuntimeError(
                f"Phase 20 consumed {result['stage_tokens_seen']:,} stage tokens; "
                f"expected {expected_sequences * settings.context_length:,}"
            )

    trainer.save(checkpoint_dir / "latest.pt")
    write_metrics_csv(log_dir / "metrics.csv", trainer.history)
    with (log_dir / "validation_history.json").open("w", encoding="utf-8") as handle:
        json.dump(trainer.validation_history, handle, indent=2)

    saved = load_checkpoint(checkpoint_dir / "latest.pt", map_location="cpu")
    restored = phase19.rebuild_model(saved, device="cpu")
    current_state = trainer.model.state_dict()
    saved_state = restored.state_dict()
    weights_match = all(
        torch.equal(value.detach().cpu(), saved_state[key].detach().cpu())
        for key, value in current_state.items()
    )
    weights_tied = weights_are_tied(restored)
    if not weights_match or not weights_tied:
        raise RuntimeError("Phase 20 checkpoint reload integrity check failed")

    peak_vram_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    summary = {
        "phase": 20,
        "scenario": "B",
        "mode": "smoke_test" if smoke else ("resume" if resume else "full_pass"),
        "run_started_utc": started_utc,
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "model": model_config.to_dict(),
        "parameter_count": parameter_count,
        "tokenizer_sha256": tokenizer_sha,
        "scenario_b_manifest_sha256": manifest_sha,
        "source_checkpoint": str(source_checkpoint),
        "starting_global_step": settings.initial_global_step,
        "training_settings": settings.to_dict(),
        "train_blocks": len(train_ds),
        "train_loader_batches": loader_batches,
        "expected_updates_for_one_pass": expected_updates,
        "train_result": result,
        "elapsed_seconds": elapsed,
        "peak_allocated_vram_mb": peak_vram_mb,
        "checkpoint_reload": {"weights_match": weights_match, "weights_tied": weights_tied,
                              "step": saved.get("step"), "tokens_seen": saved.get("tokens_seen")},
        "test_partition_opened": False,
    }
    with (log_dir / ("phase20_smoke_summary.json" if smoke else "phase20_summary.json")).open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)

    print("\n========== PHASE 20 SUMMARY ==========")
    print(json.dumps({
        "mode": summary["mode"],
        "optimizer_steps": result["stage_step"],
        "global_step": result["steps"],
        "stage_tokens_seen": result["stage_tokens_seen"],
        "best_validation_loss": result["best_val_loss"],
        "peak_allocated_vram_mb": peak_vram_mb,
        "checkpoint_reload": summary["checkpoint_reload"],
        "test_partition_opened": False,
    }, indent=2))
    print("PHASE 20 SMOKE TEST COMPLETE" if smoke else "PHASE 20 TRAINING COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
