"""PolyCodeGPT Phase 5C: complete the first base-model training pass.

Resumes the Phase 5B pilot checkpoint and consumes the training split exactly once:
8,129 full updates of 32 sequences plus one final partial update over the 14 blocks
left at the end of the epoch. Writes only to its own checkpoint and log
directories; the pilot artifacts are never modified.

``test.bin`` is never opened: integrity is checked by hashing ``train.bin`` and
``valid.bin`` and by stat-ing the test stream.

Usage:
    python scripts/train_first_pass.py --preflight-only
    python scripts/train_first_pass.py
    python scripts/train_first_pass.py --resume checkpoints/polycodegpt_first_pass/latest.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np
import torch

from train import (  # noqa: E402  (path set above)
    DATA_DIR,
    MANIFEST_PATH,
    MODEL_CONFIG_PATH,
    TOKENIZER_PATH,
    TRAINING_CONFIG_PATH,
    choose_batch_config,
    load_yaml,
    parameter_digest,
    set_seeds,
    verify_data_integrity,
)

from polycodegpt.data import build_dataloader, load_tokenizer, open_split, sha256_file  # noqa: E402
from polycodegpt.data.integrity import load_manifest  # noqa: E402
from polycodegpt.model import GPT, ModelConfig  # noqa: E402
from polycodegpt.training import (  # noqa: E402
    Trainer,
    TrainSettings,
    build_validation_loader,
    evaluate_loss,
    fixed_validation_indices,
    load_checkpoint,
    rebuild_model,
    require_cuda,
    resolve_precision,
    weights_are_tied,
    write_metrics_csv,
)

PILOT_DIR = PROJECT_ROOT / "checkpoints" / "polycodegpt_pilot"
FIRST_PASS_DIR = PROJECT_ROOT / "checkpoints" / "polycodegpt_first_pass"
LOG_DIR = PROJECT_ROOT / "logs" / "training"

# Phase 5B record, so the report can prove the pilot was not rewritten.
PILOT_SHA256 = {
    "latest.pt": "5ccce4ca69efa98252f4581c290204ec2da7bd6a0c0871a25e4230fc0e03d008",
    "best.pt": "ef16a53a2cd0ed29a0b4cc6e30ec0c9c3927a602e33185c186f866cc48535931",
    "step_000050.pt": "7942a44ee5caf72c7a859ace4ec7377cd99a7397041839cdcbc9c8701b70ce65",
    "step_000100.pt": "2d3ab842210aff1a21e65d1f3177198cb392aa4485913a7f0ce8a4e7a2e6918a",
    "step_000150.pt": "0ebc9cbb10df6f350fe1e2c34feadacdc063eb94e01ae89d3487a9f543f35450",
    "step_000200.pt": "b7c8898811d503e130fda8bddaca325236d9c7920ea2cf6e7b66e7fa12577b17",
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PolyCodeGPT first full training pass")
    parser.add_argument("--resume", default=str(PILOT_DIR / "latest.pt"), help="checkpoint to resume from")
    parser.add_argument("--checkpoint-dir", default=str(FIRST_PASS_DIR))
    parser.add_argument("--log-prefix", default="first_pass")
    parser.add_argument("--max-steps", type=int, default=8130, help="final optimizer step of the pass")
    parser.add_argument("--total-steps", type=int, default=8130, help="schedule length")
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--val-blocks", type=int, default=None)
    parser.add_argument("--val-batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--no-complete-pass",
        action="store_true",
        help="stop at max_steps without consuming the epoch's remaining batches",
    )
    parser.add_argument(
        "--no-partial-final-update",
        action="store_true",
        help="reject the final short accumulation group instead of normalising it",
    )
    parser.add_argument("--preflight-only", action="store_true", help="audit and exit without training")
    return parser.parse_args(argv)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pilot_state(paths: list[Path]) -> dict[str, dict[str, int]]:
    """Size and mtime of the pilot checkpoints, captured before and after the run."""
    return {p.name: {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in paths}


def describe_checkpoint(payload: dict, label: str) -> None:
    sched = payload["scheduler_state"]
    settings = payload.get("settings", {})
    print(f"\ncheckpoint:        {label}")
    print(f"  global step:     {payload['step']}")
    print(f"  tokens seen:     {payload['tokens_seen']:,}")
    print(f"  micro-batches:   {payload.get('micro_batches', 0):,}")
    print(f"  sequences:       {payload.get('sequences', 0):,}")
    print(f"  passes:          {payload.get('passes', 0)}")
    print(f"  best val loss:   {payload.get('best_val_loss')} (step {payload.get('best_step')})")
    print(
        f"  scheduler:       last_step={sched['last_step']} total_steps={sched['total_steps']} "
        f"warmup={sched['warmup_steps']} min_lr_ratio={sched['min_lr_ratio']}"
    )
    print(f"  optimizer:       {len(payload['optimizer_state']['state'])} state tensors, "
    f"{len(payload['optimizer_state']['param_groups'])} param groups")
    for group in payload["optimizer_state"]["param_groups"]:
        print(
            f"    group lr={group['lr']:.6e} weight_decay={group['weight_decay']} "
            f"tensors={len(group['params'])}"
        )
    print(f"  sampler/epoch:   loader_batches_consumed={payload.get('loader_batches_consumed', 0):,}")
    print(f"  precision:       {payload.get('precision')} | scaler state keys: {list(payload['scaler_state'])}")
    print(f"  rng state:       {sorted(payload['rng_state'])}")
    print(f"  weights tied:    {payload.get('weights_tied')}")
    print(f"  metrics rows:    {len(payload.get('metrics', []))}")
    print(f"  validation rows: {len(payload.get('validation', []))}")
    print(f"  saved settings:  micro={settings.get('micro_batch_size')} "
    f"accum={settings.get('gradient_accumulation_steps')} lr={settings.get('learning_rate')}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started_wall = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    run_started = time.perf_counter()

    resume_path = Path(args.resume).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    if checkpoint_dir == PILOT_DIR.resolve():
        raise SystemExit("refusing to write into the Phase 5B pilot checkpoint directory")
    if not resume_path.is_file():
        raise SystemExit(f"resume checkpoint not found: {resume_path}")

    device = require_cuda(args.device)
    model_config = ModelConfig.from_yaml(MODEL_CONFIG_PATH)
    training_cfg = load_yaml(TRAINING_CONFIG_PATH).get("training", {})
    manifest = load_manifest(MANIFEST_PATH)
    evaluation_cfg = training_cfg.get("evaluation", {})
    val_blocks = int(args.val_blocks or evaluation_cfg.get("val_blocks", 64))
    val_batch_size = int(args.val_batch_size or evaluation_cfg.get("val_batch_size", 8))

    tokenizer = load_tokenizer(TOKENIZER_PATH, vocab_size=model_config.vocab_size)
    model_config.validate_tokenizer_vocab(tokenizer.get_vocab_size())
    tokenizer_sha = sha256_file(TOKENIZER_PATH)
    manifest_sha = sha256_file(MANIFEST_PATH)

    print("=" * 78)
    print("PolyCodeGPT Phase 5C - first full training pass")
    print("=" * 78)
    print(f"device:           {torch.cuda.get_device_name(device)}")
    print(f"torch:            {torch.__version__}  cuda {torch.version.cuda}")
    print(f"tokenizer:        vocab {tokenizer.get_vocab_size()} sha {tokenizer_sha[:16]}...")
    print(f"data manifest:    v{manifest['manifest_version']} sha {manifest_sha[:16]}...")
    print(f"test.bin:         not opened")

    # --------------------------------------------------------------- settings
    settings = TrainSettings(
        micro_batch_size=int(args.micro_batch_size or training_cfg.get("micro_batch_size", 4)),
        gradient_accumulation_steps=int(
            args.gradient_accumulation_steps or training_cfg.get("gradient_accumulation_steps", 8)
        ),
        max_steps=int(args.max_steps),
        total_steps=int(args.total_steps),
        warmup_fraction=float(training_cfg.get("schedule", {}).get("warmup_fraction", 0.02)),
        min_lr_ratio=float(training_cfg.get("schedule", {}).get("min_lr_ratio", 0.1)),
        learning_rate=float(training_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 0.1)),
        gradient_clip=float(training_cfg.get("gradient_clip", 1.0)),
        precision=str(training_cfg.get("precision", "auto")),
        seed=int(args.seed if args.seed is not None else training_cfg.get("seed", 42)),
        context_length=int(manifest["context_length"]),
        log_every=int(args.log_every),
        eval_steps=(),
        checkpoint_steps=(),
        eval_every=int(args.eval_every),
        checkpoint_every=int(args.checkpoint_every),
        initial_validation=False,
        complete_pass=not args.no_complete_pass,
        partial_final_update=not args.no_partial_final_update,
    )
    if settings.max_steps > settings.total_steps:
        raise SystemExit("max_steps cannot exceed total_steps")

    # ------------------------------------------------------- preflight: data
    integrity_before = verify_data_integrity(manifest)
    train_ds = open_split(DATA_DIR, "train", manifest)
    valid_ds = open_split(DATA_DIR, "valid", manifest)
    train_loader = build_dataloader(
        train_ds,
        batch_size=settings.micro_batch_size,
        shuffle=True,
        seed=settings.seed,
    )
    epoch_batches = len(train_loader)
    tail_batch = len(train_ds) % settings.micro_batch_size or settings.micro_batch_size

    print(f"train:            {len(train_ds):,} blocks / {train_ds.token_count:,} tokens")
    print(f"valid:            {len(valid_ds):,} blocks / {valid_ds.token_count:,} tokens")

    # -------------------------------------------------- preflight: checkpoint
    payload = load_checkpoint(resume_path, map_location="cpu")
    describe_checkpoint(payload, f"{resume_path}  sha256 {sha256_of(resume_path)[:16]}...")
    if payload.get("tokenizer_sha256") != tokenizer_sha:
        raise SystemExit("checkpoint tokenizer hash does not match the tokenizer on disk")
    if payload.get("data_manifest_sha256") != manifest_sha:
        raise SystemExit("checkpoint manifest hash does not match the manifest on disk")
    if int(payload["step"]) <= 0:
        raise SystemExit("refusing to continue: the checkpoint has no completed optimizer steps")
    if payload["model_config"] != model_config.to_dict():
        raise SystemExit("checkpoint model config does not match configs/model.yaml")

    # -------------------------------------------------- preflight: coverage
    consumed_batches = int(payload.get("loader_batches_consumed", 0))
    remaining_batches = epoch_batches - consumed_batches
    full_groups = remaining_batches // settings.gradient_accumulation_steps
    leftover_batches = remaining_batches % settings.gradient_accumulation_steps
    leftover_blocks = (
        (leftover_batches - 1) * settings.micro_batch_size + tail_batch if leftover_batches else 0
    )
    expected_final_step = int(payload["step"]) + full_groups + (1 if leftover_blocks else 0)
    blocks_total = len(train_ds)
    tokens_total = blocks_total * settings.context_length

    print("\n--- coverage preflight ---")
    print(f"loader:           {epoch_batches:,} micro-batches of {settings.micro_batch_size} "
    f"(last batch {tail_batch} blocks), shuffle={type(train_loader.sampler).__name__}")
    print(f"already consumed: {consumed_batches:,} micro-batches ({consumed_batches * settings.micro_batch_size:,} blocks)")
    print(f"remaining:        {remaining_batches:,} micro-batches")
    print(f"  full updates:   {full_groups:,} x {settings.effective_batch_size} sequences")
    print(f"  final group:    {leftover_batches} micro-batches / {leftover_blocks} blocks "
    f"({leftover_blocks * settings.context_length:,} tokens)"
    + ("" if leftover_blocks else "  (none)"))
    print(f"expected end:     step {expected_final_step} (max_steps {settings.max_steps})")
    print(f"pass totals:      {blocks_total:,} blocks / {tokens_total:,} tokens")
    if settings.complete_pass and expected_final_step != settings.max_steps:
        raise SystemExit(
            f"complete-pass run would end at step {expected_final_step}, not {settings.max_steps}"
        )
    if settings.complete_pass and leftover_blocks and args.no_partial_final_update:
        raise SystemExit("leftover blocks need a partial final update but it was disabled")

    # ---------------------------------------------------------------- memory
    plan = resolve_precision(settings.precision, device)
    print(f"\nprecision:        {plan.precision} (requested {settings.precision})")
    print("--- full-context forward+backward memory probe (real tokens) ---")
    probe_micro, probe_accum, attempts = choose_batch_config(
        model_config,
        plan,
        device,
        settings.micro_batch_size,
        settings.effective_batch_size,
        train_ds,
        settings.seed,
    )
    if probe_micro != settings.micro_batch_size:
        print(
            f"  fallback applied: micro_batch {settings.micro_batch_size} -> {probe_micro}, "
            f"accum {settings.gradient_accumulation_steps} -> {probe_accum}"
        )
        raise SystemExit("fallback would change the coverage arithmetic; inspect and rerun explicitly")

    pilot_paths = sorted(PILOT_DIR.glob("*.pt"))
    pilot_before = pilot_state(pilot_paths)

    preflight = {
        "started_utc": started_wall,
        "resume_checkpoint": str(resume_path),
        "resume_step": int(payload["step"]),
        "resume_tokens_seen": int(payload["tokens_seen"]),
        "resume_scheduler": payload["scheduler_state"],
        "resume_precision": payload.get("precision"),
        "settings": settings.to_dict(),
        "train_blocks": blocks_total,
        "train_tokens": train_ds.token_count,
        "epoch_batches": epoch_batches,
        "consumed_batches": consumed_batches,
        "remaining_batches": remaining_batches,
        "full_groups": full_groups,
        "leftover_batches": leftover_batches,
        "leftover_blocks": leftover_blocks,
        "expected_final_step": expected_final_step,
        "shuffle": type(train_loader.sampler).__name__,
        "memory_probe": [a.to_dict() for a in attempts],
        "pilot_checkpoints_before": pilot_before,
        "integrity": integrity_before,
    }

    if args.preflight_only:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_DIR / f"{args.log_prefix}_preflight.json", "w", encoding="utf-8") as fh:
            json.dump(preflight, fh, indent=2)
        print(f"\npreflight written to {LOG_DIR / (args.log_prefix + '_preflight.json')}")
        print("PREFLIGHT COMPLETE")
        return 0

    # ------------------------------------------------------------- model
    set_seeds(settings.seed)
    model = GPT(model_config).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    val_indices = fixed_validation_indices(len(valid_ds), val_blocks, seed=settings.seed)
    val_loader = build_validation_loader(valid_ds, val_indices, batch_size=val_batch_size)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nmodel:            {parameter_count:,} parameters")
    print(f"validation:       fixed subset of {len(val_indices)} blocks "
    f"({len(val_indices) * settings.context_length:,} tokens), seed {settings.seed}")
    print(f"schedule:         steps {settings.max_steps} of {settings.total_steps} "
    f"(warmup {settings.warmup_steps}, cosine to {settings.min_lr_ratio:.0%} of base)")
    print(f"batch:            micro {settings.micro_batch_size} x accum "
    f"{settings.gradient_accumulation_steps} = {settings.effective_batch_size} sequences "
    f"= {settings.tokens_per_step:,} tokens/update")
    print(f"checkpoint dir:   {checkpoint_dir}")
    print(f"log prefix:       {LOG_DIR / args.log_prefix}_*")

    trainer = Trainer(
        model,
        train_loader,
        settings,
        device=device,
        val_loader=val_loader,
        train_tokens=manifest["splits"]["train"]["tokens"],
        checkpoint_dir=checkpoint_dir,
        model_config=model_config,
        tokenizer_sha256=tokenizer_sha,
        data_manifest_sha256=manifest_sha,
    )
    trainer.restore(payload)
    start_step = trainer.step_index
    start_tokens = trainer.tokens_seen
    start_sequences = trainer.sequences
    start_micro = trainer.micro_batches

    schedule_adjustment = None
    if trainer.scheduler.total_steps != settings.total_steps:
        lr_before = trainer.scheduler.get_last_lr()[0]
        trainer.scheduler.set_total_steps(settings.total_steps, warmup_fraction=settings.warmup_fraction)
        lr_after = trainer.scheduler.get_last_lr()[0]
        schedule_adjustment = {
            "total_steps_before": int(payload["scheduler_state"]["total_steps"]),
            "total_steps_after": settings.total_steps,
            "last_step": trainer.scheduler.last_step,
            "warmup_steps": trainer.scheduler.warmup_steps,
            "lr_at_resume_before": lr_before,
            "lr_at_resume_after": lr_after,
        }
        print(f"\nschedule extended: total_steps {payload['scheduler_state']['total_steps']} -> "
              f"{settings.total_steps} at step {trainer.scheduler.last_step} "
              f"(lr {lr_before:.6e} -> {lr_after:.6e})")
    print(f"resumed at step {start_step} with {trainer.scheduler.get_last_lr()[0]:.6e} learning rate\n")

    try:
        result = trainer.run()
    except KeyboardInterrupt:
        trainer.save(checkpoint_dir / "latest.pt")
        print("\ninterrupted: latest.pt saved")
        raise

    # -------------------------------------------------- final full validation
    print("\n--- final full validation over valid.bin (all blocks) ---")
    full_loader = build_validation_loader(valid_ds, None, batch_size=val_batch_size)
    val_started = time.perf_counter()
    full_val = evaluate_loss(
        model,
        full_loader,
        device,
        autocast_dtype=plan.autocast_dtype,
        autocast_device_type=plan.device_type,
    )
    full_val_seconds = time.perf_counter() - val_started
    print(
        f"full validation: loss {full_val['loss']:.6f} ppl {full_val['perplexity']:.4f} "
        f"over {full_val['tokens']:,} tokens in {full_val_seconds:.1f}s"
    )
    trainer.validation_history.append(
        {
            "step": trainer.step_index,
            "loss": full_val["loss"],
            "perplexity": full_val["perplexity"],
            "tokens": full_val["tokens"],
            "sequences": full_val["sequences"],
            "batches": full_val["batches"],
            "full_split": True,
        }
    )
    trainer.save(checkpoint_dir / "latest.pt")
    if full_val["loss"] < trainer.best_val_loss:
        trainer.best_val_loss = float(full_val["loss"])
        trainer.best_step = trainer.step_index
        trainer.save(checkpoint_dir / "best.pt")

    # ------------------------------------------------------------ finalisation
    final_digest = parameter_digest(model)
    finite_parameters = all(torch.isfinite(p).all().item() for p in model.parameters())
    write_metrics_csv(LOG_DIR / f"{args.log_prefix}_metrics.csv", trainer.history)
    with open(LOG_DIR / f"{args.log_prefix}_validation.json", "w", encoding="utf-8") as fh:
        json.dump(trainer.validation_history, fh, indent=2)

    latest_payload = load_checkpoint(checkpoint_dir / "latest.pt", map_location="cpu")
    restored = rebuild_model(latest_payload, device="cpu")
    weights_match = all(
        torch.equal(value.cpu(), restored.state_dict()[key].cpu())
        for key, value in latest_payload["model_state"].items()
    )
    tying_after_reload = weights_are_tied(restored)

    integrity_after = verify_data_integrity(manifest)
    test_untouched = integrity_before["test_bin"]["mtime_ns"] == integrity_after["test_bin"]["mtime_ns"]

    pilot_after = pilot_state(sorted(PILOT_DIR.glob("*.pt")))
    pilot_unchanged = pilot_before == pilot_after
    pilot_hashes = {p.name: sha256_of(p) for p in sorted(PILOT_DIR.glob("*.pt"))}
    pilot_hash_match = all(pilot_hashes.get(name) == digest for name, digest in PILOT_SHA256.items())

    checkpoint_hashes = {p.name: sha256_of(p) for p in sorted(checkpoint_dir.glob("*.pt"))}
    checkpoint_sizes = {p.name: p.stat().st_size for p in sorted(checkpoint_dir.glob("*.pt"))}

    grad_norms = [row["gradient_norm"] for row in trainer.history]
    new_rows = trainer.history[-(trainer.step_index - start_step):]
    new_grad_norms = [row["gradient_norm"] for row in new_rows]
    covered_blocks = trainer.sequences
    covered_tokens = trainer.tokens_seen

    checks = {
        "final_step_is_max_steps": trainer.step_index == settings.max_steps,
        "scheduler_advanced_to_final_step": trainer.scheduler.last_step == settings.max_steps,
        "scheduler_total_steps": trainer.scheduler.total_steps == settings.total_steps,
        "every_block_consumed_once": covered_blocks == blocks_total,
        "tokens_match_blocks": covered_tokens == tokens_total,
        "passes_completed": trainer.passes == 1,
        "all_parameters_finite": finite_parameters,
        "weights_tied_after_reload": tying_after_reload,
        "weights_match_checkpoint": weights_match,
        "test_bin_untouched": test_untouched,
        "pilot_checkpoints_unchanged": pilot_unchanged,
        "pilot_checkpoints_match_phase5b_hashes": pilot_hash_match,
        "gradient_norms_finite": all(np.isfinite(new_grad_norms)),
    }
    failed = [name for name, ok in checks.items() if not ok]

    summary = {
        "run_started_utc": started_wall,
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "precision": plan.precision,
        "parameter_count": parameter_count,
        "seed": settings.seed,
        "settings": settings.to_dict(),
        "schedule_adjustment": schedule_adjustment,
        "resume": {
            "path": str(resume_path),
            "sha256": sha256_of(resume_path),
            "step": start_step,
            "tokens_seen": start_tokens,
            "sequences": start_sequences,
            "micro_batches": start_micro,
        },
        "train_result": result,
        "epoch_batches": epoch_batches,
        "remaining_batches_at_resume": remaining_batches,
        "full_updates_this_run": full_groups,
        "partial_update": {"micro_batches": leftover_batches, "blocks": leftover_blocks,
                           "tokens": leftover_blocks * settings.context_length},
        "coverage": {
            "blocks": covered_blocks,
            "blocks_total": blocks_total,
            "tokens": covered_tokens,
            "tokens_total": tokens_total,
            "passes": trainer.passes,
            "omitted_blocks": blocks_total - covered_blocks,
        },
        "training_only": {
            "steps": trainer.step_index - start_step,
            "blocks": covered_blocks - start_sequences,
            "tokens": covered_tokens - start_tokens,
            "elapsed_seconds": result["elapsed_seconds"],
            "tokens_per_second": (covered_tokens - start_tokens) / result["elapsed_seconds"],
        },
        "new_metrics_rows": len(new_rows),
        "gradient_norms_this_run": {
            "min": float(np.min(new_grad_norms)),
            "max": float(np.max(new_grad_norms)),
            "mean": float(np.mean(new_grad_norms)),
            "finite": int(np.isfinite(new_grad_norms).sum()),
            "count": len(new_grad_norms),
        },
        "gradient_norms_all": {
            "min": float(np.min(grad_norms)),
            "max": float(np.max(grad_norms)),
            "mean": float(np.mean(grad_norms)),
        },
        "training_loss": {
            "first_step_this_run": new_rows[0]["training_loss"],
            "final_step": new_rows[-1]["training_loss"],
            "final_ten_mean": float(np.mean([r["training_loss"] for r in trainer.history[-10:]])),
        },
        "validation_history": trainer.validation_history,
        "final_full_validation": full_val,
        "full_validation_seconds": full_val_seconds,
        "best_val_loss": trainer.best_val_loss,
        "best_step": trainer.best_step,
        "final_weight_digest": final_digest,
        "peak_allocated_vram_mb": trainer.summary()["peak_allocated_vram_mb"],
        "checks": checks,
        "failed_checks": failed,
        "checkpoint_hashes": checkpoint_hashes,
        "checkpoint_sizes": checkpoint_sizes,
        "pilot_checkpoint_hashes_after": pilot_hashes,
        "integrity": {"before": integrity_before, "after": integrity_after},
        "wall_clock_seconds": time.perf_counter() - run_started,
    }
    with open(LOG_DIR / f"{args.log_prefix}_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    subset_rows = [r for r in trainer.validation_history if not r.get("full_split")]
    print("\n--- first pass summary ---")
    print(f"resumed at:          step {start_step}")
    print(f"final optimizer step:{trainer.step_index}")
    print(f"blocks consumed:     {covered_blocks:,} / {blocks_total:,} (omitted {blocks_total - covered_blocks})")
    print(f"tokens consumed:     {covered_tokens:,} / {tokens_total:,} "
          f"({100.0 * covered_tokens / tokens_total:.4f}% of the split)")
    print(f"passes completed:    {trainer.passes}")
    print(f"training loss:       {new_rows[0]['training_loss']:.6f} -> {new_rows[-1]['training_loss']:.6f}")
    print(f"grad norms (run):    min {np.min(new_grad_norms):.3f} max {np.max(new_grad_norms):.3f} "
          f"mean {np.mean(new_grad_norms):.3f}")
    for record in subset_rows:
        if record["step"] % 500 == 0 or record["step"] in (start_step, settings.max_steps):
            print(f"  valid @ step {record['step']:5d}: {record['loss']:.6f} (ppl {record['perplexity']:.2f})")
    print(f"final full valid:    {full_val['loss']:.6f} (ppl {full_val['perplexity']:.3f})")
    print(f"peak VRAM:           {trainer.summary()['peak_allocated_vram_mb']:.1f} MiB")
    print(f"training elapsed:    {result['elapsed_seconds']:.1f}s "
          f"({(covered_tokens - start_tokens) / result['elapsed_seconds']:,.0f} tokens/s)")
    print(f"wall clock:          {summary['wall_clock_seconds']:.1f}s")
    print(f"best val loss:       {trainer.best_val_loss:.6f} (step {trainer.best_step})")
    print(f"checks failed:       {failed if failed else 'none'}")
    print("test.bin untouched:  " + str(test_untouched))
    print("pilot intact:        " + str(pilot_unchanged and pilot_hash_match))
    print(f"checkpoints:         {checkpoint_dir}")
    print("FIRST PASS COMPLETE" if not failed else "FIRST PASS COMPLETE WITH FAILED CHECKS")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
