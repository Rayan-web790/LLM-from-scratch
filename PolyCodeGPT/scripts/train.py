"""PolyCodeGPT training entry point.

The pilot mode trains the randomly initialised Transformer on the finalised
``train.bin`` stream for a fixed number of *optimizer* steps, validating on
``valid.bin`` only. ``test.bin`` is never opened.

Usage:
    python scripts/train.py --mode pilot --max-steps 200
    python scripts/train.py --mode pilot --max-steps 200 --resume checkpoints/polycodegpt_pilot/latest.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch
import yaml

from polycodegpt.data import build_dataloader, load_tokenizer, open_split, sha256_file
from polycodegpt.data.integrity import load_manifest
from polycodegpt.model import GPT, ModelConfig
from polycodegpt.training import (
    Trainer,
    TrainSettings,
    build_validation_loader,
    evaluate_loss,
    fixed_validation_indices,
    load_checkpoint,
    rebuild_model,
    require_cuda,
    resolve_precision,
    write_metrics_csv,
)
from polycodegpt.training.evaluate import autocast_context

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "tokenized_v1_1"
MANIFEST_PATH = DATA_DIR / "manifest.json"
MODEL_CONFIG_PATH = PROJECT_ROOT / "configs" / "model.yaml"
TRAINING_CONFIG_PATH = PROJECT_ROOT / "configs" / "training.yaml"
TOKENIZER_PATH = PROJECT_ROOT / "tokenizer" / "artifacts" / "polycodegpt_tokenizer_v1_1.json"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "polycodegpt_pilot"
LOG_DIR = PROJECT_ROOT / "logs" / "training"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parameter_digest(model: torch.nn.Module) -> str:
    """Order-stable SHA-256 over every parameter tensor."""
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().to("cpu", copy=True).contiguous().numpy().tobytes())
    return digest.hexdigest()


def build_settings(training_cfg: dict[str, Any], manifest: dict[str, Any], args) -> TrainSettings:
    train_cfg = training_cfg.get("training", {})
    schedule = train_cfg.get("schedule", {})
    evaluation = train_cfg.get("evaluation", {})
    pilot = train_cfg.get("pilot", {})

    manifest_context = int(manifest["context_length"])
    context_length = int(train_cfg.get("context_length", manifest_context))
    if context_length != manifest_context:
        raise SystemExit(
            f"training context_length {context_length} != data context_length {manifest_context}"
        )
    if int(manifest["window_stride"]) != context_length or int(manifest["window_length"]) != context_length + 1:
        raise SystemExit("data window geometry does not match a context_length stride")

    max_steps = int(args.max_steps or pilot.get("max_steps", 200))
    return TrainSettings(
        micro_batch_size=int(args.micro_batch_size or train_cfg.get("micro_batch_size", 4)),
        gradient_accumulation_steps=int(
            args.gradient_accumulation_steps or train_cfg.get("gradient_accumulation_steps", 8)
        ),
        max_steps=max_steps,
        total_steps=int(args.total_steps or schedule.get("total_steps", 8129)),
        warmup_fraction=float(schedule.get("warmup_fraction", 0.02)),
        min_lr_ratio=float(schedule.get("min_lr_ratio", 0.1)),
        learning_rate=float(train_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.1)),
        gradient_clip=float(train_cfg.get("gradient_clip", 1.0)),
        precision=str(train_cfg.get("precision", "auto")),
        seed=int(args.seed if args.seed is not None else train_cfg.get("seed", 42)),
        context_length=context_length,
        eval_steps=tuple(int(x) for x in evaluation.get("steps", [50, 100, 150, 200])),
        checkpoint_steps=tuple(int(x) for x in train_cfg.get("checkpoint_steps", [50, 100, 150, 200])),
    )


def verify_data_integrity(manifest: dict[str, Any]) -> dict[str, Any]:
    """Hash the two streams used by training; never read ``test.bin``."""
    report: dict[str, Any] = {"splits": {}}
    for split in ("train", "valid"):
        path = DATA_DIR / manifest["splits"][split]["file"]
        actual = sha256_file(path)
        expected = manifest["splits"][split]["sha256"]
        if actual != expected:
            raise SystemExit(f"{split}.bin SHA-256 {actual} != manifest {expected}")
        report["splits"][split] = {
            "file": path.name,
            "sha256": actual,
            "matches_manifest": True,
            "tokens": manifest["splits"][split]["tokens"],
            "blocks": manifest["splits"][split]["blocks"],
        }

    test_path = DATA_DIR / manifest["splits"]["test"]["file"]
    stat = test_path.stat()
    report["test_bin"] = {
        "file": test_path.name,
        "opened": False,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    return report


# --------------------------------------------------------------------------- #
# GPU memory probe
# --------------------------------------------------------------------------- #
@dataclass
class ProbeAttempt:
    micro_batch_size: int
    gradient_accumulation_steps: int
    effective_batch_size: int
    ok: bool
    allocated_mb: float = 0.0
    peak_mb: float = 0.0
    loss: float = 0.0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "micro_batch_size": self.micro_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "effective_batch_size": self.effective_batch_size,
            "ok": self.ok,
            "allocated_mb": self.allocated_mb,
            "peak_mb": self.peak_mb,
            "loss": self.loss,
            "error": self.error,
        }


def _reset_cuda_stats() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def probe_memory(
    model_config: ModelConfig,
    plan,
    device: torch.device,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    dataset,
    seed: int,
) -> ProbeAttempt:
    """Full-context forward + backward on real tokens; reports actual VRAM use."""
    stacked_inputs = torch.stack([dataset[i][0] for i in range(micro_batch_size)])
    stacked_targets = torch.stack([dataset[i][1] for i in range(micro_batch_size)])

    _reset_cuda_stats()
    set_seeds(seed)
    model = GPT(model_config).to(device)
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=plan.use_grad_scaler)

    attempt = ProbeAttempt(
        micro_batch_size, gradient_accumulation_steps, micro_batch_size * gradient_accumulation_steps, False
    )
    inputs = targets = None
    try:
        inputs = stacked_inputs.to(device)
        targets = stacked_targets.to(device)
        with autocast_context(plan.device_type, plan.autocast_dtype):
            _, loss = model(inputs, targets=targets)
        scaled = loss / gradient_accumulation_steps
        if scaler.is_enabled():
            scaler.scale(scaled).backward()
        else:
            scaled.backward()
        torch.cuda.synchronize()
        attempt.ok = True
        attempt.loss = float(loss.item())
        attempt.allocated_mb = round(torch.cuda.memory_allocated() / 1024**2, 2)
        attempt.peak_mb = round(torch.cuda.max_memory_allocated() / 1024**2, 2)
    except torch.cuda.OutOfMemoryError as exc:
        attempt.error = " ".join(str(exc).split())[:300]
    finally:
        model.zero_grad(set_to_none=True)
        del model, scaler, inputs, targets, stacked_inputs, stacked_targets
        _reset_cuda_stats()
    return attempt


def choose_batch_config(
    model_config: ModelConfig,
    plan,
    device: torch.device,
    micro_batch_size: int,
    effective_batch_size: int,
    dataset,
    seed: int,
) -> tuple[int, int, list[ProbeAttempt]]:
    """Probe the configured batch, halving it (and doubling accumulation) on OOM."""
    attempts: list[ProbeAttempt] = []
    micro = max(1, micro_batch_size)
    target_effective = max(1, effective_batch_size)
    while True:
        accum = max(1, target_effective // micro)
        attempt = probe_memory(model_config, plan, device, micro, accum, dataset, seed)
        attempts.append(attempt)
        print(
            f"  probe: micro_batch={micro} accum={accum} "
            f"-> {'OK' if attempt.ok else 'OOM'} "
            f"allocated={attempt.allocated_mb:.1f} MiB peak={attempt.peak_mb:.1f} MiB"
            + ("" if attempt.ok else f" ({attempt.error[:80]})")
        )
        if attempt.ok:
            return micro, accum, attempts
        if micro == 1:
            raise SystemExit("CUDA OOM even with micro_batch_size=1; pilot aborted")
        micro //= 2


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PolyCodeGPT training (pilot)")
    parser.add_argument("--mode", default="pilot", choices=["pilot"], help="run mode")
    parser.add_argument("--max-steps", type=int, default=None, help="optimizer steps to run")
    parser.add_argument("--total-steps", type=int, default=None, help="planned schedule length")
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--val-blocks", type=int, default=None, help="size of the fixed validation subset")
    parser.add_argument("--val-batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", default=None, help="checkpoint to resume from")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started_wall = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    device = require_cuda(args.device)
    model_config = ModelConfig.from_yaml(MODEL_CONFIG_PATH)
    training_cfg = load_yaml(TRAINING_CONFIG_PATH)
    manifest = load_manifest(MANIFEST_PATH)
    evaluation_cfg = training_cfg.get("training", {}).get("evaluation", {})
    if args.val_blocks is None:
        args.val_blocks = int(evaluation_cfg.get("val_blocks", 64))
    if args.val_batch_size is None:
        args.val_batch_size = int(evaluation_cfg.get("val_batch_size", 8))

    # ------------------------------------------------------------- tokenizer
    tokenizer = load_tokenizer(TOKENIZER_PATH, vocab_size=model_config.vocab_size)
    model_config.validate_tokenizer_vocab(tokenizer.get_vocab_size())
    tokenizer_sha = sha256_file(TOKENIZER_PATH)
    manifest_sha = sha256_file(MANIFEST_PATH)

    print("=" * 78)
    print("PolyCodeGPT pilot training")
    print("=" * 78)
    print(f"device:           {torch.cuda.get_device_name(device)}")
    print(f"torch:            {torch.__version__}  cuda {torch.version.cuda}")
    print(f"tokenizer:        v1.1.0 vocab {tokenizer.get_vocab_size()} sha {tokenizer_sha[:16]}...")
    print(f"data manifest:    v{manifest['manifest_version']} sha {manifest_sha[:16]}...")
    print(
        f"data geometry:    context {manifest['context_length']} "
        f"window {manifest['window_length']} stride {manifest['window_stride']} "
        f"overlap {manifest['window_overlap']}"
    )

    # --------------------------------------------------------------- data
    integrity_before = verify_data_integrity(manifest)
    train_ds = open_split(DATA_DIR, "train", manifest)
    valid_ds = open_split(DATA_DIR, "valid", manifest)
    print(f"train:            {len(train_ds):,} blocks / {train_ds.token_count:,} tokens")
    print(f"valid:            {len(valid_ds):,} blocks / {valid_ds.token_count:,} tokens")
    print("test.bin:         not opened")

    settings = build_settings(training_cfg, manifest, args)
    plan = resolve_precision(settings.precision, device)
    print(f"precision:        {plan.precision} (requested {settings.precision})")

    # ---------------------------------------------------------- memory probe
    print("\n--- full-context forward+backward memory probe (real tokens) ---")
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
        settings.micro_batch_size = probe_micro
        settings.gradient_accumulation_steps = probe_accum
    effective_batch = settings.effective_batch_size
    if effective_batch < 32:
        print(f"  note: effective batch is {effective_batch} sequences (< 32)")

    # ------------------------------------------------------------- model
    set_seeds(settings.seed)
    model = GPT(model_config).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    estimated = model_config.estimate_parameters()
    if parameter_count != estimated:
        raise SystemExit(f"parameter count {parameter_count} != analytical {estimated}")
    initial_digest = parameter_digest(model)
    initial_parameters = [p.detach().clone() for p in model.parameters()]
    print(f"\nmodel:            {parameter_count:,} parameters (analytical {estimated:,})")
    print(f"initial weights:  sha256 {initial_digest[:16]}... seed {settings.seed}")
    print(f"weight tying:     {model.lm_head.weight is model.token_embedding.weight}")

    # ------------------------------------------------------------- loaders
    train_loader = build_dataloader(
        train_ds,
        batch_size=settings.micro_batch_size,
        shuffle=True,
        seed=settings.seed,
    )
    val_indices = fixed_validation_indices(len(valid_ds), args.val_blocks, seed=settings.seed)
    val_loader = build_validation_loader(valid_ds, val_indices, batch_size=args.val_batch_size)
    print(
        f"validation:       fixed subset of {len(val_indices)} blocks "
        f"({len(val_indices) * settings.context_length:,} tokens), seed {settings.seed}"
    )
    print(
        f"schedule:         {settings.max_steps} pilot steps of {settings.total_steps} planned "
        f"(warmup {settings.warmup_steps} steps, cosine to {settings.min_lr_ratio:.0%} of base)"
    )
    print(
        f"batch:            micro {settings.micro_batch_size} x accum "
        f"{settings.gradient_accumulation_steps} = {effective_batch} sequences "
        f"= {settings.tokens_per_step:,} tokens/step"
    )
    print(f"log dir:          {LOG_DIR}")
    print(f"checkpoint dir:   {CHECKPOINT_DIR}\n")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model,
        train_loader,
        settings,
        device=device,
        val_loader=val_loader,
        train_tokens=manifest["splits"]["train"]["tokens"],
        checkpoint_dir=CHECKPOINT_DIR,
        model_config=model_config,
        tokenizer_sha256=tokenizer_sha,
        data_manifest_sha256=manifest_sha,
    )

    resumed_from = None
    if args.resume:
        payload = load_checkpoint(args.resume, map_location="cpu")
        trainer.restore(payload)
        resumed_from = {"path": str(args.resume), "step": trainer.step_index, "tokens_seen": trainer.tokens_seen}
        print(f"resumed from {args.resume} at step {trainer.step_index}\n")

    start = time.perf_counter()
    try:
        result = trainer.run()
    except KeyboardInterrupt:
        trainer.save(CHECKPOINT_DIR / "latest.pt")
        print("\ninterrupted: latest.pt saved")
        raise

    # -------------------------------------------------- final full validation
    print("\n--- final full validation over valid.bin (all blocks) ---")
    full_loader = build_validation_loader(valid_ds, None, batch_size=args.val_batch_size)
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
    trainer.save(CHECKPOINT_DIR / "latest.pt")
    if full_val["loss"] < trainer.best_val_loss:
        trainer.best_val_loss = float(full_val["loss"])
        trainer.best_step = trainer.step_index
        trainer.save(CHECKPOINT_DIR / "best.pt")

    # ------------------------------------------------------------ finalisation
    changed = sum(
        1 for before, after in zip(initial_parameters, model.parameters()) if not torch.equal(before, after)
    )
    final_digest = parameter_digest(model)
    finite_parameters = all(torch.isfinite(p).all().item() for p in model.parameters())

    write_metrics_csv(LOG_DIR / "pilot_metrics.csv", trainer.history)
    with open(LOG_DIR / "pilot_validation.json", "w", encoding="utf-8") as fh:
        json.dump(trainer.validation_history, fh, indent=2)

    payload = load_checkpoint(CHECKPOINT_DIR / "latest.pt", map_location="cpu")
    restored = rebuild_model(payload, device="cpu")
    weights_match = all(
        torch.equal(value.cpu(), restored.state_dict()[key].cpu()) for key, value in payload["model_state"].items()
    )
    tying_after_reload = restored.lm_head.weight is restored.token_embedding.weight

    integrity_after = verify_data_integrity(manifest)
    test_untouched = integrity_before["test_bin"]["mtime_ns"] == integrity_after["test_bin"]["mtime_ns"]

    grad_norms = [row["gradient_norm"] for row in trainer.history]
    initial_training = trainer.history[0]["training_loss"] if trainer.history else float("nan")
    final_training = trainer.history[-1]["training_loss"] if trainer.history else float("nan")
    first_ten = trainer.history[:10]
    initial_val = next((v for v in trainer.validation_history if v["step"] == 0), None)

    summary = {
        "run_started_utc": started_wall,
        "mode": args.mode,
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "model": model_config.to_dict(),
        "parameter_count": parameter_count,
        "seed": settings.seed,
        "initial_weight_digest": initial_digest,
        "final_weight_digest": final_digest,
        "parameters_changed": changed,
        "parameters_total": len(initial_parameters),
        "parameters_all_finite": finite_parameters,
        "precision": plan.precision,
        "settings": settings.to_dict(),
        "memory_probe": [a.to_dict() for a in attempts],
        "train_blocks": len(train_ds),
        "train_tokens": train_ds.token_count,
        "valid_blocks": len(valid_ds),
        "valid_subset_blocks": len(val_indices),
        "train_result": result,
        "initial_training_loss": initial_training,
        "initial_ten_step_mean_loss": float(np.mean([r["training_loss"] for r in first_ten])) if first_ten else None,
        "final_training_loss": final_training,
        "final_ten_step_mean_loss": float(np.mean([r["training_loss"] for r in trainer.history[-10:]])),
        "initial_validation_loss": initial_val["loss"] if initial_val else None,
        "validation_history": trainer.validation_history,
        "final_full_validation": full_val,
        "full_validation_seconds": full_val_seconds,
        "gradient_norms": {
            "min": float(min(grad_norms)),
            "max": float(max(grad_norms)),
            "mean": float(np.mean(grad_norms)),
            "finite_count": int(len(grad_norms)),
        },
        "best_val_loss": trainer.best_val_loss,
        "best_step": trainer.best_step,
        "resumed_from": resumed_from,
        "checkpoint_reload": {
            "weights_match": weights_match,
            "weights_tied": tying_after_reload,
            "step": payload["step"],
            "tokens_seen": payload["tokens_seen"],
        },
        "integrity": {"before": integrity_before, "after": integrity_after, "test_bin_untouched": test_untouched},
        "elapsed_seconds": result["elapsed_seconds"],
    }
    with open(LOG_DIR / "pilot_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("\n--- pilot summary ---")
    print(f"optimizer steps:     {trainer.step_index}")
    print(f"tokens processed:    {trainer.tokens_seen:,}")
    print(f"blocks processed:    {trainer.sequences:,}")
    print(f"passes over train:   {trainer.tokens_seen / train_ds.token_count:.4f}")
    print(f"initial train loss:  {initial_training:.6f} (step 1)")
    print(f"final train loss:    {final_training:.6f} (step {trainer.step_index})")
    if initial_val:
        print(f"initial valid loss:  {initial_val['loss']:.6f} (ppl {initial_val['perplexity']:.3f})")
    for record in trainer.validation_history:
        if record["step"] and not record.get("full_split"):
            print(f"  valid @ step {record['step']:3d}: {record['loss']:.6f} (ppl {record['perplexity']:.3f})")
    print(f"final full valid:    {full_val['loss']:.6f} (ppl {full_val['perplexity']:.3f})")
    print(f"grad norms:          min {min(grad_norms):.3f} max {max(grad_norms):.3f} mean {np.mean(grad_norms):.3f}")
    print(f"parameters changed:  {changed}/{len(initial_parameters)} tensors")
    print(f"all params finite:   {finite_parameters}")
    print(f"checkpoint reload:   weights_match={weights_match} tied={tying_after_reload}")
    print(f"peak VRAM:           {trainer.summary()['peak_allocated_vram_mb']:.1f} MiB")
    print(f"elapsed:             {result['elapsed_seconds']:.1f}s")
    print(f"throughput:          {result['tokens_per_second']:,.0f} tokens/s (training)")
    print(f"test.bin untouched:  {test_untouched}")
    print("PILOT COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
