#!/usr/bin/env python
"""Evaluate a PolyCodeGPT Phase 20 checkpoint on the entire validation split.

This script reads only Scenario B valid.bin. It does not access train.bin or any
held-out test partition, and it does not modify model checkpoints.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from polycodegpt.data import TokenBlockDataset, sha256_file
from polycodegpt.training import (
    build_validation_loader,
    evaluate_loss,
    load_checkpoint,
    require_cuda,
    resolve_precision,
    weights_are_tied,
)
from polycodegpt.training.checkpoint import rebuild_model

DATA_DIR = ROOT / "data" / "processed" / "polycodegpt_v2" / "scenario_b"
SCENARIO_MANIFEST = DATA_DIR / "manifest.json"
TOKENIZER_PATH = ROOT / "tokenizer" / "artifacts" / "v2" / "polycodegpt_tokenizer_v2.json"
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "polycodegpt_phase20_scenario_b" / "best.pt"
DEFAULT_REPORT = ROOT / "logs" / "phase20_scenario_b" / "full_validation_best.json"
CONTEXT_LENGTH = 512
WINDOW_STRIDE = 512
EXPECTED_TOKENIZER_SHA256 = "0e7f466ee516fe34becc638a3451eba29bced5efa5317647e486ab735cc7f120"
EXPECTED_MANIFEST_SHA256 = "f0b84da44a2fc02f5a03f32867515eaeb3daf931d77bdbe48d1fe83036e1cbf8"
EXPECTED_VALID_BIN_SHA256 = "63e7e4218657734dc2f64e6ade530e6fac9156694b78caa052d60cbd04de4c80"


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial")
    try:
        temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a Phase 20 checkpoint on all Scenario B validation blocks"
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
        help="checkpoint to evaluate (default: Phase 20 best.pt)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="validation sequences per batch (default: 4)",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="evaluation device (default: cuda; CPU is not enabled by this script)",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_REPORT,
        help="JSON report destination",
    )
    parser.add_argument(
        "--overwrite-report", action="store_true",
        help="allow replacing an existing evaluation JSON report",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if not args.checkpoint.is_absolute():
        args.checkpoint = (ROOT / args.checkpoint).resolve()
    if not args.output.is_absolute():
        args.output = (ROOT / args.output).resolve()
    if args.output.exists() and not args.overwrite_report:
        raise FileExistsError(
            f"Report already exists: {args.output}. Choose another --output path "
            "or pass --overwrite-report explicitly."
        )

    valid_bin = DATA_DIR / "valid.bin"
    for required in (args.checkpoint, valid_bin, SCENARIO_MANIFEST, TOKENIZER_PATH):
        if not required.is_file():
            raise FileNotFoundError(f"Required file not found: {required}")

    print("========== FULL VALIDATION PREFLIGHT ==========")
    tokenizer_hash = sha256_file(TOKENIZER_PATH)
    manifest_hash = sha256_file(SCENARIO_MANIFEST)
    valid_hash = sha256_file(valid_bin)
    print("Checkpoint:", args.checkpoint)
    print("Tokenizer SHA-256:", tokenizer_hash)
    print("Scenario manifest SHA-256:", manifest_hash)
    print("valid.bin SHA-256:", valid_hash)

    if tokenizer_hash != EXPECTED_TOKENIZER_SHA256:
        raise RuntimeError("Tokenizer hash differs from the verified Phase 19/20 tokenizer")
    if manifest_hash != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("Scenario B manifest hash differs from the verified manifest")
    if valid_hash != EXPECTED_VALID_BIN_SHA256:
        raise RuntimeError("Validation binary hash differs from the verified Scenario B file")

    device = require_cuda(args.device)
    plan = resolve_precision("auto", device)
    if plan.precision != "bf16":
        raise RuntimeError(
            f"Expected BF16 evaluation on this GPU, but precision resolver selected {plan.precision}"
        )

    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    checkpoint_tokenizer_hash = checkpoint.get("tokenizer_sha256")
    checkpoint_manifest_hash = checkpoint.get("data_manifest_sha256")
    if checkpoint_tokenizer_hash != tokenizer_hash:
        raise RuntimeError("Checkpoint tokenizer hash does not match tokenizer artifact")
    if checkpoint_manifest_hash != manifest_hash:
        raise RuntimeError("Checkpoint dataset-manifest hash does not match Scenario B manifest")
    if not checkpoint.get("model_config"):
        raise RuntimeError("Checkpoint is missing model_config")

    checkpoint_meta = {
        "format_version": checkpoint.get("format_version"),
        "global_step": int(checkpoint.get("step", -1)),
        "best_step": checkpoint.get("best_step"),
        "best_validation_loss_on_training_subset": checkpoint.get("best_val_loss"),
        "tokens_seen": int(checkpoint.get("tokens_seen", 0)),
        "weights_tied_recorded": bool(checkpoint.get("weights_tied", False)),
        "tokenizer_sha256": checkpoint_tokenizer_hash,
        "data_manifest_sha256": checkpoint_manifest_hash,
    }

    print("\n========== CHECKPOINT ==========")
    print(json.dumps(checkpoint_meta, indent=2))
    if not checkpoint_meta["weights_tied_recorded"]:
        raise RuntimeError("Checkpoint metadata says embeddings are not tied")

    model = rebuild_model(checkpoint, device=device)
    del checkpoint
    gc.collect()
    if not weights_are_tied(model):
        raise RuntimeError("Restored evaluation model does not have tied embeddings")
    model.eval()

    # Only the validation binary is opened below. No training or test partition is read.
    dataset = TokenBlockDataset(
        valid_bin,
        context_length=CONTEXT_LENGTH,
        vocab_size=int(model.config.vocab_size),
        stride=WINDOW_STRIDE,
    )
    loader = build_validation_loader(dataset, indices=None, batch_size=args.batch_size)
    expected_sequences = len(dataset)
    expected_batches = (expected_sequences + args.batch_size - 1) // args.batch_size
    print("\n========== EVALUATION PLAN ==========")
    print("Device:", torch.cuda.get_device_name(device))
    print("Precision:", plan.precision)
    print("Validation blocks:", expected_sequences)
    print("Batch size:", args.batch_size)
    print("Expected batches:", expected_batches)
    print("Test partition opened: no")

    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    result = evaluate_loss(
        model,
        loader,
        device,
        autocast_dtype=plan.autocast_dtype,
        autocast_device_type=plan.device_type,
    )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / 1024**2

    complete = (
        int(result["sequences"]) == expected_sequences
        and int(result["batches"]) == expected_batches
    )
    if not complete:
        raise RuntimeError(
            "Full validation did not consume the expected dataset: "
            f"sequences {result['sequences']}/{expected_sequences}, "
            f"batches {result['batches']}/{expected_batches}"
        )

    report: dict[str, Any] = {
        "evaluation_type": "full_validation",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_metadata": checkpoint_meta,
        "model_parameter_count": int(sum(p.numel() for p in model.parameters())),
        "model_config": model.config.to_dict(),
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "precision": plan.precision,
        "batch_size": args.batch_size,
        "expected_validation_blocks": expected_sequences,
        "expected_validation_batches": expected_batches,
        "validation_result": {
            "loss": float(result["loss"]),
            "perplexity": float(result["perplexity"]),
            "tokens": int(result["tokens"]),
            "sequences": int(result["sequences"]),
            "batches": int(result["batches"]),
        },
        "complete_validation_consumed": complete,
        "elapsed_seconds": elapsed,
        "tokens_per_second": float(result["tokens"]) / elapsed if elapsed > 0 else 0.0,
        "peak_allocated_vram_mb": peak_vram_mb,
        "tokenizer_sha256": tokenizer_hash,
        "scenario_b_manifest_sha256": manifest_hash,
        "valid_bin_sha256": valid_hash,
        "test_partition_opened": False,
    }
    write_json_atomic(args.output, report)

    print("\n========== FULL VALIDATION RESULT ==========")
    print(json.dumps(report, indent=2))
    print("\nReport saved to:", args.output)
    print("FULL VALIDATION COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
