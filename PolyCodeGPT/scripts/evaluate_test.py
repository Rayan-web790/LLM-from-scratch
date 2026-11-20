"""PolyCodeGPT Phase 6A: held-out test evaluation (read-only).

Loads the final checkpoint (step 8,130), evaluates the entire held-out test split
with the existing token-weighted cross-entropy, and writes a markdown report.

This script performs INFERENCE ONLY: no optimizer, no gradients, no training.
It does not modify any checkpoint, tokenizer, or dataset.

The user runs this directly in their terminal; Freebuff does NOT execute it.

Usage:
    C:/Users/UltraPc/Desktop/LLM FROM SCRATCH/.venv/Scripts/python.exe scripts/evaluate_test.py
    C:/Users/UltraPc/Desktop/LLM FROM SCRATCH/.venv/Scripts/python.exe scripts/evaluate_test.py --checkpoint checkpoints/polycodegpt_first_pass/latest.pt
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch

from polycodegpt.data import build_dataloader, load_tokenizer, open_split, sha256_file
from polycodegpt.data.integrity import load_manifest
from polycodegpt.data.tokenizer_io import verify_special_tokens
from polycodegpt.model import GPT, ModelConfig
from polycodegpt.model.checkpoint import load_checkpoint as load_train_checkpoint
from polycodegpt.training import rebuild_model, require_cuda, weights_are_tied
from polycodegpt.training.checkpoint import load_checkpoint as load_training_checkpoint
from polycodegpt.training.evaluate import evaluate_loss, perplexity

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "tokenized_v1_1"
MANIFEST_PATH = DATA_DIR / "manifest.json"
MODEL_CONFIG_PATH = PROJECT_ROOT / "configs" / "model.yaml"
TOKENIZER_PATH = PROJECT_ROOT / "tokenizer" / "artifacts" / "polycodegpt_tokenizer_v1_1.json"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "polycodegpt_first_pass" / "latest.pt"
REPORT_PATH = PROJECT_ROOT / "reports" / "phase6_test_evaluation_report.md"

EXPECTED_TEST_BLOCKS = 14546
EXPECTED_TEST_TARGET_TOKENS = 7447552
EXPECTED_TEST_DROPPED = 270


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="PolyCodeGPT held-out test evaluation")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-batches", type=int, default=None, help="optional cap for smoke testing")
    return parser.parse_args(argv)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    args = parse_args(argv)
    started_wall = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    run_started = time.perf_counter()

    device = require_cuda(args.device)

    manifest = load_manifest(args.manifest)
    assert manifest["completed_successfully"], "manifest not completed successfully"

    # ---- consistency: manifest vs config ----
    model_config = ModelConfig.from_yaml(MODEL_CONFIG_PATH)
    if model_config.to_dict() != manifest["model_config"]:
        raise SystemExit("model config does not match manifest model_config")
    if model_config.context_length != manifest["context_length"]:
        raise SystemExit("context length mismatch between config and manifest")

    tokenizer = load_tokenizer(TOKENIZER_PATH, vocab_size=model_config.vocab_size)
    if tokenizer.get_vocab_size() != manifest["tokenizer"]["vocab_size"]:
        raise SystemExit("tokenizer vocab size does not match manifest")
    if sha256_of(TOKENIZER_PATH) != manifest["tokenizer"]["sha256"]:
        raise SystemExit("tokenizer SHA-256 does not match manifest")
    verify_special_tokens(tokenizer)

    # ---- consistency: test split geometry ----
    test_info = manifest["splits"]["test"]
    test_path = args.data_dir / test_info["file"]
    if not test_path.is_file():
        raise SystemExit(f"test split not found: {test_path}")

    test_stat_before = test_path.stat()

    # ---- load checkpoint ----
    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint_path}")

    payload = load_training_checkpoint(checkpoint_path, map_location="cpu")
    if payload.get("tokenizer_sha256") and payload["tokenizer_sha256"] != sha256_of(TOKENIZER_PATH):
        raise SystemExit("checkpoint tokenizer hash does not match tokenizer on disk")
    if payload.get("data_manifest_sha256") and payload["data_manifest_sha256"] != sha256_of(args.manifest):
        raise SystemExit("checkpoint manifest hash does not match manifest on disk")
    if payload["model_config"] != model_config.to_dict():
        raise SystemExit("checkpoint model config does not match configs/model.yaml")

    # rebuild model + load weights
    model = rebuild_model(payload, device="cpu")
    if not all(torch.isfinite(p).all().item() for p in model.parameters()):
        raise SystemExit("checkpoint contains non-finite parameters")
    if not weights_are_tied(model):
        raise SystemExit("tied embeddings did not survive checkpoint loading")

    model = model.to(device).eval()

    # ---- open ONLY the test split ----
    test_ds = open_split(args.data_dir, "test", manifest)
    # geometry assertions
    assert test_ds.num_blocks == EXPECTED_TEST_BLOCKS, (
        f"test blocks {test_ds.num_blocks} != expected {EXPECTED_TEST_BLOCKS}"
    )
    assert test_ds.dropped_trailing_tokens == EXPECTED_TEST_DROPPED, (
        f"dropped trailing {test_ds.dropped_trailing_tokens} != {EXPECTED_TEST_DROPPED}"
    )
    assert test_ds.num_blocks * model_config.context_length == EXPECTED_TEST_TARGET_TOKENS, (
        "target token count mismatch"
    )

    test_loader = build_dataloader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ---- inference only ----
    plan_precision = payload.get("precision", "bf16")
    autocast_dtype = torch.bfloat16 if plan_precision == "bf16" else None
    autocast_device_type = "cuda"

    torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        result = evaluate_loss(
            model,
            test_loader,
            device,
            autocast_dtype=autocast_dtype,
            autocast_device_type=autocast_device_type,
            max_batches=args.max_batches,
        )

    elapsed = time.perf_counter() - run_started
    allocated, peak = (torch.cuda.memory_allocated(device) / 1024**2,
                       torch.cuda.max_memory_allocated(device) / 1024**2)

    test_stat_after = test_path.stat()
    test_untouched_stat = (
        test_stat_before.st_size == test_stat_after.st_size
        and test_stat_before.st_mtime_ns == test_stat_after.st_mtime_ns
    )

    # ---- report ----
    report_text = f"""# Phase 6 — Held-Out Test Evaluation Report

**Status:** RUN COMPLETE (executed by user)

**Project:** PolyCodeGPT

## 1. Checkpoint and model

- **Checkpoint:** `{checkpoint_path}`
- **Global step:** {payload['step']}
- **Parameters:** {sum(p.numel() for p in model.parameters()):,}
- **Precision:** {plan_precision}
- **Weights tied after load:** {weights_are_tied(model)}
- **All parameters finite:** {all(torch.isfinite(p).all().item() for p in model.parameters())}

## 2. Data consistency

- **Manifest:** `{args.manifest}` (v{manifest['manifest_version']})
- **Tokenizer:** `{TOKENIZER_PATH}` (sha256 {sha256_of(TOKENIZER_PATH)[:16]}...)
- **Tokenizer vocab size:** {tokenizer.get_vocab_size()} (matches model and manifest)
- **Context length:** {model_config.context_length}
- **Window geometry:** overlapping, stride {manifest['window_stride']}, window {manifest['window_length']}, overlap 1
- **Checkpoint tokenizer hash matches:** True
- **Checkpoint manifest hash matches:** True
- **Model config matches manifest:** True

## 3. Test split

- **File:** `{test_path}`
- **Blocks evaluated:** {result['sequences']:,} (of {test_ds.num_blocks:,})
- **Target tokens evaluated:** {result['tokens']:,}
- **Trailing tokens excluded:** {test_ds.dropped_trailing_tokens:,}
- **Batches:** {result['batches']:,}
- **Batch size:** {args.batch_size}

## 4. Results

- **Token-weighted cross-entropy (loss):** {result['loss']:.6f}
- **Perplexity:** {result['perplexity']:.4f}
- **Evaluated target tokens:** {result['tokens']:,}
- **Loss sum:** {result['loss_sum']:.6f}

## 5. Runtime and memory

- **Elapsed evaluation time:** {elapsed:.1f} s
- **Peak GPU memory:** {peak:.1f} MiB
- **Allocated GPU memory (end):** {allocated:.1f} MiB

## 6. Test-data access record

- **test.bin opened:** True (read-only memmap via TokenBlockDataset, for evaluation only)
- **test.bin size before/after:** {test_stat_before.st_size} B / {test_stat_after.st_size} B (unchanged: {test_untouched_stat})
- **test.bin mtime_ns before/after:** {test_stat_before.st_mtime_ns} / {test_stat_after.st_mtime_ns} (unchanged: {test_untouched_stat})
- **train.bin / valid.bin:** NOT opened by this script
- **No training, no gradients, no optimizer updates performed.**

## 7. Integrity

- All model parameters finite: {all(torch.isfinite(p).all().item() for p in model.parameters())}
- Tied embeddings survive load: {weights_are_tied(model)}
- Test split stat unchanged: {test_untouched_stat}

---
Generated by scripts/evaluate_test.py
"""

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report_text, encoding="utf-8")
    print(report_text)
    print(f"\nReport written to {args.report}")
    test_ds.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())