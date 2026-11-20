#!/usr/bin/env python
"""PolyCodeGPT Phase 19 pilot trainer for the verified Phase 18 Scenario B corpus.

This is deliberately separate from scripts/train.py (the v1.1 pilot). It reads
only Scenario B train.bin and valid.bin, uses the v2 tokenizer, and writes to
Phase 19-specific checkpoint/log directories. It never opens a test partition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from polycodegpt.data import TokenBlockDataset, build_dataloader, sha256_file  # noqa: E402
from polycodegpt.model import GPT, ModelConfig  # noqa: E402
from polycodegpt.training import (  # noqa: E402
    Trainer,
    TrainSettings,
    build_validation_loader,
    fixed_validation_indices,
    load_checkpoint,
    rebuild_model,
    require_cuda,
    write_metrics_csv,
)

DATA_DIR = ROOT / "data" / "processed" / "polycodegpt_v2" / "scenario_b"
SCENARIO_MANIFEST = DATA_DIR / "manifest.json"
VERIFICATION_REPORT = ROOT / "data" / "manifests" / "v2" / "verification_report.json"
SPLIT_REPORT = ROOT / "data" / "manifests" / "v2" / "split_report.json"
TOKENIZER_PATH = ROOT / "tokenizer" / "artifacts" / "v2" / "polycodegpt_tokenizer_v2.json"
TOKENIZER_METADATA = ROOT / "tokenizer" / "artifacts" / "v2" / "tokenizer_metadata_v2.json"
MODEL_CONFIG_PATH = ROOT / "configs" / "phase19_model.yaml"
TRAINING_CONFIG_PATH = ROOT / "configs" / "phase19_training.yaml"
CHECKPOINT_DIR = ROOT / "checkpoints" / "polycodegpt_phase19_scenario_b"
LOG_DIR = ROOT / "logs" / "phase19_scenario_b"
CONTEXT_LENGTH = 512
WINDOW_STRIDE = 512
STREAM_DTYPE = np.dtype("<u2")


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required file missing: {path}")
    with path.open("r", encoding="utf-8") as fh:
        value = json.load(fh)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"configuration missing: {path}")
    with path.open("r", encoding="utf-8") as fh:
        value = yaml.safe_load(fh) or {}
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML mapping in {path}")
    return value


def check_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    actual = sha256_file(path)
    if actual.lower() != str(expected).lower():
        raise RuntimeError(f"{label} SHA-256 mismatch: actual={actual}, expected={expected}")
    return actual


def inspect_token_ids(path: Path, vocab_size: int, label: str) -> dict[str, int]:
    """Scan the uint16 stream in chunks and enforce the vocabulary range."""
    values = np.memmap(path, dtype=STREAM_DTYPE, mode="r")
    if values.size == 0:
        del values
        raise ValueError(f"{label} is empty")
    minimum = int(values.min())
    maximum = int(values.max())
    count = int(values.size)
    del values
    if minimum < 0 or maximum >= vocab_size:
        raise ValueError(
            f"{label} token IDs outside [0, {vocab_size - 1}]: min={minimum}, max={maximum}"
        )
    return {"tokens": count, "min_token_id": minimum, "max_token_id": maximum}


def validate_v2_tokenizer() -> tuple[Tokenizer, dict[str, Any], str]:
    metadata = read_json(TOKENIZER_METADATA)
    if metadata.get("tokenizer_version") != "2.0":
        raise ValueError(f"expected tokenizer v2.0, got {metadata.get('tokenizer_version')!r}")
    actual_sha = check_hash(TOKENIZER_PATH, metadata.get("artifact_sha256", ""), "v2 tokenizer")
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    actual_vocab = int(tokenizer.get_vocab_size())
    expected_vocab = int(metadata.get("vocab_size_actual", -1))
    if actual_vocab != expected_vocab or actual_vocab != 24000:
        raise ValueError(
            f"v2 tokenizer vocabulary mismatch: loaded={actual_vocab}, metadata={expected_vocab}, expected=24000"
        )

    recorded_ids = metadata.get("special_token_ids", {})
    vocab = tokenizer.get_vocab()
    if not recorded_ids:
        raise ValueError("tokenizer metadata has no special_token_ids")
    for token, expected_id in recorded_ids.items():
        expected_id = int(expected_id)
        if vocab.get(token) != expected_id:
            raise ValueError(
                f"special-token ID mismatch for {token}: loaded={vocab.get(token)}, expected={expected_id}"
            )
        encoded = tokenizer.encode(token, add_special_tokens=False).ids
        if encoded != [expected_id]:
            raise ValueError(f"special token {token} does not encode to its own ID: {encoded}")
    return tokenizer, metadata, actual_sha


def preflight() -> tuple[dict[str, Any], ModelConfig, dict[str, Any], str, str]:
    """Validate files, hashes, tokenizer and datasets without initializing a model."""
    manifest = read_json(SCENARIO_MANIFEST)
    verification = read_json(VERIFICATION_REPORT)
    split_report = read_json(SPLIT_REPORT)
    if manifest.get("scenario") != "B" or manifest.get("status") != "CAP_APPLIED":
        raise ValueError("Scenario B manifest is not in CAP_APPLIED state")
    if manifest.get("test_partition_read") is not False:
        raise ValueError("Scenario B manifest does not confirm an untouched test partition")
    if verification.get("phase") != 18 or verification.get("scenario") != "B":
        raise ValueError("verification report does not identify Phase 18 Scenario B")
    if verification.get("verdict") != "PHASE 18 COMPLETE — SCENARIO B DATASET VERIFIED":
        raise ValueError(f"unexpected Phase 18 verification verdict: {verification.get('verdict')!r}")
    if verification.get("test_partition_read") is not False:
        raise ValueError("Phase 18 verification does not confirm the test partition was untouched")
    corpus_report = verification.get("corpus", {})
    if corpus_report.get("repository_overlap_count") != 0:
        raise ValueError("Scenario B verification reports repository overlap")
    if int(manifest.get("repository_overlap_count", -1)) != 0:
        raise ValueError("Scenario B manifest reports repository overlap")

    tokenizer, tokenizer_meta, tokenizer_sha = validate_v2_tokenizer()
    if int(tokenizer_meta.get("train_records", -1)) != 1_189_261:
        raise ValueError("tokenizer metadata training record count differs from the verified full pool")
    pre_hash = split_report.get("pre_tokenization_hashes", {}).get("train_sha256")
    if pre_hash != tokenizer_meta.get("train_sha256"):
        raise ValueError("tokenizer training-data hash does not match Phase 18 pre-tokenization provenance")
    full_pool_hash = manifest.get("full_pool_hashes", {}).get("train.jsonl")
    expected_post_hash = split_report.get("post_tokenization", {}).get("train_sha256")
    if full_pool_hash != expected_post_hash:
        raise ValueError("Scenario B full-pool hash does not match the Phase 18 split report")

    files = manifest.get("files", {})
    train_sha = check_hash(DATA_DIR / "train.bin", files.get("train.bin", ""), "Scenario B train.bin")
    valid_sha = check_hash(DATA_DIR / "valid.bin", files.get("valid.bin", ""), "Scenario B valid.bin")

    train_tokens_expected = int(manifest.get("total_selected_tokens", -1))
    valid_tokens_expected = int(corpus_report.get("valid_bin_tokens_from_bytes", -1))
    train_bytes = (DATA_DIR / "train.bin").stat().st_size
    valid_bytes = (DATA_DIR / "valid.bin").stat().st_size
    if train_bytes % 2 or valid_bytes % 2:
        raise ValueError("binary token stream size is not divisible by uint16 item size")
    if train_bytes // 2 != train_tokens_expected:
        raise ValueError("Scenario B train.bin token count does not match its manifest")
    if valid_tokens_expected <= 0 or valid_bytes // 2 != valid_tokens_expected:
        raise ValueError("Scenario B valid.bin token count does not match the Phase 18 verification report")

    train_id_report = inspect_token_ids(DATA_DIR / "train.bin", 24000, "Scenario B train.bin")
    valid_id_report = inspect_token_ids(DATA_DIR / "valid.bin", 24000, "Scenario B valid.bin")

    model_config = ModelConfig.from_yaml(MODEL_CONFIG_PATH)
    model_config.validate_tokenizer_vocab(tokenizer.get_vocab_size())
    if model_config.context_length != CONTEXT_LENGTH:
        raise ValueError(f"Phase 19 context length must be {CONTEXT_LENGTH}, got {model_config.context_length}")

    # Construct the real data-loader objects; the geometry is calculated from
    # the verified stream sizes because the Scenario B manifest is a provenance
    # manifest, not the older v1.1 `splits`-schema manifest.
    train_ds = TokenBlockDataset(
        DATA_DIR / "train.bin", context_length=CONTEXT_LENGTH,
        vocab_size=tokenizer.get_vocab_size(), stride=WINDOW_STRIDE,
    )
    valid_ds = TokenBlockDataset(
        DATA_DIR / "valid.bin", context_length=CONTEXT_LENGTH,
        vocab_size=tokenizer.get_vocab_size(), stride=WINDOW_STRIDE,
    )
    train_first = train_ds.block(0)
    train_last = train_ds.block(len(train_ds) - 1)
    valid_first = valid_ds.block(0)
    valid_last = valid_ds.block(len(valid_ds) - 1)
    for label, block in (("train first", train_first), ("train last", train_last),
                         ("valid first", valid_first), ("valid last", valid_last)):
        if block.shape != (CONTEXT_LENGTH + 1,):
            raise ValueError(f"{label} block has unexpected shape {block.shape}")
        if int(block.min()) < 0 or int(block.max()) >= tokenizer.get_vocab_size():
            raise ValueError(f"{label} block contains an out-of-range token ID")

    report = {
        "phase": 19,
        "scenario": "B",
        "status": "PREFLIGHT_PASSED",
        "test_partition_opened": False,
        "data_dir": str(DATA_DIR),
        "context_length": CONTEXT_LENGTH,
        "window_stride": WINDOW_STRIDE,
        "dtype": "little-endian uint16",
        "tokenizer": {
            "version": tokenizer_meta["tokenizer_version"],
            "vocab_size": tokenizer.get_vocab_size(),
            "sha256": tokenizer_sha,
            "train_data_sha256": tokenizer_meta["train_sha256"],
            "special_token_count": len(tokenizer_meta["special_token_ids"]),
        },
        "train": {
            "sha256": train_sha,
            **train_id_report,
            "blocks": len(train_ds),
            "dropped_trailing_tokens": train_ds.dropped_trailing_tokens,
            "first_last_blocks_checked": True,
        },
        "validation": {
            "sha256": valid_sha,
            **valid_id_report,
            "blocks": len(valid_ds),
            "dropped_trailing_tokens": valid_ds.dropped_trailing_tokens,
            "first_last_blocks_checked": True,
        },
        "repository_overlap_count": 0,
        "phase18_verdict": verification["verdict"],
        "model": model_config.to_dict(),
        "parameter_count": model_config.estimate_parameters(),
    }
    train_ds.close()
    valid_ds.close()
    return report, model_config, manifest, tokenizer_sha, sha256_file(SCENARIO_MANIFEST)


def make_settings(config: dict[str, Any], micro_batch: int, max_steps: int) -> TrainSettings:
    cfg = config.get("training", {})
    schedule = cfg.get("schedule", {})
    evaluation = cfg.get("evaluation", {})
    effective_batch = int(cfg.get("effective_batch_size", 32))
    if effective_batch % micro_batch:
        raise ValueError(f"effective_batch_size {effective_batch} is not divisible by micro batch {micro_batch}")
    total_steps = max(max_steps, int(schedule.get("total_steps", max_steps)))
    return TrainSettings(
        micro_batch_size=micro_batch,
        gradient_accumulation_steps=effective_batch // micro_batch,
        max_steps=max_steps,
        additional_steps=max_steps,
        total_steps=total_steps,
        warmup_fraction=float(schedule.get("warmup_fraction", 0.02)),
        min_lr_ratio=float(schedule.get("min_lr_ratio", 0.1)),
        learning_rate=float(cfg.get("learning_rate", 3e-4)),
        weight_decay=float(cfg.get("weight_decay", 0.1)),
        gradient_clip=float(cfg.get("gradient_clip", 1.0)),
        precision=str(cfg.get("precision", "auto")),
        seed=int(cfg.get("seed", 42)),
        context_length=CONTEXT_LENGTH,
        log_every=int(cfg.get("log_every", 10)),
        eval_steps=tuple(int(x) for x in evaluation.get("steps", [50, 100, 150, 200])),
        checkpoint_steps=tuple(int(x) for x in cfg.get("checkpoint_steps", [50, 100, 150, 200])),
        initial_validation=True,
        max_skipped_steps=int(cfg.get("max_skipped_steps", 10)),
        deterministic=bool(cfg.get("deterministic", True)),
    )


def probe_micro_batch(
    model_config: ModelConfig,
    train_ds: TokenBlockDataset,
    device: torch.device,
    training_config: dict[str, Any],
) -> tuple[int, list[dict[str, Any]]]:
    """Find the largest configured micro-batch that completes a real optimizer update."""
    cfg = training_config.get("training", {})
    start = int(cfg.get("micro_batch_size", 4))
    effective_batch = int(cfg.get("effective_batch_size", 32))
    candidates: list[int] = []
    value = start
    while value >= 1:
        if effective_batch % value == 0:
            candidates.append(value)
        if value == 1:
            break
        value = max(1, value // 2)
    if not candidates or candidates[-1] != 1:
        candidates.append(1)

    attempts: list[dict[str, Any]] = []
    seed = int(cfg.get("seed", 42))
    precision = str(cfg.get("precision", "auto"))
    learning_rate = float(cfg.get("learning_rate", 3e-4))
    weight_decay = float(cfg.get("weight_decay", 0.1))
    gradient_clip = float(cfg.get("gradient_clip", 1.0))

    for micro in candidates:
        probe_model = None
        probe_trainer = None
        try:
            set_seeds(seed)
            probe_model = GPT(model_config).to(device)
            loader = build_dataloader(
                train_ds, batch_size=micro, shuffle=False, seed=seed,
                pin_memory=True, drop_last=False,
            )
            probe_settings = TrainSettings(
                micro_batch_size=micro,
                gradient_accumulation_steps=1,
                max_steps=1,
                additional_steps=1,
                total_steps=1,
                warmup_fraction=0.0,
                min_lr_ratio=1.0,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                gradient_clip=gradient_clip,
                precision=precision,
                seed=seed,
                context_length=CONTEXT_LENGTH,
                log_every=1,
                eval_steps=(),
                checkpoint_steps=(),
                initial_validation=False,
                deterministic=True,
            )
            probe_trainer = Trainer(
                probe_model, loader, probe_settings,
                device=device, val_loader=None,
                train_tokens=train_ds.token_count,
                checkpoint_dir=None,
                model_config=model_config,
                log_fn=lambda _message: None,
            )
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            probe_trainer.run()  # one actual forward/backward/AdamW update
            torch.cuda.synchronize(device)
            peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
            attempts.append({
                "micro_batch_size": micro,
                "ok": True,
                "peak_allocated_vram_mb": round(peak_mb, 2),
                "seconds": round(time.perf_counter() - started, 3),
            })
            del probe_trainer, probe_model, loader
            torch.cuda.empty_cache()
            return micro, attempts
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()
            if not is_oom:
                raise
            attempts.append({"micro_batch_size": micro, "ok": False, "error": "CUDA out of memory"})
            del probe_trainer, probe_model
            torch.cuda.empty_cache()
    raise RuntimeError("CUDA memory probe failed even with micro_batch_size=1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PolyCodeGPT Phase 19 / Scenario B pilot")
    parser.add_argument("--preflight-only", action="store_true", help="verify data/model/tokenizer without training")
    parser.add_argument("--max-steps", type=int, default=None, help="override pilot optimizer steps")
    parser.add_argument("--device", default="cuda", help="training device (default: cuda)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    preflight_report, model_config, manifest, tokenizer_sha, manifest_sha = preflight()
    training_config = load_yaml(TRAINING_CONFIG_PATH)
    cfg = training_config.get("training", {})
    max_steps = int(args.max_steps or cfg.get("max_steps", 200))
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    print(json.dumps(preflight_report, indent=2))
    if args.preflight_only:
        print("\nPHASE 19 PREFLIGHT PASSED — no model training performed")
        return 0

    device = require_cuda(args.device)
    train_ds = TokenBlockDataset(
        DATA_DIR / "train.bin", context_length=CONTEXT_LENGTH,
        vocab_size=model_config.vocab_size, stride=WINDOW_STRIDE,
    )
    valid_ds = TokenBlockDataset(
        DATA_DIR / "valid.bin", context_length=CONTEXT_LENGTH,
        vocab_size=model_config.vocab_size, stride=WINDOW_STRIDE,
    )

    print("\n--- CUDA micro-batch memory probe ---")
    chosen_micro, probe_attempts = probe_micro_batch(model_config, train_ds, device, training_config)
    effective_batch = int(cfg.get("effective_batch_size", 32))
    settings = make_settings(training_config, chosen_micro, max_steps)
    print(f"selected micro batch: {chosen_micro}")
    print(f"gradient accumulation: {settings.gradient_accumulation_steps}")
    print(f"effective batch: {effective_batch} sequences")
    print(f"supervised tokens/update: {settings.tokens_per_step:,}")

    seed = settings.seed
    train_loader = build_dataloader(
        train_ds, batch_size=chosen_micro, shuffle=True, seed=seed,
        pin_memory=True, drop_last=False,
    )
    eval_cfg = cfg.get("evaluation", {})
    val_count = int(eval_cfg.get("val_blocks", 256))
    val_batch_requested = int(eval_cfg.get("val_batch_size", 4))
    val_batch = min(val_batch_requested, chosen_micro)
    val_indices = fixed_validation_indices(len(valid_ds), val_count, seed=seed)
    val_loader = build_validation_loader(valid_ds, val_indices, batch_size=val_batch)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with (LOG_DIR / "preflight_report.json").open("w", encoding="utf-8") as fh:
        json.dump(preflight_report, fh, indent=2)

    set_seeds(settings.seed)
    model = GPT(model_config).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    if parameter_count != model_config.estimate_parameters():
        raise RuntimeError("model parameter count differs from analytical estimate")

    trainer = Trainer(
        model,
        train_loader,
        settings,
        device=device,
        val_loader=val_loader,
        train_tokens=train_ds.token_count,
        checkpoint_dir=CHECKPOINT_DIR,
        model_config=model_config,
        tokenizer_sha256=tokenizer_sha,
        data_manifest_sha256=manifest_sha,
    )
    run_started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    run_result = trainer.run()
    trainer.save(CHECKPOINT_DIR / "latest.pt")
    elapsed = time.perf_counter() - started

    write_metrics_csv(LOG_DIR / "metrics.csv", trainer.history)
    with (LOG_DIR / "validation_history.json").open("w", encoding="utf-8") as fh:
        json.dump(trainer.validation_history, fh, indent=2)

    checkpoint = load_checkpoint(CHECKPOINT_DIR / "latest.pt", map_location="cpu")
    if checkpoint.get("tokenizer_sha256") != tokenizer_sha:
        raise RuntimeError("checkpoint tokenizer SHA-256 does not match the Phase 19 tokenizer")
    if checkpoint.get("data_manifest_sha256") != manifest_sha:
        raise RuntimeError("checkpoint dataset-manifest SHA-256 does not match Scenario B")
    restored = rebuild_model(checkpoint, device="cpu")
    current_state = model.state_dict()
    restored_state = restored.state_dict()
    weights_match = all(
        torch.equal(value.detach().cpu(), restored_state[key].detach().cpu())
        for key, value in current_state.items()
    )
    weights_tied = restored.lm_head.weight is restored.token_embedding.weight
    if not weights_match or not weights_tied:
        raise RuntimeError("checkpoint reload integrity check failed")

    peak_vram_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    summary = {
        "phase": 19,
        "scenario": "B",
        "run_started_utc": run_started_utc,
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "model": model_config.to_dict(),
        "parameter_count": parameter_count,
        "tokenizer_sha256": tokenizer_sha,
        "scenario_b_manifest_sha256": manifest_sha,
        "scenario_b_binary_hashes": manifest["files"],
        "training_settings": settings.to_dict(),
        "memory_probe": probe_attempts,
        "train_tokens_available": train_ds.token_count,
        "train_blocks": len(train_ds),
        "validation_tokens_available": valid_ds.token_count,
        "validation_blocks": len(valid_ds),
        "validation_subset_blocks": len(val_indices),
        "validation_batch_size": val_batch,
        "validation_subset_tokens": len(val_indices) * CONTEXT_LENGTH,
        "initial_validation": trainer.validation_history[0] if trainer.validation_history else None,
        "final_validation": trainer.validation_history[-1] if trainer.validation_history else None,
        "validation_history": trainer.validation_history,
        "train_result": run_result,
        "optimizer_steps": trainer.step_index,
        "tokens_seen_by_trainer": trainer.tokens_seen,
        "elapsed_seconds": round(elapsed, 3),
        "peak_allocated_vram_mb": round(peak_vram_mb, 2),
        "checkpoint_reload": {
            "weights_match": weights_match,
            "weights_tied": weights_tied,
            "step": int(checkpoint.get("step", -1)),
            "tokens_seen": int(checkpoint.get("tokens_seen", 0)),
        },
        "test_partition_opened": False,
    }
    with (LOG_DIR / "phase19_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, allow_nan=False)

    print("\n--- PHASE 19 PILOT SUMMARY ---")
    print(f"optimizer steps: {trainer.step_index}")
    print(f"tokens seen: {trainer.tokens_seen:,}")
    print(f"peak allocated VRAM: {peak_vram_mb:.1f} MiB")
    if trainer.validation_history:
        print(f"initial validation loss: {trainer.validation_history[0]['loss']:.6f}")
        print(f"final validation loss: {trainer.validation_history[-1]['loss']:.6f}")
    print(f"checkpoint reload: weights_match={weights_match}, weights_tied={weights_tied}")
    print("test partition opened: no")
    print("PHASE 19 PILOT COMPLETE")
    train_ds.close()
    valid_ds.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
