"""Phase 12A Part A — freeze and verify the experiment baseline.

Read-only. Computes SHA-256 for every protected artifact, checks the hashes the
checkpoint recorded at save time against the files on disk, and proves the
optimizer / scheduler / RNG state can be restored from a copy without writing
anything.

    python experiments/phase12a/baseline_manifest.py

Writes ``reports/phase12a_baseline.json`` and ``reports/phase12a_baseline.md``.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from polycodegpt.model import GPT, ModelConfig  # noqa: E402
from polycodegpt.training.optimizer import OptimizerConfig, build_optimizer  # noqa: E402
from polycodegpt.training.scheduler import ScheduleConfig, lr_factor  # noqa: E402

CHECKPOINT_DIR = ROOT / "checkpoints" / "polycodegpt_first_pass"
BASELINE_CHECKPOINT = CHECKPOINT_DIR / "step_008130.pt"
TOKENIZER = ROOT / "tokenizer" / "artifacts" / "polycodegpt_tokenizer_v1_1.json"
DATA_MANIFEST = ROOT / "data" / "processed" / "tokenized_v1_1" / "manifest.json"

OUT_JSON = ROOT / "reports" / "phase12a_baseline.json"
OUT_MD = ROOT / "reports" / "phase12a_baseline.md"

TRACKED_GLOBS = {
    "checkpoint": ["checkpoints/polycodegpt_first_pass/step_008130.pt",
                   "checkpoints/polycodegpt_first_pass/best.pt",
                   "checkpoints/polycodegpt_first_pass/latest.pt"],
    "tokenizer": ["tokenizer/artifacts/*.json", "tokenizer/artifacts/*.csv"],
    "data_manifest": ["data/manifests/*.json", "data/manifests/*.csv",
                      "data/processed/tokenized_v1_1/manifest.json"],
    "config": ["configs/*.yaml"],
    "report": ["reports/*.json", "reports/*.md", "reports/*.csv", "reports/*.yaml"],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect() -> dict[str, list[dict]]:
    inventory: dict[str, list[dict]] = {}
    for kind, patterns in TRACKED_GLOBS.items():
        entries: dict[str, dict] = {}
        for pattern in patterns:
            for path in sorted(ROOT.glob(pattern)):
                if not path.is_file():
                    continue
                rel = path.relative_to(ROOT).as_posix()
                # This phase's own outputs are produced after the baseline is
                # frozen, so they must not become part of the frozen inventory.
                if rel.startswith("reports/phase12a_"):
                    continue
                if rel in entries:
                    continue
                stat = path.stat()
                entries[rel] = {
                    "path": rel,
                    "sha256": sha256_file(path),
                    "bytes": stat.st_size,
                    "mtime_utc": datetime.fromtimestamp(
                        stat.st_mtime, timezone.utc
                    ).isoformat(timespec="seconds"),
                }
        inventory[kind] = list(entries.values())
    return inventory


def verify_checkpoint() -> dict:
    payload = torch.load(BASELINE_CHECKPOINT, map_location="cpu", weights_only=False)
    tokenizer_sha = sha256_file(TOKENIZER)
    manifest_sha = sha256_file(DATA_MANIFEST)

    checks: dict[str, dict] = {}
    checks["step_is_8130"] = {"ok": payload["step"] == 8130, "value": payload["step"]}
    checks["tokenizer_hash_matches"] = {
        "ok": payload["tokenizer_sha256"] == tokenizer_sha,
        "checkpoint": payload["tokenizer_sha256"], "on_disk": tokenizer_sha,
    }
    checks["data_manifest_hash_matches"] = {
        "ok": payload["data_manifest_sha256"] == manifest_sha,
        "checkpoint": payload["data_manifest_sha256"], "on_disk": manifest_sha,
    }

    model = GPT(ModelConfig(**payload["model_config"]))
    missing, unexpected = model.load_state_dict(payload["model_state"], strict=True)
    checks["model_state_loads_strictly"] = {
        "ok": not missing and not unexpected,
        "missing": len(missing), "unexpected": len(unexpected),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "checkpoint_parameter_count": payload["parameter_count"],
    }

    settings = payload["settings"]
    optimizer = build_optimizer(
        model,
        OptimizerConfig(
            learning_rate=float(settings["learning_rate"]),
            weight_decay=float(settings["weight_decay"]),
        ),
    )
    optimizer.load_state_dict(payload["optimizer_state"])
    opt_state = payload["optimizer_state"]
    param_count = sum(len(g["params"]) for g in optimizer.param_groups)
    exp_avg_present = sum(
        1 for s in opt_state["state"].values() if "exp_avg" in s and "exp_avg_sq" in s
    )
    steps = {int(s.get("step", -1)) for s in opt_state["state"].values()}
    checks["optimizer_state_restores"] = {
        "ok": exp_avg_present == param_count and len(steps) == 1,
        "parameter_groups": len(optimizer.param_groups),
        "tensors_with_moments": exp_avg_present,
        "parameters": param_count,
        "adam_step_values": sorted(steps),
        "group_weight_decay": [g.get("weight_decay") for g in optimizer.param_groups],
    }

    sched_state = payload["scheduler_state"]
    schedule = ScheduleConfig(
        total_steps=int(sched_state["total_steps"]),
        warmup_fraction=float(settings["warmup_fraction"]),
        min_lr_ratio=float(sched_state["min_lr_ratio"]),
    )
    last_step = int(sched_state["last_step"])
    base_lr = float(sched_state["base_lrs"][0])
    next_lr = base_lr * lr_factor(last_step, schedule.total_steps, schedule.warmup_steps,
                                  schedule.min_lr_ratio)
    checks["scheduler_state_restores"] = {
        "ok": schedule.total_steps == int(settings["total_steps"])
        and schedule.warmup_steps == int(sched_state["warmup_steps"]),
        "total_steps": schedule.total_steps,
        "warmup_steps": schedule.warmup_steps,
        "min_lr_ratio": schedule.min_lr_ratio,
        "last_step": last_step,
        "base_lr": base_lr,
        "next_step_lr": round(next_lr, 10),
    }

    rng = payload["rng_state"]
    loader_state = payload["loader_epoch_state"]
    checks["rng_state_present"] = {
        "ok": set(rng) == {"torch", "cuda", "python", "numpy", "loader_generator"},
        "keys": sorted(rng),
        "loader_epoch_state_shape": list(loader_state.shape),
        "loader_epoch_state_dtype": str(loader_state.dtype),
    }

    # Accounting follows engine.py: tokens accumulate per micro-batch, sequences
    # per optimizer step, and the final step may hold a partial group
    # (``partial_final_update``), which the engine normalises by actual size.
    context = int(settings["context_length"])
    effective_batch = int(settings["effective_batch_size"])
    micro_batch_size = int(settings["micro_batch_size"])
    sequences = int(payload["sequences"])
    micro_batches = int(payload["micro_batches"])
    tokens_seen = int(payload["tokens_seen"])
    partial_group = sequences - (int(payload["step"]) - 1) * effective_batch
    micro_slack = micro_batches * micro_batch_size - sequences
    checks["token_accounting"] = {
        "ok": (
            tokens_seen == sequences * context
            and 0 < partial_group <= effective_batch
            and 0 <= micro_slack < micro_batch_size
        ),
        "tokens_seen": tokens_seen,
        "sequences_x_context": sequences * context,
        "sequences": sequences,
        "micro_batches": micro_batches,
        "partial_final_group_sequences": partial_group,
        "partial_final_micro_batch_slack": micro_slack,
    }

    return {
        "checkpoint": BASELINE_CHECKPOINT.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": sha256_file(BASELINE_CHECKPOINT),
        "step": payload["step"],
        "tokens_seen": payload["tokens_seen"],
        "sequences": payload["sequences"],
        "micro_batches": payload["micro_batches"],
        "best_val_loss": payload["best_val_loss"],
        "best_step": payload["best_step"],
        "precision": payload["precision"],
        "seed": payload["seed"],
        "settings": settings,
        "scheduler_state": {k: sched_state[k] for k in ("total_steps", "warmup_steps",
                                                        "min_lr_ratio", "last_step")},
        "model_config": payload["model_config"],
        "checks": checks,
        "all_checks_passed": all(c["ok"] for c in checks.values()),
    }


def write_markdown(payload: dict) -> None:
    checkpoint = payload["checkpoint"]
    inv = payload["inventory"]
    lines = [
        "# Phase 12A — Baseline Freeze and Verification",
        "",
        f"Recorded {payload['recorded_utc']}. Read-only: every hash below was computed from the "
        "existing files, and nothing was written outside `reports/`.",
        "",
        "## Baseline checkpoint",
        "",
        "| field | value |",
        "|---|---|",
        f"| file | `{checkpoint['checkpoint']}` |",
        f"| sha256 | `{checkpoint['checkpoint_sha256']}` |",
        f"| optimizer step | {checkpoint['step']} |",
        f"| tokens seen | {checkpoint['tokens_seen']:,} |",
        f"| best validation loss | {checkpoint['best_val_loss']:.6f} (step {checkpoint['best_step']}) |",
        f"| precision | {checkpoint['precision']} |",
        f"| seed | {checkpoint['seed']} |",
        "",
        "## Training configuration recorded in the checkpoint",
        "",
        "| setting | value |",
        "|---|---|",
    ]
    for key, value in checkpoint["settings"].items():
        lines.append(f"| {key} | {value} |")
    lines += [
        f"| total steps | {checkpoint['scheduler_state']['total_steps']} |",
        f"| warmup steps | {checkpoint['scheduler_state']['warmup_steps']} |",
        f"| min lr ratio | {checkpoint['scheduler_state']['min_lr_ratio']} |",
        "",
        "## Model configuration",
        "",
        "```json",
        json.dumps(checkpoint["model_config"], indent=2),
        "```",
        "",
        "## Verification checks",
        "",
        "| check | result | detail |",
        "|---|---|---|",
    ]
    for name, check in checkpoint["checks"].items():
        detail = {k: v for k, v in check.items() if k != "ok"}
        rendered = json.dumps(detail)
        if len(rendered) > 160:
            rendered = rendered[:157] + "..."
        lines.append(f"| `{name}` | {'PASS' if check['ok'] else 'FAIL'} | `{rendered}` |")
    lines += [
        "",
        f"All checks passed: **{checkpoint['all_checks_passed']}**",
        "",
        "Optimizer state can be restored consistently: the checkpoint carries AdamW moments for "
        "every trainable tensor, a single shared Adam step counter, both decay/no-decay groups, "
        "the scheduler's step counters, and Torch/CUDA/Python/NumPy/loader RNG states plus the "
        "loader epoch state tensor. Experiment branches will resume from a **copy** of "
        "`step_008130.pt`; the original is never opened for writing.",
        "",
        "## Protected artifacts",
        "",
        "| kind | files |",
        "|---|---|",
    ]
    for kind, entries in inv.items():
        lines.append(f"| {kind} | {len(entries)} |")
    lines += [
        "",
        "Full per-file hashes are in `reports/phase12a_baseline.json`.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> int:
    inventory = collect()
    checkpoint = verify_checkpoint()
    payload = {
        "phase": "12A",
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inventory": inventory,
        "checkpoint": checkpoint,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    write_markdown(payload)

    total = sum(len(v) for v in inventory.values())
    print(f"hashed {total} protected files")
    for name, check in checkpoint["checks"].items():
        print(f"[{'PASS' if check['ok'] else 'FAIL'}] {name}")
    print(f"wrote {OUT_JSON.name} and {OUT_MD.name}")
    return 0 if checkpoint["all_checks_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
