"""Phase 12B Part B — validate an experiment YAML and print effective settings.

Does not load training data for optimization and does not touch the checkpoint.

    python experiments/phase12b/validate_config.py configs/experiments/control.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CHECKPOINT = ROOT / "checkpoints" / "polycodegpt_first_pass" / "step_008130.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    args = ap.parse_args()
    path = Path(args.config)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))

    exp = cfg["experiment"]
    data = cfg["data"]
    tr = cfg["training"]
    sched = cfg["schedule"]
    ckpt = cfg["checkpointing"]

    print(f"config: {path}")
    print(f"experiment id: {exp['id']}  status: {exp['status']}")
    print(f"base checkpoint: {exp['base_checkpoint']} (step {exp['base_checkpoint_step']})")
    print(f"checkpoint exists: {CHECKPOINT.is_file()}")
    print()
    print("effective settings:")
    print(f"  max_steps (absolute): {tr['max_steps']}  (additional updates: {tr['additional_steps']})")
    print(f"  total_steps (schedule): {sched['total_steps']}")
    print(f"  reset_schedule: {sched['reset_schedule']}  reset_optimizer_step: {sched['reset_optimizer_step']}")
    print(f"  lr base: {tr['learning_rate']}  warmup_steps: {sched['warmup_steps']}  min_lr_ratio: {sched['min_lr_ratio']}")
    print(f"  micro_batch: {tr['micro_batch_size']}  grad_accum: {tr['gradient_accumulation_steps']}  "
          f"effective: {tr['effective_batch_size']}  tokens/step: {tr['tokens_per_step']}")
    print(f"  grad_clip: {tr['gradient_clip']}  context: {tr['context_length']}  seed: {tr['seed']}")
    print(f"  deterministic: {tr['deterministic']}  algorithms: {tr.get('deterministic_algorithms')}")
    print(
        "  seeds: python={} torch={} cuda={} dataloader_worker={} workers={}".format(
            tr.get("python_seed"), tr.get("torch_seed"), tr.get("cuda_seed"),
            tr.get("dataloader_worker_seed"), tr.get("dataloader_num_workers"),
        )
    )
    print(
        "  cudnn: deterministic={} benchmark={}".format(
            tr.get("cudnn_deterministic"), tr.get("cudnn_benchmark")
        )
    )
    print()
    print("budget:")
    print(f"  total: {data['token_budget_total']}  (code {data['code_tokens']} + english docs {data['english_doc_tokens']})")
    if "python" in data.get("token_budget_per_language", {}):
        print(f"  python: {data['token_budget_per_language']['python']}  javascript: {data['token_budget_per_language']['javascript']}")
    print()
    print("checkpointing:", ckpt)
    print()
    # validations
    issues = []
    if tr["additional_steps"] != sched["total_steps"]:
        issues.append("additional_steps != total_steps (schedule length)")
    if tr["max_steps"] != 8130 + tr["additional_steps"]:
        issues.append("max_steps must equal 8130 + additional_steps (absolute final step)")
    if tr["additional_steps"] > sched["total_steps"]:
        issues.append("additional_steps exceeds the schedule length (max_steps > total_steps)")
    if round(sched["total_steps"] * sched["warmup_fraction"]) != sched["warmup_steps"]:
        issues.append("warmup_steps does not match warmup_fraction x total_steps")
    if tr["effective_batch_size"] != tr["micro_batch_size"] * tr["gradient_accumulation_steps"]:
        issues.append("effective_batch_size mismatch")
    if tr["tokens_per_step"] != tr["effective_batch_size"] * tr["context_length"]:
        issues.append("tokens_per_step mismatch")
    required_seed_keys = ("python_seed", "torch_seed", "cuda_seed", "dataloader_worker_seed")
    missing_seeds = [k for k in required_seed_keys if k not in tr]
    if missing_seeds:
        issues.append(f"missing deterministic seed keys: {missing_seeds}")
    for key in ("deterministic_algorithms", "cudnn_deterministic", "cudnn_benchmark",
                "dataloader_num_workers"):
        if key not in tr:
            issues.append(f"missing deterministic key: {key}")
    print("validation:", "OK" if not issues else "ISSUES: " + "; ".join(issues))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
