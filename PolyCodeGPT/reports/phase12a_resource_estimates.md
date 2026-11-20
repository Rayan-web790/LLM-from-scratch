# Phase 12A — Estimated Training Resource Requirements (ESTIMATES)

All figures below are estimates, not measurements. They assume a single
consumer-grade GPU (e.g. 12-24 GB VRAM), bf16 mixed precision, and the
hyperparameters recorded in `configs/experiments/*.yaml`. They are provided for
planning only; actual cost must be measured during a pilot run.

## Per-branch cost

| item | estimate | basis |
|---|---|---|
| parameters | 16.99 M | checkpoint `parameter_count` |
| tokens per branch | 133.2 M | one pass over the training split |
| optimizer steps | 8,129 | `tokens / 16,384` |
| VRAM (model + optimizer + activations) | 2-4 GB | 17 M params bf16 + AdamW m/v + batch 32x512 |
| wall-clock per branch (GPU) | 0.5-2.0 h | 17 M params is small; throughput depends on host |
| wall-clock per branch (CPU only) | 24-72 h | not recommended |
| checkpoint size | ~200 MB | observed `step_008130.pt` |
| checkpoints written per branch | 5 | steps 500/1000/2000/4000/8129 |
| disk per branch | ~1.0 GB | 5 checkpoints |

## All four branches

- Wall-clock (sequential, single GPU): 2-8 h.
- Wall-clock (parallel, 4 GPUs): 0.5-2.0 h.
- Disk: ~4 GB.
- No re-tokenization is required for branches A/B/C (they resample existing
  records); branch C reuses the curated set. The context-window proposal
  (1024 tokens) would require re-tokenization and is excluded from these
  estimates.

## Data preparation cost (already incurred in Phase 12A)

- JS audit + curation: 102,169 records scanned, 35,658 kept. Runtime on the
  machine used for this phase: a few minutes. Peak memory: a few hundred MB.
- No network access was used beyond the (already completed) Docker image pull
  reported in Phase 11.

## Caveats

- Throughput is dominated by data loading and the Python host; the estimates
  assume the existing `prepare.py` streaming pipeline is reused.
- Validation loss and the Phase 10/11 development benchmarks add roughly 10-20%
  to wall-clock per branch.
- These are estimates; treat them as planning inputs only.
