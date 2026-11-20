# Phase 12B — Resource and Execution Readiness (Part E, ESTIMATES)

All figures are estimates, not guarantees.

## GPU memory (per branch)
- 16.99 M parameters in bf16: ~34 MB weights.
- AdamW m/v states: ~68 MB.
- Activations at batch 32 x 512: roughly 1-2 GB.
- Estimated peak: 2-4 GB VRAM. A 12 GB card is comfortable; 8 GB is acceptable
  with gradient checkpointing (not currently enabled).

## Storage (all four branches)
- Checkpoint size ~200 MB each; 5 checkpoints per branch = ~1 GB per branch.
- Four branches = ~4 GB of checkpoints.
- Curated dataset + audit artifacts ~120 MB.
- Independent benchmark ~1 MB.
- Recommended free space: 10 GB.

## Compute (per branch)
- 8,130 steps x 16,384 tokens = 130.3 M tokens.
- Estimated wall-clock on a single consumer GPU: 0.5-2.0 h.
- All four branches sequential: 2-8 h. Parallel on 4 GPUs: 0.5-2.0 h.
- CPU-only: 24-72 h (not recommended).

## Required checkpoint backups
- Original `step_008130.pt` is never written; keep an offline copy before any run.
- Each branch writes to its own directory (`checkpoints/phase12b_<branch>/`) and
  never to `checkpoints/polycodegpt_first_pass/`.
- Back up the best checkpoint per branch (by validation loss) before evaluation.

## Recovery after interrupted training
- Restore from the last periodic checkpoint (every 1000 steps) using
  `trainer.restore(payload)`; `loader_epoch_state` and `_pending_skip` resume the
  data stream at the exact micro-batch.
- Re-run with `--resume <checkpoint>`; verify `step_index`, `tokens_seen` and
  `scheduler.last_step` before continuing.
- If the checkpoint is corrupt, fall back to the previous periodic checkpoint and
  replay from there (at most 1000 steps of recompute).

## Output paths (original checkpoint untouched)
- Branch checkpoints: `checkpoints/phase12b_control/`, `..._exp_a/`, `..._exp_b/`,
  `..._exp_c/`.
- Training logs: `reports/phase12b_<branch>_training.jsonl`.
- Evaluation outputs: `reports/phase12b_<branch>_dev_eval.json`.
- The independent benchmark is evaluated only in Phase 14 and is never touched
  during training or selection.
