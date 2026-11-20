# Phase 12B — Experiment Configuration Audit (Part A)

## 1. Base checkpoint
All four branches set `base_checkpoint: checkpoints/polycodegpt_first_pass/step_008130.pt`
with sha256 `5562acc2fd58837f9d4245042b7f0b10de65520aa2ba292999c80045470590bd`
and `base_checkpoint_step: 8130`. The original file is opened read-only; experiment
outputs go to separate checkpoint directories (see §7).

## 2. Optimizer state
Restored from the checkpoint via `optimizer.load_state_dict` (AdamW moments for every
trainable tensor, single step counter, decay + no-decay groups). `reset_optimizer_step: false`
in every config means the moments are carried over and only the learning-rate
schedule is reset. No config silently clears or reinitialises the optimizer.

## 3. Scheduler state — deliberate reset (documented)
The checkpoint's scheduler was at `last_step = 8130` on a `total_steps = 8129` curve,
i.e. the LR was at the cosine floor (~3e-5). Continuing from that would train at the
floor. Every branch therefore sets `reset_schedule: true` with a fresh
`total_steps = 8130`, `warmup_steps = 163`, `warmup_fraction = 0.02`,
`min_lr_ratio = 0.1`. The optimizer step counter is **not** reset
(`reset_optimizer_step: false`), so optimizer moments persist but the LR curve restarts
with a short warmup to the base rate. This is an explicit, documented override, not a
silent change.

## 4. Learning rate semantics
`learning_rate: 0.0003` is the **base/initial learning rate** of the new stage, applied
after the 163-step warmup. It is not the resumed LR (which would be ~3e-5) and not a
mid-run override of a continuing curve.

## 5. Scheduler behaviour during the new stage
Fresh cosine: linear warmup over steps 1..163 from ~1.8e-6 to 3e-4, then cosine decay
to 3e-5 at step 8130. `max_steps = 8130`, `additional_steps = 8130`, so each branch
runs exactly one more pass worth of updates.

## 6. Token budget comparability
All four branches: `token_budget_total = 130345937` (measured code-corpus total).
Control/A: 98,807,136 Python + 31,538,801 JavaScript. B/C: 65,172,968 / 65,172,968.
See `reports/phase12b_token_sampling.json`.

## 7. Optimizer updates / batch / clip comparability
All four: micro_batch_size 4, gradient_accumulation_steps 8, effective_batch_size 32,
tokens_per_step 16384, gradient_clip 1.0, context_length 512. Comparable.

## 8. Seeds and determinism
All four: seed 42, `deterministic: true`. Deterministic algorithm flags
(`torch.use_deterministic_algorithms`) are not expressed in the YAML and must be set by
the training entry point; this is an open item for the run script.

## Configuration gaps fixed in this phase
- Added `max_steps` (was missing; would have defaulted to the pilot value 200).
- Corrected `total_steps` 8129 -> 8130 and token budgets to measured values.
- Added `reset_schedule`, `reset_optimizer_step`, `deterministic`, `additional_steps`.

## Unresolved
- Deterministic-algorithm flags are not in the YAML; must be enforced by the runner.
- `complete_pass` semantics for a resumed partial final update are inherited from the
  original run and should be re-confirmed at launch.
