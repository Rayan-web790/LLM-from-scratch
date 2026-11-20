# Phase 12C — Control Run Audit (read-only)

Scope: verify the completed control continuation (8,130 updates, global step 8130 → 16260)
without modifying any checkpoint, dataset, or run artifact. No training launched.

## 1. Final partial update and checkpoint ordering — CODE TRACE

Source: `src/polycodegpt/training/engine.py`, `Trainer.run()` and `Trainer.train_step()`.

Call sequence inside one successful iteration of `while self.stage_step < self.additional_steps`:

1. `result = self.train_step(batches)` — runs the forward/backward over the micro-batch
   group, calls `optimizer.step()`, then `scheduler.step()`, then
   `self.stage_step += 1` and `self.step_index = self.initial_global_step + self.stage_step`.
   **The optimizer update and both counter increments happen here, before `run()` regains
   control.** For the final partial group (`result["partial"] is True`) this is the
   14-sequence / 4-micro-batch update.
2. `due_eval` is computed using the **already-advanced** `self.step_index`
   (`self.stage_step == self.additional_steps` is True on the final iteration).
3. `self.validate()` runs at the new step (model already updated) → recorded as
   `validation_history[-1]["step"] == 16260`.
4. `due_checkpoint` is computed using the advanced step; `self.save(...)` writes
   `step_016260.pt` and `latest.pt` from the already-updated model/optimizer.
5. Only then is `result["partial"]` logged, producing the console line
   `step 16260 | final partial update over 14 sequences ...` **after** the checkpoint
   line. The log label ordering is therefore misleading; the update precedes it.

So the console sequence

```
step 16260 | val loss 2.2290
step 16260 | checkpoint step_016260.pt
step 16260 | final partial update over 14 sequences / 7,168 tokens (4 micro-batches)
step 16260 | loss 2.3674 ...
```

does **not** mean the checkpoint was saved before the update. The update occurred in
step 1 of that iteration; validation and the checkpoint were steps 3–4.

## 2. Is the final partial update in `step_016260.pt`?

Yes. Verified by loading the checkpoint (read-only):

| field | value |
|---|---:|
| `step` | 16260 |
| `stage_step` | 8130 |
| `initial_global_step` | 8130 |
| `tokens_seen` | 266,385,408 |
| `micro_batches` | 130,072 (65,036 phase-1 + 65,036 stage) |
| `sequences` | 520,284 (260,142 + 260,142) |
| `best_val_loss` / `best_step` | 2.228988948278129 / 16260 |
| optimizer param `step` (all 100 params) | 16260 |
| scheduler `base_lrs` | [0.0003, 0.0003] |
| param-group `lr` | 3.0e-05 (end of cosine, 0.1× base) |

`step == 16260 == initial_global_step + 8130`, and every optimizer state entry carries
`step = 16260`, i.e. all 8,130 stage updates (including the final 14-sequence partial
group) were applied and persisted. Model weights, optimizer moments, scheduler and
scaler state in the file are post-update.

One benign artifact: `passes = 0` in the checkpoint versus `passes = 1` in
`run_summary.json`. `self.passes += 1` runs in the post-loop `complete_pass` block,
which executes **after** the final in-loop checkpoint save. The weights/optimizer are
unaffected; only this bookkeeping counter differs. No change made.

## 3. Throughput accounting — corrected

Bug: `run()` reported `tokens_per_second = self.tokens_seen / elapsed`, using the
cumulative `tokens_seen` (phase-1 133,192,704 + stage 133,192,704) over the
continuation-only elapsed time, double-counting the original phase.

Fix applied in `src/polycodegpt/training/engine.py`:
- added `self.initial_tokens_seen`, captured at `restore()` (falls back to the
  restored `tokens_seen` for checkpoints written before this field existed);
- `checkpoint_state()` persists `initial_tokens_seen`;
- `run()` now reports
  - `tokens_seen` (cumulative, unchanged),
  - `initial_tokens_seen`,
  - `stage_tokens_seen = tokens_seen - initial_tokens_seen`,
  - `tokens_per_second = stage_tokens_seen / elapsed`.

Corrected control numbers:

| metric | value |
|---|---:|
| cumulative `tokens_seen` | 266,385,408 |
| stage `tokens_seen` | 133,192,704 |
| `elapsed_seconds` | 3,749.312573 |
| corrected `tokens_per_second` | 35,524.57 |
| previously reported | 71,049.13 (2×) |

`run_summary.json` on disk still contains the pre-fix value 71,049.13 (the run
finished before the fix). It is evidence and is not overwritten.

## 4. Read-only verification

- `step_016260.pt` exists: `experiments/phase12b/runs/control/step_016260.pt`
  (207,215,053 bytes).
- `latest.pt` and `best.pt` also present in the same directory.
- Protected starting checkpoint SHA256 unchanged:
  `5562acc2fd58837f9d4245042b7f0b10de65520aa2ba292999c80045470590bd`
  (`checkpoints/polycodegpt_first_pass/step_008130.pt`).
- Final checkpoint SHA256: `81a02873fa2f52838ecaf6ca141f9b665f5941d7ee67f325b34d16cb7cb2f1f9`.
- Tokenizer SHA256 unchanged between start and final checkpoints:
  `0459f8556a6f79ca1ba1c3dc3f76c26ae2e55653ebbf97beaa9bd23467358a03`.
- Model config / parameter count unchanged (16,988,160 params, 6-layer/384-dim/16k vocab).
- Data manifest SHA256 unchanged: `367636ca923c8e9b6d0693a7606ae06504e76067908c686bcbf0e746508f5544`.
- Control dataset manifest: `data/processed/phase12b_experiments/control/dataset_manifest.json`
  SHA256 `01bc3e05c76de253c9008a3eaa32152722b2aeae97f69aebe8f8a1bf32c65261`,
  `train.bin` SHA256 `6b8ffbf653e8a3b3dbe5c641463dd78366bdd3e4387758fbc98866a6aba6e021`,
  133,192,957 tokens, 260,142 blocks, 8,129 full + 14-sequence final partial = 8,130 updates,
  65,036 loader batches.
- Failed zero-update evidence preserved:
  `reports/phase12b_failed_zero_update_run/run_summary.json` (copy of the original
  `experiments/.../control/run_summary.json` from the 0-update attempt); the live
  `experiments/phase12b/runs/control/run_summary.json` now holds the completed run.
- Held-out test split and `independent_v3` were not accessed.

## 5. Tests added

- `tests/unit/test_training.py::test_final_partial_update_is_applied_before_validation_and_checkpoint`
  — 7-block stream → 1 full + 1 partial (3-seq) update; asserts step/stage advance to 2,
  final validation recorded at step 2, `step_000002.pt` exists, optimizer state
  `step == 2`, and checkpoint weights equal the post-run model.
- `tests/unit/test_training.py::test_throughput_uses_stage_delta_not_cumulative_tokens`
  — sets a restored baseline of 50 tokens, runs, asserts `stage_tokens_seen`,
  `tokens_per_second == stage_tokens_seen / elapsed`, and cumulative `tokens_seen`
  preserved.

Result: 297 unit tests pass (`python -m pytest tests/unit -q`).

## 6. Is control ready to serve as the experiment baseline?

Yes, for the model/training baseline:
- 8,130/8,130 updates completed, `skipped_steps = 0`, final global step 16260 exactly.
- Final partial update applied and persisted in `step_016260.pt` (post-update weights +
  optimizer + scheduler + scaler).
- Fresh 163-step warmup cosine schedule consumed fully (base lr 3e-4 → 3e-5).
- Validation loss improved to 2.22899 (best at step 16260).
- Determinism config and seeds intact; checkpoint/tokenizer/manifest hashes stable.

Caveats:
- `passes = 0` in `step_016260.pt` vs `passes = 1` in the summary (bookkeeping only;
  weights unaffected). If a downstream consumer reads `passes` from the checkpoint,
  treat it as 1.
- `run_summary.json` on disk reports the pre-fix throughput (71,049.13); the corrected
  value is 35,524.57. No re-run was performed to regenerate it (per no-overwrite
  evidence policy).
- Full-run wall-clock and VRAM were not re-measured here; the existing preflight
  record (7.064 s / 1 update on CUDA) stands for the smoke path only.

No experiment (exp_a/b/c) has been started.
PYEOF
echo done