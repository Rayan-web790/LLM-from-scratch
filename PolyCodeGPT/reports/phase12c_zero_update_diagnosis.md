# Phase 12C — Zero-Update Control Run: Diagnosis and Fix

Status: **root cause found, fixed, and the production `--train` path is now proven to
execute optimizer updates** (3-update regression run through `main()` itself).
No full training run was launched; the checkpoint is unchanged.

## 1. Root cause (proven, not assumed)

The loop was never entered, and the guard that ended the run was not the step
comparison — it was the **complete-pass loader accounting**.

`checkpoints/polycodegpt_first_pass/step_008130.pt` stores
`loader_batches_consumed = 65036` and `loader_epoch_state` for the phase-1 run, where
the loader had 65,036 batches. `Trainer.restore()` replayed both:

```python
self._epoch_start_state = payload.get("loader_epoch_state")
self._epoch_batches_consumed = int(payload.get("loader_batches_consumed", 0))
self._pending_skip = self._epoch_batches_consumed          # 65036
```

`Trainer.run()` in `complete_pass` mode then computed:

```python
already_consumed = self._pending_skip                      # 65036
remaining_batches = len(self.train_loader) - already_consumed
```

The continuation stage trains on a **new** dataset (`control/train.bin`), but its
loader also exposes 65,036 batches. So `remaining_batches == 0`, the loop condition
`remaining_batches <= 0` fired on the very first iteration **before** any batch was
pulled, `while self.stage_step < self.additional_steps` was exited, and `run()`
returned with `stage_step = 0`.

Evidence from the failed run, exactly as reported:

| field | observed | meaning |
|---|---|---|
| `stage_step` / `additional_updates` | 0 | no `train_step()` call ever happened |
| `steps` | 8130 | global counter untouched (`initial_global_step + 0`) |
| `elapsed_seconds` | 0.00268 | no compute |
| `micro_batches` / `sequences` | 65,036 / 260,142 | restored phase-1 counters, unchanged |
| `tokens_seen` | 133,192,704 | restored cumulative count, unchanged |
| `passes` | 1 | the `complete_pass` block ran its `self.passes += 1` after the empty loop |
| `best_val_loss` | 2.39286 | restored from the checkpoint, not computed |

So the returned JSON was **newly constructed from `run()` but populated entirely by
restored checkpoint metadata** — it was not recomputed from this session's work.

Two secondary defects followed from the same root:

1. `complete_pass` mode raises
   `"complete-pass run stopped with N batches unconsumed"` if the loop ends early.
   Here `remaining_batches` was already 0 so that check passed silently — a zero-update
   run looks successful.
2. With a stale count *smaller* than the new loader (the general case), the run would
   silently skip that many batches of the new dataset before training anything.

### Why `--smoke` did not catch this

`--smoke` builds with `complete_pass=False` (one update cannot finish a pass), so
`run()` takes `remaining_batches = None` and never consults `_pending_skip`. The
stale count was invisible to it. `--train` uses `complete_pass=True`, where that same
count is a termination condition.

### Termination and stage values (verified by inspection + tests)

| value passed to `TrainSettings` | `--smoke` | `--train` |
|---|---:|---:|
| `initial_global_step` | 8130 | 8130 |
| `max_steps` | 1 | 8130 |
| `total_steps` (schedule) | 8130 | 8130 |
| `additional_steps` | 1 | 8130 |
| `complete_pass` | False | True |
| `stage_step` after restore | 0 | 0 |
| `checkpoint_dir` | None | `experiments/phase12b/runs/<branch>` |

`max_steps <= total_steps` held in both cases (8130 ≤ 8130 for `--train`), and the loop
condition `stage_step < additional_steps` (0 < 8130) was **true** — the loop was
entered and immediately `break`ed by `remaining_batches <= 0`. The legacy
"global step vs stage limit" comparison was *not* the cause.

## 2. Fix (smallest correct change)

`src/polycodegpt/training/engine.py` — `Trainer.restore(payload, *, reset_loader_state: bool = False)`:

```python
if reset_loader_state:
    self._epoch_start_state = None
    self._epoch_batches_consumed = 0
    self._pending_skip = 0
else:
    ... # resume behaviour unchanged
```

`scripts/run_phase12_experiment.py`:
* `build_trainer(..., reset_loader_state: bool = False)` forwards the flag to `restore()`;
* the `--train` branch calls `build_trainer(..., reset_loader_state=True)`;
* the `--smoke` branch does the same (its behaviour is unchanged in substance).

Why a *new dataset* must reset the flag: a continuation stage is a new one-pass
epoch over a different corpus. The checkpoint's "batches consumed" describes the
phase-1 corpus, so replaying it against the new loader is meaningless and unsafe.
Genuine resume of an interrupted stage still uses the stored value when the flag is
left `False` (existing resume tests still pass).

Preserved semantics, all verified:

* `initial_global_step = 8130`, `stage_step` starts at 0, `additional_steps = 8130`
* 65,036 batches / 260,142 blocks → 8,129 full updates + one final partial group of 4
  batches (14 sequences) = **exactly 8,130 updates**
* final `step_index = 16,260`, final `stage_step = 8,130`
* fresh stage scheduler (163 warmup, 8,130 cosine) advances 8,130 times
* AdamW moments and per-parameter `step` counters preserved (still restored, now 8131+)

No model architecture, data budget, filter, or experiment YAML change was required.

## 3. Files changed

| file | change |
|---|---|
| `src/polycodegpt/training/engine.py` | `restore(..., reset_loader_state=False)`; loader resume state cleared on continuation stages |
| `scripts/run_phase12_experiment.py` | `build_trainer(..., reset_loader_state=...)`; `--train` and `--smoke` pass `True`; smoke's "no checkpoint written" check now compares the output directory's file list (a pre-existing `run_summary.json` is evidence, not a failure) |
| `tests/unit/test_continuation.py` | new engine-level regression test |
| `tests/unit/test_experiment_runner.py` | new end-to-end `main()` `--train` regression + loader-reset wiring test |

## 4. Tests added and results

1. `test_reset_loader_state_prevents_zero_update_stage` (engine level):
   with the real checkpoint's counters, `restore()` **without** the flag aborts
   (`"batches were consumed"`); **with** the flag it performs exactly 4 updates over a
   4-block-per-update dataset and ends at global 8134 with `scheduler.last_step == 4`.
2. `test_production_train_command_performs_updates_and_reports_them` (production path):
   runs `runner.main()` with the real CLI arguments
   `--train --confirm-training`, a 3-update stage, the real checkpoint payload (real
   counters, tiny-model optimizer tensors) and synthetic blocks. It asserts the
   command passes `reset_loader_state=True`, that 3 updates actually happened, and that
   the written `run_summary.json` reports `stage_step=3`, `additional_updates=3`,
   `steps=8133`, `skipped_steps=0`, `elapsed_seconds>0`, `micro_batches=65036+24`,
   `sequences=260142+96`, `tokens_seen=133193472`.
3. `test_train_path_resets_loader_state_before_running` — `_pending_skip == 0`,
   `_epoch_batches_consumed == 0`, `complete_pass is True`, `additional_steps == 8130`.

```
"C:\Users\UltraPc\Desktop\LLM FROM SCRATCH\.venv\Scripts\python.exe" -m pytest tests/unit -q
295 passed in 52.00s
```

Focused run:

```
... -m pytest tests/unit/test_continuation.py tests/unit/test_training.py \
        tests/unit/test_experiment_runner.py tests/unit/test_experiment_builder.py -q
62 passed in 44.00s
```

## 5. Evidence from the failed attempt (preserved, not deleted)

`experiments/phase12b/runs/control/` — the only file written by the failed attempt:

```
experiments/phase12b/runs/control/run_summary.json   520 bytes
checkpoints in run dir: []            # no .pt was written
```

No checkpoint was saved because `due_checkpoint` requires either a configured
`checkpoint_steps` hit, an `eval_every` hit, or `stage_step == additional_steps`
(0 == 8130 is false), and no validation ran (`eval_steps: ()`, and a zero-update loop
never reaches the final-step branch).

The file is copied unchanged to `reports/phase12b_failed_zero_update_run/run_summary.json`.
Because the corrected run writes to the same path, that copy is the durable record —
treat it as read-only.

## 6. Checkpoint integrity

```
before/after this session: 5562acc2fd58837f9d4245042b7f0b10de65520aa2ba292999c80045470590bd
```

Matches the required hash. No checkpoint file was modified.

## 7. Corrected command for the control run

```
"C:\Users\UltraPc\Desktop\LLM FROM SCRATCH\.venv\Scripts\python.exe" scripts\run_phase12_experiment.py --experiment control --train --confirm-training
```

Before starting it, verify the first few logged lines show
`training from global step 8130 (stage step 0) to 16260` **followed by a
`step 8131 | loss ...` line within seconds**. The run is expected to end with
`stage_step: 8130`, `steps: 16260`, `additional_updates: 8130`.

## 8. Is the production training path proven to execute updates?

**Yes, for the update mechanics and the stop condition**: the real `main()` `--train`
branch now performs optimizer updates, advances global and stage counters, advances
the fresh scheduler, and reports those new numbers in `run_summary.json`. Proven with
mocked heavy components (tiny model, synthetic blocks) on CPU — 3 of 3 updates, exact
counters, no stale metadata.

**Not yet proven**: the full 8,130-update CUDA run has not been executed. Its
throughput, wall-clock and memory behaviour remain unmeasured, and Phase 12C training
is therefore **not** claimed as successful.
