# Phase 12B — Final Integration Verification

Status: **integration complete, training not started.**
No Phase 12C run has been launched. `checkpoints/polycodegpt_first_pass/step_008130.pt`
(`sha256 5562acc2…90bd`) is unchanged; the original datasets, the tokenizer and
independent benchmark v3 are unchanged; the old test split was never opened.

All commands below were run with the project interpreter:

```
"C:\Users\UltraPc\Desktop\LLM FROM SCRATCH\.venv\Scripts\python.exe"   # Python 3.11.9, torch 2.15.0+cu132
```

---

## 1. Scheduler and stopping semantics (Part A)

### Counters

`TrainSettings` now carries explicit continuation accounting, and `Trainer` derives
everything from it — there is no second notion of "step":

| field | value | meaning |
|---|---:|---|
| `initial_global_step` | 8,130 | global step restored from the checkpoint |
| `stage_step` | 0 → 8,130 | updates completed **in this continuation stage** |
| `additional_steps` | 8,130 | length of this stage |
| `max_steps` | 8,130 | engine loop bound, `max_steps <= total_steps` holds |
| `total_steps` (schedule) | 8,130 | cosine horizon, stage-local |
| `expected_final_global_step` | 16,260 | `initial_global_step + additional_steps` |

The experiment YAML's `max_steps: 16260` is the **final global step**; the runner
derives the stage length as `max_steps - base_step = 8130`, asserts it equals
`additional_steps` and `schedule.total_steps`, and passes 8,130 to the engine. The
engine's `max_steps <= total_steps` constraint is therefore satisfied by
construction, not bypassed (`tests/unit/test_experiment_runner.py::test_every_config_resolves_to_the_frozen_stage_plan`).

### Restore path (`src/polycodegpt/training/engine.py`)

```python
load_weights_into(self.model, payload["model_state"])
self.optimizer.load_state_dict(payload["optimizer_state"])      # AdamW moments + step counters
if self.settings.fresh_schedule:
    for group in self.optimizer.param_groups:                   # base rate reset to the stage rate
        group["lr"] = float(self.settings.learning_rate)
    self.scheduler = WarmupCosineScheduler(...)                 # stage step 0, 163 warmup, 8,130 cosine
else:
    self.scheduler.load_state_dict(payload["scheduler_state"])
...
if fresh_schedule and step_index != initial_global_step: raise ValueError(...)
stage_step = 0 if fresh_schedule else max(0, step_index - initial_global_step)
```

* AdamW moments (`exp_avg`, `exp_avg_sq`) and each parameter's internal `step`
  counter are restored verbatim — the continuation continues the *same* optimizer,
  it does not restart it.
* The scheduler is deliberately *not* restored: a fresh warmup+cosine over 8,130
  stage steps is installed, and the parameter-group learning rate is reset to the
  configured `3e-4` base. **This fixed a real bug**: `optimizer.load_state_dict()`
  also restores the checkpointed `lr` (≈3e-5, the nearly decayed phase-1 rate), so a
  scheduler built after the load would have used a decayed base. The reset is what
  makes the fresh schedule actually start at `3e-4 / 163 = 1.84e-6`.
* The run loop is `while self.stage_step < self.additional_steps`, and `train_step`
  advances `stage_step` and recomputes `step_index = initial_global_step + stage_step`.
  Evaluation/checkpoint "final step" conditions compare `stage_step`, so they fire at
  the stage boundary rather than at global step 8,130 (which is never reached again).

### Scheduler progression (measured)

| stage step | LR |
|---:|---:|
| 0 | 1.840491e-06 (= 3e-4 / 163) |
| 1 | 3.680982e-06 |
| 4 | 9.202454e-06 |
| 8,125 | 3.000017e-05 |
| 8,129 | 3.000000e-05 (= 3e-4 × 0.1) |

### Tests (`tests/unit/test_continuation.py`)

Uses a **temporary byte-copy** of the real checkpoint plus a tiny synthetic model and
synthetic token stream; the original file is re-hashed at the end of the run test.

| test | proves |
|---|---|
| `test_adamw_moments_and_step_counters_are_restored` | (1) moments survive, (2) `step` counters survive |
| `test_real_checkpoint_optimizer_counters_are_present` | real checkpoint stores non-zero moments and `step == 8130` for all 36 tensors |
| `test_continuation_starts_at_stage_zero_with_global_step_8130` | (3) scheduler at stage step 0, (4) global step 8130, warmup 3/6 → first LR `base/3` |
| `test_stage_local_scheduler_matches_production_schedule_shape` | production shape: 8,130 steps, 163 warmup, first LR `3e-4/163` |
| `test_exactly_additional_steps_updates_and_checkpoint_unchanged` | (5) exactly 6/6 updates in the 6-step proxy, (6) final global step 8136, (7) scheduler ends at stage 6, (8) checkpoint sha unchanged |
| `test_run_stops_at_stage_boundary_not_global_step` | loop stops on `stage_step`, not on the global step |
| `test_restore_rejects_mismatched_global_step` | a checkpoint that is not step 8130 is refused |
| `test_deterministic_settings_make_cpu_runs_bitwise_identical` | two seeded runs produce identical weights |
| `test_settings_validate_stage_constraints` | 8,130-stage settings are valid; `expected_final_global_step == 16260`; `warmup_steps == 163` |

```
9 passed
```

---

## 2. Materialized experiment datasets (Part B)

The four branches are no longer YAML budgets. `src/polycodegpt/data/experiment_builder.py`
(reused by `experiments/phase12b/build_dataset.py` and by the runner) tokenizes each
source once, records the exact token range of **every record**, then copies whole
records into one experiment-specific `train.bin` until the language budget is full.
Records are never split; the residual that does not fit is reported, never hidden.

Both code languages are read from `data/processed/polycodegpt_corpus/train.jsonl`
(the authoritative phase-1 training partition, `sha256 2a9c7160…5c74`). This mattered:
the per-language `codesearchnet_*_filtered/train.jsonl` files contain **44 extra Python
records** (67,996 characters) that the combined corpus excludes, so drawing from them
would have trained on records the split construction deliberately left out.

### Measured artifacts (`data/processed/phase12b_experiments/<branch>/train.bin`)

| branch | tokens | sha256 (prefix) | Python tokens / records / epochs | JS tokens / records / epochs | max repeats | blocks |
|---|---:|---|---|---|---:|---:|
| control | 133,192,957 | `6b8ffbf653e8` | 98,807,136 / 290,463 / 1.000 | 31,538,801 / 102,169 / 1.000 | 1 | 260,142 |
| exp_a | 133,192,950 | `37692f87f8c6` | 98,807,136 / 290,463 / 1.000 | 31,538,794 / 35,658 / 4.821 | 5 | 260,142 |
| exp_b | 133,192,915 | `eab16b96373c` | 65,172,938 / 191,985 / 0.661 | 65,172,957 / 102,169 / 2.065 | 4 | 260,142 |
| exp_c | 133,192,925 | `99c7077c570f` | 65,172,938 / 191,985 / 0.661 | 65,172,967 / 35,658 / 9.961 | 10 | 260,142 |

English documentation is identical in all four branches: **2,847,020 tokens,
500/500 documents, 1.000 epochs** (appended last, constant exposure).

**Control reproduces the phase-1 training stream byte for byte**:
`sha256 6b8ffbf653e8a3b3dbe5c641463dd78366bdd3e4387758fbc98866a6aba6e021` is exactly the
`train.bin` hash recorded in `data/processed/tokenized_v1_1/manifest.json`. The
re-materialized token count is 133,192,957 = 98,807,136 Python + 31,538,801 JavaScript
+ 2,847,020 English, i.e. the Part A reconciliation is now confirmed by construction.

Shortfalls vs. the declared budget are whole-record granularity only: control −0,
exp_a −7, exp_b −42, exp_c −32 tokens. Every branch still has 260,142 blocks.

### Loader verification (real `TokenBlockDataset` + `build_dataloader`)

For all four branches (`dataloader_verification` in each `dataset_manifest.json`):

* first batch `input_ids` shape `[4, 512]`, `targets` shape `[4, 512]` → 2,048 model
  input tokens and 2,048 target tokens per micro-batch;
* 32 sequences/update × 512 = **16,384 model input tokens and 16,384 target tokens per
  optimizer update**;
* 260,142 blocks → **8,129 full updates + one final partial update over 14 sequences**
  = **8,130 updates = `additional_steps` exactly**;
* 252/245/210/220 trailing tokens dropped by the last incomplete window.

That is the arithmetic the stage plan depends on: 8,130 updates consume the entire
one-pass stream, and the final update is a *partial* batch by construction (32 × 8,129
= 260,128 blocks; the remaining 14 blocks form update 8,130).

---

## 3. Training-runner integration (Part C)

`scripts/run_phase12_experiment.py` has three modes:

| command | effect |
|---|---|
| `--dry-run` | validates YAML + checkpoint + budget arithmetic, prints plan, writes nothing |
| `--prepare` (default non-dry-run) | materializes/verifies `train.bin`, writes `dataset_manifest.json`, prints `TRAINING NOT STARTED: data preparation only.` |
| `--train --confirm-training` | builds the **existing** `polycodegpt.training.Trainer` and runs the stage |

`build_trainer()` is thin wiring only — model reconstruction (`rebuild_model`),
optimizer, scheduler, restore, validation loader, metrics and the loop all come from
`polycodegpt.training`. No second training loop exists.

Code path:

```
configs/experiments/<branch>.yaml
  └─ run_phase12_experiment.resolve()            # invariants, stage plan, determinism
      ├─ prepare_dataset() → data/experiment_builder.py → data/processed/phase12b_experiments/<branch>/train.bin
      │                                        └─ verify_with_dataloader() → TokenBlockDataset + build_dataloader
      └─ build_trainer() → load_checkpoint(step_008130.pt) → rebuild_model()
                         → TrainSettings(initial_global_step=8130, additional_steps=8130,
                                         fresh_schedule=True, complete_pass=True, partial_final_update=True)
                         → Trainer(...).restore(payload)   # AdamW moments + step counters, fresh schedule
                         → Trainer.run()                   # only with --train --confirm-training
```

### Wired-but-not-run evidence (`tests/unit/test_experiment_runner.py`)

`test_build_trainer_uses_the_existing_engine_with_continuation_state` constructs the
real trainer on CPU against the prepared control dataset and asserts:

* class is `polycodegpt.training.engine.Trainer`, `complete_pass` and
  `partial_final_update` are on;
* `step_index == 8130`, `stage_step == 0`, `additional_steps == 8130`,
  `expected_final_global_step == 16260`;
* `scheduler.last_step == 0`, `total_steps == 8130`, `warmup_steps == 163`,
  first LR `= 3e-4/163`;
* AdamW state has `step == 8130` for every parameter, with non-zero moments;
* loader dataset is the prepared `train.bin`, 260,142 blocks = 8,129 full + 14;
* weights equal the checkpoint tensors;
* **no checkpoint file is written** by construction.

### Fail-closed behaviour

| guard | test |
|---|---|
| `--train` without `--confirm-training` exits | `test_train_without_confirmation_exits` |
| training with an unprepared dataset exits | `test_unprepared_dataset_blocks_training` |
| output inside `tokenizer/`, `data/curated`, `data/processed/tokenized_v1_1`, the corpus dir, the work dir or the checkpoint path exits | `test_training_output_may_not_overlap_protected_paths` |
| prepared dataset must re-hash to its manifest and still yield 8,130 updates | `verify_prepared_dataset()` |
| `max_steps`/`total_steps` mismatch in a YAML exits | `test_max_steps_mismatch_is_rejected`, `test_schedule_length_mismatch_is_rejected` |

Training output goes only to `experiments/phase12b/runs/<branch>/`; that directory
does not exist yet.

---

## 4. Deterministic configuration (Part D)

Declared per branch in the `training:` block and validated by
`experiments/phase12b/validate_config.py`:

```yaml
  deterministic: true
  python_seed: 42
  torch_seed: 42
  cuda_seed: 42
  dataloader_worker_seed: 42
  dataloader_num_workers: 0
  deterministic_algorithms: true
  cudnn_deterministic: true
  cudnn_benchmark: false
```

`TrainSettings` carries the same fields and `Trainer.__init__` calls
`apply_determinism()`, which records what it applied:

* `random.seed`, `numpy.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`;
* `torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`;
* `torch.use_deterministic_algorithms(True, warn_only=True)` — **warn-only on purpose**:
  strict mode raises for CUDA ops without a deterministic kernel (the embedding
  backward uses atomic accumulation) and would abort a long run at an arbitrary step;
* `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `PYTHONHASHSEED=<python_seed>`;
* the training loader is seeded through its generator, which the engine saves and
  restores with the checkpoint, so the sampling order is reproducible.

Unavoidable nondeterminism, reported not hidden:

1. BF16/FP16 matmul reductions and embedding-gradient atomic accumulation are not
   bitwise reproducible on CUDA; identical seeds give statistically identical, not
   byte-identical, runs. (CPU FP32 runs *are* bitwise identical — asserted by
   `test_deterministic_settings_make_cpu_runs_bitwise_identical`.)
2. Changing `dataloader_num_workers` or the number of devices changes the sampling
   order, so `dataloader_num_workers` is pinned to 0 in the configs.
3. `PYTHONHASHSEED` only takes effect in a freshly started process; it is set for
   child processes, not retroactively for the current one.

---

## 5. Validation performed (Part E)

```
$ python -m pytest tests/unit/test_training.py tests/unit/test_continuation.py \
      tests/unit/test_data_pipeline.py tests/unit/test_experiment_builder.py \
      tests/unit/test_experiment_runner.py -q
96 passed in 21.44s

$ python -m pytest tests/unit -q
287 passed in 27.67s
```

New/changed test modules: `test_continuation.py` (9), `test_experiment_builder.py` (6),
`test_experiment_runner.py` (15); `test_training.py` (24) unchanged and still green.

```
$ for e in control exp_a exp_b exp_c; do python scripts/run_phase12_experiment.py --experiment $e --dry-run; done
```

Each branch: checkpoint sha verified, `initial step 8130`, `stage 8,130 updates
(max_steps <= total_steps: 8130 <= 8130)`, `expected final 16,260`, first LR
1.840491e-06 → 9.202454e-06 over the warmup, last LR 3.000000e-05, determinism
`seeds(py/torch/cuda/loader)=42/42/42/42 workers=0 cudnn(det/bench)=True/False`,
`DRY RUN: no files written, no training started.`

```
$ python experiments/phase12b/validate_config.py configs/experiments/<branch>.yaml
validation: OK          # all four branches, including the new determinism keys
```

```
$ python scripts/run_phase12_experiment.py --experiment control --prepare
materialized:  133,192,957 tokens  sha256=6b8ffbf653e8a3b3...
blocks/updates: 260,142 blocks -> 8,129 full + 14 sequences final partial = 8,130 updates
TRAINING NOT STARTED: data preparation only.
```

```
$ python functional/independent_v3/spec/build_manifest.py --check
validated 100 tasks, 328 cases        # benchmark v3 untouched
```

Integrity re-checked after all work:

| artifact | sha256 |
|---|---|
| `checkpoints/polycodegpt_first_pass/step_008130.pt` | `5562acc2…90bd` (unchanged) |
| `data/processed/tokenized_v1_1/train.bin` | `6b8ffbf6…e021` (unchanged) |
| `data/processed/polycodegpt_corpus/train.jsonl` | `2a9c7160…5c74` (unchanged) |
| `data/curated/js_v1/train.jsonl` | `b0ae31c0…9e6bd7` (unchanged) |
| `tokenizer/artifacts/polycodegpt_tokenizer_v1_1.json` | `0459f855…58a03` (unchanged) |

---

## 6. Exact future training commands (do not run yet)

```
"C:\Users\UltraPc\Desktop\LLM FROM SCRATCH\.venv\Scripts\python.exe" scripts/run_phase12_experiment.py --experiment control --prepare
"C:\Users\UltraPc\Desktop\LLM FROM SCRATCH\.venv\Scripts\python.exe" scripts/run_phase12_experiment.py --experiment control --train --confirm-training
```

…and identically for `exp_a`, `exp_b`, `exp_c`. `--prepare` is safe to run now and
rewrites only `data/processed/phase12b_experiments/<branch>/`; `--train` is gated
behind `--confirm-training` and has not been used.

---

## 7. Remaining issues / not claimed

* **No Phase 12C run was performed.** Throughput, memory and wall-clock for the
  8,130-update stage are still estimates, not measurements.
* Prepared artifacts total ≈1.02 GB under `data/processed/phase12b_experiments/`
  plus ≈300 MB of per-language tokenization cache under
  `data/processed/phase12b_work/` (`_python.bin`, `_javascript_original.bin`,
  `_javascript_curated.bin`, `_english_docs.bin` and their `.index.json`). The cache
  is derivable from the sources and can be deleted.
* Whole-record granularity means a branch's materialized total can sit a few dozen
  tokens under its declared budget (worst case exp_b, −42). Blocks and update counts
  are unaffected.
* `curated` JavaScript caps at 6,542,851 unique tokens (35,658 records), so exp_c
  repeats records up to 10 times. Whether that repetition is acceptable for the
  quality hypothesis is an evaluation decision, not a pipeline defect.
* Evaluation is not wired into the runner: `independent_v3` is frozen and must be run
  by the existing evaluator only after the four checkpoints exist.
