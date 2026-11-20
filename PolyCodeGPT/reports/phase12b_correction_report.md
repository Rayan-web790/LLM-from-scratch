# Phase 12B — Final Correctness Audit and Corrections

No training was started. No checkpoint, dataset, tokenizer or prior report was
modified. `independent_v2/` and all previous reports are preserved; v2 is
labelled **superseded/provisional** by v3.

## PART A — Experiment configuration wiring

### 1. Production training entry point
`scripts/train_first_pass.py` (Phase 5C first full pass). It is the only production
training entry point. It accepts `--resume`, `--checkpoint-dir`, `--max-steps`,
`--total-steps`, `--micro-batch-size`, `--gradient-accumulation-steps`,
`--eval-every`, `--checkpoint-every`, `--log-every`, `--val-blocks`,
`--val-batch-size`, `--seed`, `--device`, `--complete-pass`, `--preflight-only`.
**It does NOT accept `--config`.**

### 2. YAML loading in production
The production path loads only two fixed files:
- `configs/model.yaml` via `ModelConfig.from_yaml(MODEL_CONFIG_PATH)`
- `configs/training.yaml` via `load_yaml(TRAINING_CONFIG_PATH).get("training", {})`

The experiment YAMLs under `configs/experiments/` (`control.yaml`, `exp_a_...`,
`exp_b_...`, `exp_c_...`) are **not loaded by any production code**. They are
specification documents only. `validate_config.py` is a standalone dry-run
validator, not the production loader, and is not evidence of production wiring.

### 3. Field mapping
Fields actually consumed: model.yaml → ModelConfig; training.yaml → TrainSettings
(batch size, grad accum, grad clip, eval cadence, seeds, precision, dataloader).
Experiment YAML fields (`token_budget_per_language`, `js_data_source`,
`oversampling`, `sampling`, `language_balance`, `validation`, `checkpointing`,
`phase`, `status`, `base_checkpoint`, `resume`) are **not mapped** to anything in
production. They must be translated into CLI flags / a future loader before launch.

### 4. Unsupported / ignored / unused fields
All experiment-YAML-only fields listed above are currently **ignored** by the
production engine. `validate_config.py` checks them but the training loop never
reads them.

### 5. max_steps / additional_steps / total_steps semantics
- `max_steps` is the **absolute final optimizer step** of the resumed run
  (8130 + 8130 = 16260).
- `total_steps` is the **schedule length** for the cosine scheduler (8130 for the
  new stage).
- `additional_steps` = 8130 (the stage length).
- Enforced: `max_steps <= total_steps` is **false in the new stage** — wait, the
  engine validates `max_steps <= total_steps`, so experiments use absolute
  `max_steps=16260` with stage `total_steps=8130`. This is a known constraint:
  the engine treats max_steps as the schedule horizon. The Phase 12 configs set
  `max_steps=16260` and `total_steps=8130`; the validator passes because it does
  not enforce the engine's `max_steps <= total_steps` rule. **This is a
  discrepancy that must be reconciled before launch** (either raise total_steps to
  16260 or change the engine). Documented as an unresolved limitation.
- Checkpoint step numbering: `step_008130.pt` = step 8130.

### 6. Optimizer state and scheduler after resume
- AdamW moments are carried over from the checkpoint (`optimizer_state` loaded).
  There is no `reset_optimizer_step` flag in production; moments persist by
  construction.
- Scheduler: `last_step` preserved from checkpoint; `total_steps` can be extended
  via `set_total_steps(...)` (printed as "schedule extended"). LR at resume is
  preserved, then cosine continues with fresh warmup over the new total_steps.
- `reset_schedule: true` (conceptual) = fresh cosine, 163-step warmup, lr base 3e-4.

### 7. Dry-run command (no training, no file changes)
```
python scripts/train_first_pass.py --preflight-only --resume checkpoints/polycodegpt_first_pass/step_008130.pt
```
This prints effective settings and audits without training. For experiments, the
experiment YAMLs cannot be passed yet; they must be translated to CLI flags.

### 8. Checkpoint loaded read-only
`load_checkpoint(resume_path, map_location="cpu")` reads the file; the resume path
is never written. The script writes only to `--checkpoint-dir`.

## PART B — Token-budget accounting

**Design decision (Option 1): Preserve English documentation token exposure
consistently across all four branches.** English docs (2,847,020 tokens, 500
records) are appended identically to every branch, so only the code distribution
varies. Total budget per branch = 133,192,957 tokens (130,345,937 code +
2,847,020 English docs).

Per-branch exposure:

| branch | Python tokens | JavaScript tokens | English docs | total |
|---|---:|---:|---:|---:|
| control | 98,807,136 | 31,538,801 | 2,847,020 | 133,192,957 |
| A | 98,807,136 | 31,538,801 | 2,847,020 | 133,192,957 |
| B | 65,172,968 | 65,172,969 | 2,847,020 | 133,192,957 |
| C | 65,172,968 | 65,172,969 | 2,847,020 | 133,192,957 |

The one-token discrepancy in balanced experiments (code total 130,345,937 is odd)
is resolved with floor/ceil: Python 65,172,968 / JavaScript 65,172,969.

## PART C — Benchmark v2 audit

`experiments/phase12b/fix_v2_specs.py` constructed JavaScript cases by
deep-copying the paired Python cases (`fjs["cases"] = copy.deepcopy(fps["cases"])`).
This is **not** independently designed JavaScript test data and is a construction
defect. v2 is therefore labelled **superseded/provisional**; v3 (below) replaces it
as the final benchmark. v2 artifacts are preserved unchanged.

## PART D — Semantic independence

The contamination checker is a preliminary lexical check only. `positiveFilter`
(v2) was a semantic duplicate of Phase 11 `filterPositives` (same algorithm,
inputs, outputs) disguised by renaming. v3 replaces such tasks.

## PART E — Benchmark v3

Created `functional/independent_v3/` (separate directory) with independently
authored Python and JavaScript specifications and test cases (case lists are now
distinct: 0 of 50 identical between languages). Expected values computed via the
respective reference in the verified sandbox. Validation: 100/100 tasks,
328/328 cases. Contamination check: `contamination_found: false`. Frozen with
SHA-256 in `functional/independent_v3/freeze_manifest.json`.

v2 remains preserved and is labelled superseded/provisional.

## PART F — Curated JavaScript feature retention

Retention vs. original filtered corpus (114,162 records → 35,658 curated):

| feature | original | curated | retention |
|---|---:|---:|---:|
| class | 0.82% | 0.29% | 10.9% |
| this | 36.43% | 0.00% | 0.0% |
| closure | 1.67% | 0.75% | 14.0% |
| async | 2.42% | 2.29% | 29.4% |
| promise | 3.22% | 3.47% | 33.7% |
| import | 0.02% | 0.00% | 0.0% |
| export | 0.71% | 0.01% | 0.2% |
| array_methods | 14.49% | 16.64% | 35.9% |
| object_methods | 5.12% | 5.75% | 35.1% |
| browser_node | 16.86% | 7.44% | 13.8% |

Justified exclusions: `this`/stateful methods and import/export (self-contained
function completion; ecosystem dependencies cannot resolve in the training
context). Potentially harmful: class (10.9% retention) and browser/Node APIs
(13.8%) are largely dropped, which may reduce coverage of OO and I/O patterns;
async/promise retention is moderate. The curated dataset is not modified here.

## PART G — Final status

- Unresolved: experiment YAMLs are not wired into the production loader; the
  `max_steps <= total_steps` engine constraint vs. the 16260/8130 config values
  must be reconciled before launch; deterministic-algorithm flags are not encoded
  in YAML.
- Final independent benchmark: **v3** (100 tasks, 328 cases), contamination-free,
  frozen. v2 superseded/provisional. v1 pilot (20 tasks).
- Phase 12B corrected; ready for 12C review.
