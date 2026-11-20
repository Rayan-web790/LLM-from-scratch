# Phase 12A — Independent Final Functional Benchmark Design

## Purpose

A held-out functional benchmark that is **not** used for training selection or
tuning. It exists to give an unbiased estimate of whether a trained branch
solves real programming tasks. It is evaluated once per branch after training.

## Size and shape

- 100 tasks: 50 Python, 50 JavaScript.
- Same spec format as Phase 11 (`functional/spec/python_specs.py`,
  `functional/spec/javascript_specs.py`): signature, behaviour, multiple normal
  cases, boundary cases, expected outputs, error behaviour.
- Same sandbox harness (`functional/sandbox/`) and outcome taxonomy
  (`syntax_failure`, `missing_entry`, `runtime_exception`, `timeout`,
  `resource_limit_violation`, `incorrect_output`, `complete_success`).
- Success rule unchanged: all mandatory cases pass.

## Source of tasks

Tasks are authored from a held-out repository set that is **disjoint** from the
CodeSearchNet training corpus, plus a small number of hand-written algorithmic
tasks. They are never drawn from `data/processed/polycodegpt_corpus/*` or from
the Phase 11 task list. Each task records its provenance (repository + path, or
"hand-authored") so the contamination check below can be run.

## Contamination check (mandatory before scoring)

For each task, compute against the training corpus of the branch under test:

1. Normalised code hash of the reference solution vs every training record.
2. 5-gram token-shingle Jaccard vs every training record (threshold 0.8).
3. Normalised function-name overlap within the same language.
4. Source-repository overlap.

Any hit rejects the task; the rejection and the offending training record id are
logged in the benchmark manifest. A branch is scored only on tasks that pass.

## Disjointness from Phase 11

The Phase 11 task list (`functional/manifest.json`, 100 tasks) is treated as a
development benchmark. The independent benchmark must share no task id, no
function name, and no reference solution with it. A cross-check script compares
the two manifests and fails if any overlap is found.

## Evaluation procedure

1. Freeze the branch checkpoint (best step by validation loss).
2. Generate one completion per task with the Phase 10 generation harness
   (temperature/top-p settings identical to Phase 10).
3. Run `functional/evaluate_functional.py` unchanged, with the independent
   manifest swapped in via a `--manifest` flag (to be added when the benchmark
   is implemented).
4. Record the full per-task outcome JSON; do not aggregate away failures.

## Status

Design only. Implementation is deferred to Phase 12B. No code, data, or model
changes are made in Phase 12A for this benchmark.
