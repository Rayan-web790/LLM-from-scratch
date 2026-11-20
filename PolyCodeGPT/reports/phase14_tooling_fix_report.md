# Phase 14 — Independent evaluation tooling fix report

Minimal compatibility changes only. No training, checkpoint modification, independent-suite modification, model generation, or finalist evaluation was performed.

## 1. Exact files changed

- `scripts/evaluate_phase10_benchmark.py`
  - Added `--task-manifest` / `--task_manifest` CLI option.
  - Added `load_task_manifest()` and `_phase10_generation_tasks()`.
  - `run_benchmark()` now accepts `task_manifest` and `task_manifest_path`.
  - Added task suite, manifest path/hash, task IDs, checkpoint SHA256, and tokenizer SHA256 to generated JSON.
  - Preserved the original Phase 10 task loading when no manifest is supplied.
- `functional/evaluate_functional.py`
  - `load_manifest()` now accepts an optional manifest path.
  - Added `--manifest` CLI option.
  - Added `build_candidate_indexes()` and `resolve_candidate()` (with `find_candidate` / `match_candidate` aliases).
  - Candidate matching now supports stable independent task IDs and validates language, docstring, and code-stub identity.
  - Added manifest path and SHA256 to the functional JSON payload.
- `tests/unit/test_phase14_tooling.py`
  - New focused compatibility tests.

No checkpoint, tokenizer, dataset, manifest, reference solution, freeze manifest, or existing report was modified.

## 2. CLI options added

Generation benchmark:
- `--task-manifest` / `--task_manifest PATH`
  - Default: omitted, which preserves the historical Phase 10 100-task inventory.

Functional evaluator:
- `--manifest PATH`
  - Default: omitted, which preserves `functional/manifest.json`.

## 3. Default Phase 10 / Phase 11 behavior preserved

- Omitting `--task-manifest` still loads the original 50 Python + 50 JavaScript Phase 10 tasks.
- Omitting `--manifest` still loads `functional/manifest.json` (100 tasks, 420 cases).
- Existing JSON fields remain present; only additive compatibility fields were added.
- Existing tests continue to pass: 308/308 unit tests passed.

## 4. Independent schema mapping

Independent v3 task fields map as follows:
- `task_id` -> generation `task_id` and functional task identity.
- `language` -> prompt language and syntax validator selection.
- `docstring` -> generation prefix docstring and functional candidate key.
- `signature` -> code stub / completion prefix and functional candidate key.
- `entry` -> sandbox entry point.
- `cases` -> functional test cases.
- `behavior`, `resolution`, `convention_note`, `params`, `params_with_defaults` -> preserved unchanged in the manifest.

No task data was translated or dropped. The functional evaluator matches candidates by stable task ID when present, with fallback to `(language, docstring, signature)` for older reports.

## 5. Tests run and results

Command:
```
C:\Users\UltraPc\Desktop\LLM FROM SCRATCH\.venv\Scripts\python.exe -m pytest tests/unit -q
```

Result: **308 passed**.

New tests cover:
- Default generation task set remains 100 Phase 10 tasks.
- Independent manifest loads 100 tasks with 50 per language.
- Default functional manifest remains 100 tasks / 420 cases.
- Independent functional manifest loads 100 tasks / 328 cases (50 tasks and 164 cases per language).
- No implicit fallback from independent_v3 to Phase 11.
- Candidate matching by independent task ID.
- Missing, duplicate, unknown, and mismatched candidates fail safely.
- Sandbox configuration unchanged.

## 6. Independent reference validation

The updated functional evaluator loaded `functional/independent_v3/manifest.json` and evaluated reference solutions supplied through a temporary generation report (reference code only, no model generation).

Result:
- Python: 50/50 tasks, 164/164 cases.
- JavaScript: 50/50 tasks, 164/164 cases.
- Total: **100/100 tasks, 328/328 cases**.

This validates evaluator compatibility only; it is not a finalist model evaluation.

## 7. Sandbox configuration confirmation

Unchanged:
- Images: `polycodegpt-sandbox-python:v1`, `polycodegpt-sandbox-node:v1`.
- Memory: 128m, swap 128m.
- CPU: 0.50.
- Pids: 32.
- Nofile: 64, fsize: 1MB.
- Timeout: 10.0s.
- Network: none.
- Safety probes: 16/16 passed.

## 8. Finalist and frozen-suite integrity

- Exp B checkpoint SHA256: `fc70f5264a97d22c162986e172f8e70881713d5ae235c992e516a91ff67f5db6` (unchanged).
- Exp C checkpoint SHA256: `cf0544099d73f85cdeb40bc1e0a93ce17bb790db4cb7136254be65d16c3b45b4` (unchanged).
- Independent v3 freeze manifest: all 7 listed files match their recorded SHA256 values (unchanged).
- No finalist candidate generation or independent finalist evaluation was launched.

PHASE 14 TOOLING READY — PREFLIGHT AGAIN