# Phase 14 — Final tooling corrections report

The two remaining blockers from `reports/phase14_second_preflight.md` are fixed. No finalist candidates were generated and no independent model evaluation was launched.

## 1. Exact files changed

### `scripts/evaluate_phase10_benchmark.py`

- Removed the module-level `from scripts.phase10_tasks import ...` dependency.
- Phase 10 task imports now occur only inside `_phase10_generation_tasks()`.
- Added `--json-out` and `--md-out` CLI options.
- Added output-path helpers:
  - `_resolve_output_path()` resolves relative paths against the project root.
  - `prepare_output_paths()` enforces explicit-output no-overwrite rules and creates parent directories.
  - `_open_output()` uses exclusive creation (`x`) for explicit outputs and normal overwrite (`w`) for legacy defaults.
- Explicit output paths are checked before model generation starts, so an existing target fails before any candidates are generated.
- Updated the `run_benchmark()` docstring to describe both the default Phase 10 suite and manifest-driven loading.

### `tests/unit/test_phase14_tooling_fix2.py` (new)

Focused tests for CLI options, explicit output paths, no-overwrite behavior, default compatibility, import isolation, and report provenance.

### `tests/unit/test_phase10_integration.py`

Updated test imports to get `python_tasks` and `javascript_tasks` directly from `scripts.phase10_tasks`, since the generation module no longer re-exports them.

No functional evaluator, manifest, checkpoint, tokenizer, sandbox, or existing report was modified.

## 2. New CLI syntax

Generation benchmark:

```
python scripts/evaluate_phase10_benchmark.py --json-out reports/phase14_exp_b_generation.json --md-out reports/phase14_exp_b_generation.md --task-manifest functional/independent_v3/manifest.json
```

Supported output arguments:

- `--json-out PATH`
- `--md-out PATH`

Both options accept relative or absolute paths. Relative paths are resolved against the project root.

### Omitted output options

When both are omitted, the historical Phase 10 behavior is preserved:

- JSON: `reports/phase10_generation_evaluation_<timestamp>.json`
- Markdown: `reports/phase10_evaluation_summary.md`
- The existing default overwrite behavior remains unchanged for backward compatibility.

### Explicit output options

When either option is supplied:

- The supplied path is honored.
- If only one is supplied, the other is derived as a sibling file with the matching extension.
- Both paths are resolved against the project root.
- Parent directories are created as needed.
- If either target already exists, the run fails before generation with a clear `refusing to overwrite existing ... output` message.
- Files are opened with exclusive creation, so an existing target cannot be overwritten even if it appears after the pre-check.
- JSON and Markdown cannot be pointed at the same path.

This means the intended finalist paths:

- `reports/phase14_exp_b_generation.json`
- `reports/phase14_exp_b_generation.md`
- `reports/phase14_exp_c_generation.json`
- `reports/phase14_exp_c_generation.md`

can be used without touching Phase 10, Phase 11, Phase 13, reference-validation, or pre-existing Phase 14 artifacts.

No such finalist output files were created during this fix.

## 3. No-overwrite proof

Tests prove:

- Existing JSON target is rejected and its contents are unchanged.
- Existing Markdown target is rejected and its contents are unchanged.
- A failed pre-check does not run generation.
- Parent directories for explicit outputs are created safely.
- Explicit JSON and Markdown paths are honored.
- Default Phase 10 names and legacy overwrite behavior are preserved when both options are omitted.

The actual write path also uses `open(..., "x")` for explicit outputs, so it fails rather than overwriting if a target appears after the pre-check.

## 4. Independent mode does not import `scripts.phase10_tasks`

Static inspection now shows only one reference to the task module in `scripts/evaluate_phase10_benchmark.py`:

```python
# inside _phase10_generation_tasks()
from scripts.phase10_tasks import python_tasks, javascript_tasks
```

There is no module-level import.

A subprocess test installs a `sys.meta_path` blocker that raises if `scripts.phase10_tasks` is imported, then imports the generation module and loads the independent manifest. The test proves:

- `load_task_manifest(functional/independent_v3/manifest.json)` succeeds.
- 100 tasks load.
- 50 Python and 50 JavaScript tasks load.
- All 100 tasks retain task IDs.
- `scripts.phase10_tasks` is not in `sys.modules`.

The default Phase 10 path is separately tested and still loads the original 100 tasks with the original 50/50 language split.

## 5. Report provenance preserved

The JSON report still includes:

- `checkpoint`
- `checkpoint_sha256`
- `checkpoint_step`
- `device`
- `seed`
- `temperature`
- `max_new_tokens`
- `tokenizer_version`
- `tokenizer_path`
- `tokenizer_sha256`
- `task_suite`
- `task_manifest`
- `task_manifest_path`
- `task_manifest_sha256`
- `task_manifest_hash`
- `task_ids`
- `prompts_generated`
- `python_tasks`
- `javascript_tasks`
- all syntax, EOS, repetition, and per-task metrics

Per-task entries retain `task_id`, `language`, `docstring`, `code_stub`, `generated_text`, `syntax_candidate`, `syntax_valid`, `eos_hit`, token metrics, and generated token count.

## 6. Verification results

### Full unit suite

Command:

```
C:\Users\UltraPc\Desktop\LLM FROM SCRATCH\.venv\Scripts\python.exe -m pytest tests/unit -q
```

Result: **319 passed** in 142.45s.

This is 308 previous tests plus 11 new Phase 14 tooling-correction tests.

### CLI help

Generation help confirms:

- `--task-manifest TASK_MANIFEST`
- `--json-out JSON_OUT`
- `--md-out MD_OUT`

Functional evaluator help remains:

- `--report REPORT`
- `--manifest MANIFEST`
- `--limit LIMIT`
- `--timeout TIMEOUT`
- `--json-out JSON_OUT`
- `--md-out MD_OUT`

### Independent suite

- Manifest SHA256: `2a7fbf3ddc43f6a2fee368d6faa711dbe8fa841343c7cf9507c919cd3baeb46c`
- Tasks: 100
- Python tasks: 50
- JavaScript tasks: 50
- Cases: 328
- Python cases: 164
- JavaScript cases: 164
- Freeze manifest SHA256: `ec36c8803f5c22692a9b789dda4315a4f1c9901e08dc80ac09e435d3a065997e`
- Frozen files checked: 7
- Hash mismatches: none

### Finalist checkpoints

- Exp B: `fc70f5264a97d22c162986e172f8e70881713d5ae235c992e516a91ff67f5db6`
- Exp C: `cf0544099d73f85cdeb40bc1e0a93ce17bb790db4cb7136254be65d16c3b45b4`

Both match the expected hashes.

### Generation-to-functional contract

Static contract check:

- 100 generation tasks matched 100 functional tasks.
- 100 unique task IDs.
- 50 Python and 50 JavaScript.
- Language, docstring, and code-stub fields matched.
- Candidate resolution succeeded 100/100 by task ID.

No model outputs were generated for this check.

### Sandbox security

- `reports/phase11_sandbox_security.json`: 16/16 probes passed.
- Limits and isolation flags remain unchanged.

### Reference validation

Existing evidence remains:

- 100/100 tasks
- 328/328 cases
- Python: 50/50 tasks, 164/164 cases
- JavaScript: 50/50 tasks, 164/164 cases

This is reference validation, not a finalist model result. It was not rerun during this fix.

## 7. Confirmation

- No finalist candidates were generated.
- No independent model evaluation was launched.
- No Phase 10, Phase 11, Phase 13, reference-validation, or pre-existing Phase 14 artifact was overwritten.
- No output was written to the intended finalist destinations.
- The old test partition was not accessed.

PHASE 14 TOOLING FIXED — PREFLIGHT AGAIN