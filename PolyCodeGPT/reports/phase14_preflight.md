# Phase 14 — Independent final evaluation preflight

Strictly read-only preflight. No evaluation was launched. No checkpoint, tokenizer, dataset, manifest, reference solution, script, or existing report was modified. Only this report was created. The old test partition was not accessed.

## 1. Finalist checkpoints (hashes verified)

| Label | Checkpoint path | SHA256 | Verified |
|---|---|---|---|
| Exp B | `experiments/phase12b/runs/exp_b/step_016260.pt` | `fc70f5264a97d22c162986e172f8e70881713d5ae235c992e516a91ff67f5db6` | yes |
| Exp C | `experiments/phase12b/runs/exp_c/step_016260.pt` | `cf0544099d73f85cdeb40bc1e0a93ce17bb790db4cb7136254be65d16c3b45b4` | yes |

Both checkpoints are the same 16,988,160-parameter decoder-only transformer configuration used in Phase 13 (vocab 16000, d_model 384, 6 layers/heads, ffn 1536, context 512, tokenizer SHA `0459f855…8a03`).

## 2. Independent suite (`functional/independent_v3/`)

Verified against `functional/independent_v3/manifest.json` and `functional/independent_v3/freeze_manifest.json`.

- Manifest benchmark: `independent_v3`.
- Manifest SHA256: `2a7fbf3ddc43f6a2fee368d6faa711dbe8fa841343c7cf9507c919cd3baeb46c`.
- Freeze manifest SHA256: `ec36c8803f5c22692a9b789dda4315a4f1c9901e08dc80ac09e435d3a065997e`.
- Tasks: **100** (50 Python, 50 JavaScript). Verified.
- Cases: **328** total (164 Python, 164 JavaScript). Verified.
- Task IDs: `py_v2_01` … `py_v2_50`, `js_v2_01` … `js_v2_50`; 100 unique IDs.
- Case kinds: normal/boundary/error cases; at least two normal cases and at least one boundary or error case per task.
- Manifest fields include `task_id`, `language`, `index`, `entry`, `signature`, `docstring`, `behavior`, `resolution`, `convention_note`, `params`, `params_with_defaults`, `cases`, `benchmark`.
- Freeze-manifest integrity: all 7 listed files (`spec/python_specs.py`, `spec/javascript_specs.py`, `spec/build_manifest.py`, `manifest.json`, `reference/python_refs.py`, `reference/javascript_refs.js`, `reference/validate_references.py`) match their recorded SHA256 values. No mismatch or missing file.
- `build_manifest.py --check` validates 100 tasks / 328 cases.
- Reference solutions validated in the Docker sandbox: `100/100` tasks, `328/328` cases (re-run during this preflight: same result).

## 3. Provenance and contamination-audit limitations

Contamination report: `reports/phase12b_contamination_v3.json`:

- `independent_tasks`: 100
- `phase11_entry_overlaps`: `[]`
- `v1_entry_overlaps`: `[]`
- `corpus_function_name_hits`: `[]`
- `reference_file_exact_hash_hits`: `[]`
- `behavior_near_duplicate_pairs`: `[]`
- `contamination_found`: `false`

This is a **preliminary lexical contamination check**, not proof of full independence. It checks entry-name overlap with Phase 11 and v1, corpus function-name hits, exact reference-file hash hits against corpus code, and behavior-text near-duplicates (Jaccard ≥ 0.6). It does not establish provenance of task ideas, hidden overlap, or semantic duplication beyond those lexical checks. Do not claim full independence from this evidence alone.

## 4. Evaluator compatibility

Existing evaluator: `functional/evaluate_functional.py`.

Verified compatible aspects:
- Loads any Phase 10-style generation report via `--report`.
- Uses the same tokenizer/model-generation provenance indirectly through the generation report (`checkpoint_step`, `seed`, `candidate_sha256`).
- Python syntax checking via `ast`-based candidate validation and JavaScript syntax checking via `node --check` (Node v24.12.0 available).
- All candidate execution occurs only through `functional/sandbox/runner.py` inside Docker.
- Sandbox security probes: 16/16 passed (`reports/phase11_sandbox_security.json`).
- Docker: 29.0.1; required images present: `polycodegpt-sandbox-python:v1`, `polycodegpt-sandbox-node:v1`.
- Sandbox limits: 128m memory/swap, 0.50 CPU, 32 pids, nofile 64, fsize 1MB, 10.0s timeout, no network.

**Blocker:** the evaluator does **not** support `independent_v3` directly:
- `MANIFEST = ROOT / "functional" / "manifest.json"` is hardcoded.
- `load_manifest()` reads only that Phase 11 manifest.
- There is no `--manifest` (or equivalent) CLI option.
- Candidate lookup keys use `(language, docstring, signature)`, which is compatible with the independent manifest's fields, but the hardcoded manifest path prevents using `functional/independent_v3/manifest.json`.
- The generation benchmark (`scripts/evaluate_phase10_benchmark.py`) is also hardcoded to `scripts.phase10_tasks` and asserts exactly 50 Python + 50 JavaScript Phase 10 tasks. It has no option to generate candidates for the independent task set.

Therefore, both candidate generation and functional evaluation for `independent_v3` require code changes. Per the preflight rules, this is reported as a blocker rather than patched.

## 5. Fixed evaluation protocol (to be used only after the blocker is resolved)

Identical settings for Exp B and Exp C:
- Task set/order: `functional/independent_v3/manifest.json`, 100 tasks (50 Python, 50 JavaScript), 328 cases.
- Prompts: task docstring and signature from the independent manifest, built with the same generation prefix format as Phase 13 (`<|lang:X|><|docstring|>{docstring}<|code|>{code_stub}`, no trailing EOS).
- Generation (if required): greedy decoding, temperature 0.0, max 64 new tokens, seed 42, tokenizer v1.1.0, same model config and checkpoint loading path as Phase 13.
- Syntax validators: Python `ast.parse`; JavaScript `node --check`.
- Functional execution: existing Docker sandbox, 10.0s timeout, identical limits and images.
- No per-finalist decoding tuning.
- Candidate code executed only inside Docker.

## 6. Predeclared model-selection criteria

Report separately for Python and JavaScript:
1. Fully passed tasks / total tasks.
2. Passed test cases / total cases.
3. Syntax-failure tasks.
4. Runtime errors and timeouts.
5. Incorrect outputs.
6. Other outcome categories supported by the evaluator (missing entry, resource limit, infrastructure error).

A task succeeds only when all of its cases pass. Complete functional task success is primary; case-level success is secondary. Do not hide a language-specific regression in an aggregate score. Criteria are fixed and must not be changed after results are seen.

## 7. Intended commands and outputs (not runnable until the blocker is fixed)

Do not run these until the evaluator/generator support `independent_v3`. They are shown to document the intended output locations only.

Intended functional invocation form after an approved code change:
```
"C:\Users\UltraPc\Desktop\LLM FROM SCRATCH\.venv\Scripts\python.exe" functional\evaluate_functional.py --report reports\phase14_exp_b_generation_evaluation_<TS>.json --json-out reports\phase14_exp_b_independent_functional.json --md-out reports\phase14_exp_b_independent_functional.md --timeout 10.0
"C:\Users\UltraPc\Desktop\LLM FROM SCRATCH\.venv\Scripts\python.exe" functional\evaluate_functional.py --report reports\phase14_exp_c_generation_evaluation_<TS>.json --json-out reports\phase14_exp_c_independent_functional.json --md-out reports\phase14_exp_c_independent_functional.md --timeout 10.0
```
The current CLI has no `--manifest` option, so these commands would still use the Phase 11 manifest and are not valid for Phase 14 as written.

Intended output files:
- `reports/phase14_exp_b_generation_evaluation_<timestamp>.json`
- `reports/phase14_exp_c_generation_evaluation_<timestamp>.json`
- `reports/phase14_exp_b_independent_functional.json` / `.md`
- `reports/phase14_exp_c_independent_functional.json` / `.md`

Note: the generation benchmark overwrites `reports/phase10_evaluation_summary.md` on each run; preserve any needed prior copy before execution.

## 8. Blockers and risks

- **Blocker 1:** `functional/evaluate_functional.py` hardcodes `functional/manifest.json` and has no manifest-selection option. It cannot evaluate `functional/independent_v3/manifest.json`.
- **Blocker 2:** `scripts/evaluate_phase10_benchmark.py` is hardcoded to `scripts.phase10_tasks` and cannot generate candidates for the independent task set.
- Risk: running the current tools would silently evaluate the wrong (Phase 11) task set.
- Risk: candidate generation for independent tasks is not yet implemented; Phase 14 requires an independent-task-aware generation path.
- Risk: contamination evidence is preliminary and should not be overstated as full independence.
- No evaluation was launched during this preflight.

## 9. Verdict

FIX FIRST — DO NOT RUN INDEPENDENT EVALUATION