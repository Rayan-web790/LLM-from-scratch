# Phase 14 — Final preflight after tooling fix 2

Strictly read-only preflight. No finalist candidates were generated and no independent model evaluation was launched. The old test partition was not accessed. Only this report was created.

## 1. Frozen finalist checkpoints

| Finalist | Checkpoint | Size | SHA256 | Status |
|---|---|---:|---|---|
| Exp B | `experiments/phase12b/runs/exp_b/step_016260.pt` | 207,215,053 bytes | `fc70f5264a97d22c162986e172f8e70881713d5ae235c992e516a91ff67f5db6` | matches expected |
| Exp C | `experiments/phase12b/runs/exp_c/step_016260.pt` | 207,215,053 bytes | `cf0544099d73f85cdeb40bc1e0a93ce17bb790db4cb7136254be65d16c3b45b4` | matches expected |

Both checkpoints report:

- `step`: 16260
- `model_config.architecture`: `decoder_only_transformer`
- `model_config.vocab_size`: 16000
- `model_config.d_model`: 384
- `model_config.n_layers`: 6
- `model_config.n_heads`: 6
- `model_config.ffn_dim`: 1536
- `model_config.context_length`: 512
- `model_config.tie_embeddings`: true
- `parameter_count`: 16,988,160
- `weights_tied`: true
- `tokenizer_sha256`: `0459f8556a6f79ca1ba1c3dc3f76c26ae2e55653ebbf97beaa9bd23467358a03`

The tokenizer artifact `tokenizer/artifacts/polycodegpt_tokenizer_v1_1.json` has the same SHA256:
`0459f8556a6f79ca1ba1c3dc3f76c26ae2e55653ebbf97beaa9bd23467358a03`.

The configuration matches `configs/model.yaml`. Neither checkpoint was modified.

## 2. Frozen independent suite

| Item | Verified value |
|---|---|
| Manifest | `functional/independent_v3/manifest.json` |
| Manifest SHA256 | `2a7fbf3ddc43f6a2fee368d6faa711dbe8fa841343c7cf9507c919cd3baeb46c` |
| Benchmark | `independent_v3` |
| Tasks | 100 |
| Unique task IDs | 100 |
| Python tasks | 50 |
| JavaScript tasks | 50 |
| Cases | 328 |
| Python cases | 164 |
| JavaScript cases | 164 |

Freeze manifest:
- Path: `functional/independent_v3/freeze_manifest.json`
- SHA256: `ec36c8803f5c22692a9b789dda4315a4f1c9901e08dc80ac09e435d3a065997e`
- Listed files: 7
- Hash mismatches: none
- Recorded reference validation: `100/100 tasks, 328/328 cases (sandbox)`

The seven frozen files were verified individually:

- `functional/independent_v3/spec/python_specs.py` — OK
- `functional/independent_v3/spec/javascript_specs.py` — OK
- `functional/independent_v3/spec/build_manifest.py` — OK
- `functional/independent_v3/manifest.json` — OK
- `functional/independent_v3/reference/python_refs.py` — OK
- `functional/independent_v3/reference/javascript_refs.js` — OK
- `functional/independent_v3/reference/validate_references.py` — OK

No independent-suite file was modified. The old test partition was not accessed.

## 3. Generation tool verification

### CLI options

`scripts/evaluate_phase10_benchmark.py --help` confirms:

- `--checkpoint CHECKPOINT`
- `--task-manifest TASK_MANIFEST` / `--task_manifest TASK_MANIFEST`
- `--json-out JSON_OUT`
- `--md-out MD_OUT`
- `--temperature TEMPERATURE`
- `--max-new-tokens MAX_NEW_TOKENS` / `--max_new_tokens MAX_NEW_TOKENS`
- `--device {cpu,cuda}`

Exact help description for the new output options:

```
--json-out JSON_OUT   Explicit JSON report path (refuses to overwrite an
                      existing file)
--md-out MD_OUT       Explicit Markdown summary path (refuses to overwrite
                      an existing file)
```

### Import isolation

The module-level `scripts.phase10_tasks` import has been removed. The only task-module import now occurs inside `_phase10_generation_tasks()`:

```python
from scripts.phase10_tasks import python_tasks, javascript_tasks
```

When `--task-manifest` selects the independent suite, `load_task_manifest()` is used and `scripts.phase10_tasks` is not imported.

Evidence:
- Static grep shows no module-level task import.
- `tests/unit/test_phase14_tooling_fix2.py::test_independent_mode_does_not_import_phase10_tasks` installs a `sys.meta_path` blocker that raises on any `scripts.phase10_tasks` import, imports the generation module in a fresh subprocess, and loads 100 independent tasks successfully.
- The same test asserts `scripts.phase10_tasks` is absent from `sys.modules`.
- `test_default_phase10_path_still_loads_100_tasks` confirms the default path still loads the original 100 Phase 10 tasks.

### Default compatibility

When `--task-manifest`, `--json-out`, and `--md-out` are all omitted:

- The original Phase 10 inventory is used: 100 tasks, 50 Python and 50 JavaScript.
- JSON goes to `reports/phase10_generation_evaluation_<timestamp>.json`.
- Markdown goes to `reports/phase10_evaluation_summary.md`.
- Historical default overwrite behavior is preserved.

### Explicit output safety

When `--json-out` or `--md-out` is supplied:

- Relative paths resolve against the project root.
- If only one path is supplied, the other is derived as a sibling with the matching extension.
- Parent directories are created.
- Existing JSON or Markdown targets are rejected before generation starts.
- Explicit writes use exclusive creation (`open(..., "x")`), so they cannot overwrite an existing file.
- JSON and Markdown cannot use the same path.

### Report provenance

The generated JSON retains:

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
- prompts/tasks/syntax/EOS/metrics
- `per_task_results`

Each per-task result retains `task_id`, `language`, `docstring`, `code_stub`, `generated_text`, `syntax_candidate`, `syntax_valid`, `eos_hit`, token metrics, and generated token count.

### Tests covering generation corrections

`tests/unit/test_phase14_tooling_fix2.py` contains 11 tests:

- `test_generation_help_lists_output_options`
- `test_independent_mode_does_not_import_phase10_tasks`
- `test_default_phase10_path_still_loads_100_tasks`
- `test_independent_manifest_retains_all_ids_and_languages`
- `test_explicit_paths_are_resolved_and_parents_created`
- `test_one_explicit_path_derives_sibling`
- `test_existing_json_is_rejected`
- `test_existing_markdown_is_rejected`
- `test_default_paths_preserve_phase10_names_and_overwrite_behavior`
- `test_main_honors_explicit_paths_and_provenance`
- `test_main_refuses_to_overwrite_before_generation`

Targeted result: **11 passed**.

## 4. Functional evaluator verification

`functional/evaluate_functional.py --help` confirms:

- `--report REPORT`
- `--manifest MANIFEST`
- `--limit LIMIT`
- `--timeout TIMEOUT`
- `--json-out JSON_OUT`
- `--md-out MD_OUT`

Behavior:

- Omitting `--manifest` preserves the original Phase 11 manifest: 100 tasks, 420 cases.
- Supplying `--manifest functional/independent_v3/manifest.json` loads 100 tasks and 328 cases.
- A non-existent or malformed manifest raises `SystemExit`; it does not fall back to the Phase 11 manifest.
- Candidate matching uses stable independent task IDs when present.
- Language, docstring, code stub/signature, and task ID are validated.
- Missing candidates are reported and not evaluated.
- Duplicate, unknown, or mismatched candidate IDs raise `ValueError` and stop the run.

Targeted functional tests: **34 passed** across `test_phase14_tooling.py` and `test_functional_evaluator.py`.

### Sandbox configuration

Unchanged:

- Images: `polycodegpt-sandbox-python:v1`, `polycodegpt-sandbox-node:v1`
- Docker server: 29.0.1, Linux
- `--network none`
- `--read-only`
- `--tmpfs /tmp:rw,noexec,nosuid,nodev,size=16m`
- `--user 10001:10001`
- `--cap-drop ALL`
- `--security-opt no-new-privileges:true`
- Memory: 128m
- Memory swap: 128m
- CPUs: 0.50
- PIDs: 32
- nofile: 64
- fsize: 1,048,576
- Timeout: 10.0s
- No bind mounts; payload over stdin, results over stderr
- Container removed after every run

Sandbox containment evidence: **16/16 passed** (`reports/phase11_sandbox_security.json`).

## 5. Generation-to-functional contract

Static checks confirm:

- 100 generation tasks map to 100 functional tasks.
- 100 unique task IDs.
- 50 Python and 50 JavaScript tasks.
- Every task ID matches.
- Every language label matches.
- Every docstring matches.
- Every generation `code_stub` matches the functional `signature`.
- Candidate resolution succeeds 100/100 by task ID.
- Phase 10 key overlap: 0.
- Phase 10 task-ID overlap: 0.

No Phase 10 candidate records or task IDs can be reused for the independent suite. No model outputs were generated for this check.

## 6. Future output destinations

All eight destinations are currently unused:

| Path | Status |
|---|---|
| `reports/phase14_exp_b_generation.json` | unused |
| `reports/phase14_exp_b_generation.md` | unused |
| `reports/phase14_exp_c_generation.json` | unused |
| `reports/phase14_exp_c_generation.md` | unused |
| `reports/phase14_exp_b_functional.json` | unused |
| `reports/phase14_exp_b_functional.md` | unused |
| `reports/phase14_exp_c_functional.json` | unused |
| `reports/phase14_exp_c_functional.md` | unused |

No placeholder files were created. If any destination exists at execution time, stop and choose a new unique path rather than overwriting it.

## 7. Current tests and security evidence

Full unit suite, run against the current source:

```
C:\Users\UltraPc\Desktop\LLM FROM SCRATCH\.venv\Scripts\python.exe -m pytest tests/unit -q
```

Result: **319 passed** in 89.79s.

CLI help was verified directly for both tools:

- Generation: `--task-manifest`, `--json-out`, and `--md-out` present.
- Functional: `--manifest`, `--report`, `--json-out`, and `--md-out` present.

Security and reference evidence:

- Sandbox containment: 16/16 passed.
- Reference validation: 100/100 tasks, 328/328 cases (recorded in `reports/phase14_tooling_fix_report.md` and the freeze manifest). It was not rerun because rerunning could create or overwrite evidence artifacts.
- Independent freeze manifest: seven files unchanged.

Contamination evidence remains preliminary:

- `reports/phase12b_contamination_v3.json` reports `contamination_found: false`.
- No Phase 11 or v1 entry overlaps, no corpus function-name hits, no reference-file exact-hash hits, and no behavior near-duplicate pairs.
- This is a **preliminary lexical check only**, not proof of full independence. It does not establish provenance of task ideas, hidden overlap, or semantic duplication beyond those checks.

## 8. Exact Windows CMD commands

Run all commands from the project root. The `--task-manifest` relative path is resolved from the current working directory, so the `cd /d` step is required.

### A. Exp B generation

```
cd /d "C:\Users\UltraPc\Desktop\LLM FROM SCRATCH\PolyCodeGPT"
"C:\Users\UltraPc\Desktop\LLM FROM SCRATCH\.venv\Scripts\python.exe" scripts\evaluate_phase10_benchmark.py --device cuda --checkpoint experiments\phase12b\runs\exp_b\step_016260.pt --task-manifest functional\independent_v3\manifest.json --temperature 0.0 --max-new-tokens 64 --json-out reports\phase14_exp_b_generation.json --md-out reports\phase14_exp_b_generation.md
```

### B. Exp C generation

```
cd /d "C:\Users\UltraPc\Desktop\LLM FROM SCRATCH\PolyCodeGPT"
"C:\Users\UltraPc\Desktop\LLM FROM SCRATCH\.venv\Scripts\python.exe" scripts\evaluate_phase10_benchmark.py --device cuda --checkpoint experiments\phase12b\runs\exp_c\step_016260.pt --task-manifest functional\independent_v3\manifest.json --temperature 0.0 --max-new-tokens 64 --json-out reports\phase14_exp_c_generation.json --md-out reports\phase14_exp_c_generation.md
```

### C. Exp B functional evaluation

Run only after the Exp B generation JSON passes the verification step below.

```
cd /d "C:\Users\UltraPc\Desktop\LLM FROM SCRATCH\PolyCodeGPT"
"C:\Users\UltraPc\Desktop\LLM FROM SCRATCH\.venv\Scripts\python.exe" functional\evaluate_functional.py --report reports\phase14_exp_b_generation.json --manifest functional\independent_v3\manifest.json --timeout 10.0 --json-out reports\phase14_exp_b_functional.json --md-out reports\phase14_exp_b_functional.md
```

### D. Exp C functional evaluation

Run only after the Exp C generation JSON passes the verification step below.

```
cd /d "C:\Users\UltraPc\Desktop\LLM FROM SCRATCH\PolyCodeGPT"
"C:\Users\UltraPc\Desktop\LLM FROM SCRATCH\.venv\Scripts\python.exe" functional\evaluate_functional.py --report reports\phase14_exp_c_generation.json --manifest functional\independent_v3\manifest.json --timeout 10.0 --json-out reports\phase14_exp_c_functional.json --md-out reports\phase14_exp_c_functional.md
```

### Verification after Exp B generation

```
"C:\Users\UltraPc\Desktop\LLM FROM SCRATCH\.venv\Scripts\python.exe" -c "import json; p=r'reports\phase14_exp_b_generation.json'; d=json.load(open(p,encoding='utf-8')); ids=[t['task_id'] for t in d['per_task_results']]; assert len(ids)==100 and len(set(ids))==100; assert d['python_tasks']==50 and d['javascript_tasks']==50; assert d['task_manifest_sha256']=='2a7fbf3ddc43f6a2fee368d6faa711dbe8fa841343c7cf9507c919cd3baeb46c'; assert d['checkpoint_sha256']=='fc70f5264a97d22c162986e172f8e70881713d5ae235c992e516a91ff67f5db6'; print('Exp B generation JSON OK: 100 unique IDs, 50 Python, 50 JavaScript')"
```

### Verification after Exp C generation

```
"C:\Users\UltraPc\Desktop\LLM FROM SCRATCH\.venv\Scripts\python.exe" -c "import json; p=r'reports\phase14_exp_c_generation.json'; d=json.load(open(p,encoding='utf-8')); ids=[t['task_id'] for t in d['per_task_results']]; assert len(ids)==100 and len(set(ids))==100; assert d['python_tasks']==50 and d['javascript_tasks']==50; assert d['task_manifest_sha256']=='2a7fbf3ddc43f6a2fee368d6faa711dbe8fa841343c7cf9507c919cd3baeb46c'; assert d['checkpoint_sha256']=='cf0544099d73f85cdeb40bc1e0a93ce17bb790db4cb7136254be65d16c3b45b4'; print('Exp C generation JSON OK: 100 unique IDs, 50 Python, 50 JavaScript')"
```

Do not launch the corresponding functional evaluation unless the check prints OK.

## 9. Remaining limitations

- Contamination evidence is preliminary lexical evidence only, not proof of full independence.
- Reference validation was not rerun during this preflight; the existing 100/100 task and 328/328 case result is preserved.
- The generation Markdown template still uses the historical heading “Phase 10 Evaluation Summary” even for manifest-driven runs. This is cosmetic and does not affect the JSON or evaluation.
- The generation `--task-manifest` relative path is resolved from the current working directory, so the commands must run from the project root.
- `--device cuda` requires a working CUDA device. CUDA is available on this host: 1 device, NVIDIA GeForce RTX 5060 Laptop GPU.
- If any intended output path exists at execution time, stop and choose a new unique path; the tooling will also refuse to overwrite it.

## 10. Final confirmation

- No finalist candidates were generated.
- No finalist functional evaluation was launched.
- No checkpoint, tokenizer, manifest, reference solution, freeze file, sandbox setting, or existing report was modified.
- The four generation destinations and four functional destinations are unused.

PHASE 14 READY — SAFE TO GENERATE FINALIST CANDIDATES