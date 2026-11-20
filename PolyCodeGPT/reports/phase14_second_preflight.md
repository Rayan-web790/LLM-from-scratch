# Phase 14 — Second preflight after tooling fix

Strictly read-only preflight. No finalist candidate generation and no finalist functional evaluation was launched. The old test partition was not accessed. Only this report was created.

## 1. Frozen model identity

| Finalist | Checkpoint | SHA256 | Status |
|---|---|---|---|
| Exp B | `experiments/phase12b/runs/exp_b/step_016260.pt` | `fc70f5264a97d22c162986e172f8e70881713d5ae235c992e516a91ff67f5db6` | matches expected |
| Exp C | `experiments/phase12b/runs/exp_c/step_016260.pt` | `cf0544099d73f85cdeb40bc1e0a93ce17bb790db4cb7136254be65d16c3b45b4` | matches expected |

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

The checkpoint model configuration matches `configs/model.yaml`.

## 2. Independent suite remains frozen

| Item | Verified value |
|---|---|
| Manifest | `functional/independent_v3/manifest.json` |
| Manifest SHA256 | `2a7fbf3ddc43f6a2fee368d6faa711dbe8fa841343c7cf9507c919cd3baeb46c` |
| Benchmark label | `independent_v3` |
| Tasks | 100 |
| Python tasks | 50 |
| JavaScript tasks | 50 |
| Unique task IDs | 100 |
| Cases | 328 |
| Python cases | 164 |
| JavaScript cases | 164 |

Freeze manifest:
- Path: `functional/independent_v3/freeze_manifest.json`
- SHA256: `ec36c8803f5c22692a9b789dda4315a4f1c9901e08dc80ac09e435d3a065997e`
- Listed files: 7
- Hash mismatches: none
- Missing files: none

The seven frozen files are the two spec modules, the manifest builder, the independent manifest, the two reference files, and the reference validator. No independent-suite file was modified.

## 3. Generation evaluator verification

### CLI help evidence

Actual `--help` output includes:

```
usage: evaluate_phase10_benchmark.py [-h] [--device {cpu,cuda}]
                                     [--checkpoint CHECKPOINT] [--verify-only]
                                     [--temperature TEMPERATURE]
                                     [--max-new-tokens MAX_NEW_TOKENS]
                                     [--task-manifest TASK_MANIFEST]
...
  --task-manifest TASK_MANIFEST, --task_manifest TASK_MANIFEST
                        Optional JSON task manifest (e.g.
                        functional/independent_v3/manifest.json)
```

### Default behavior

When `--task-manifest` is omitted:

- `_phase10_generation_tasks()` is used.
- The original Phase 10 inventory is loaded: 100 tasks, 50 Python and 50 JavaScript.
- Phase 10 tasks retain `task_id = null`.
- The prompt format is unchanged:
  `<|lang:python|>` or `<|lang:javascript|>`, then `<|docstring|>{docstring}<|code|>{code_stub}`.
- Existing JSON fields remain present; new fields are additive.

### Independent-manifest behavior

When `--task-manifest functional/independent_v3/manifest.json` is supplied:

- `load_task_manifest()` reads the manifest's `tasks` list.
- `task_id` is retained.
- `language` is validated as `python` or `javascript`.
- `docstring` is retained.
- `signature` is mapped to the generation `code_stub`.
- Duplicate task IDs, missing docstrings, missing signatures, and unsupported languages fail closed.
- The suite label is `independent_v3`.
- All 100 independent task IDs are retained: `py_v2_01` through `py_v2_50`, and `js_v2_01` through `js_v2_50`.

### Candidate report provenance

The generated JSON records:

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
- `per_task_results`

Each `per_task_results` entry records `task_id`, `language`, `docstring`, `code_stub`, `generated_text`, `syntax_candidate`, `syntax_valid`, `eos_hit`, token metrics, and generated token count.

### Generation settings

Unchanged:

- Seed: 42
- Temperature: 0.0
- Greedy decoding: `temperature == 0` selects `argmax`
- Maximum new tokens: 64
- EOS: generation stops when EOS is produced; EOS is removed from the returned continuation and `eos_hit` is recorded.

### Static compatibility check

A read-only check compared the generation manifest loader against the functional manifest loader:

- 100 generation tasks matched 100 functional tasks in order.
- 100/100 task IDs matched.
- 100/100 languages matched.
- 100/100 docstrings matched.
- 100/100 `code_stub` values matched the functional `signature`.
- 50 Python and 50 JavaScript tasks.
- Synthetic contract records resolved 100/100 candidates by task ID with no language confusion.

No model outputs were generated for this check.

### Residual import issue

`scripts/evaluate_phase10_benchmark.py` still has a module-level import:

```python
from scripts.phase10_tasks import python_tasks, javascript_tasks, task_counts, all_generation_prefixes
```

Those names are not used by the independent-manifest path; task loading for that path uses only `load_task_manifest()`. However, the module is still imported even when `--task-manifest` selects `independent_v3`. This does not inject Phase 10 task data into the independent run, but it does not meet the literal requirement that no import of `scripts.phase10_tasks` occur when the independent manifest is selected.

## 4. Functional evaluator verification

### CLI help evidence

Actual `--help` output includes:

```
usage: evaluate_functional.py [-h] [--report REPORT] [--manifest MANIFEST]
                              [--limit LIMIT] [--timeout TIMEOUT]
                              [--json-out JSON_OUT] [--md-out MD_OUT]
...
  --manifest MANIFEST  functional task manifest to evaluate
```

### Manifest selection

- Omitting `--manifest` retains `functional/manifest.json` (Phase 11): 100 tasks, 420 cases, 210 per language.
- Supplying `--manifest functional/independent_v3/manifest.json` loads the independent suite: 100 tasks, 328 cases, 164 per language.
- A non-existent or malformed manifest raises `SystemExit`; it does not fall back to `functional/manifest.json`.

### Candidate matching

Candidates are indexed in two ways:

- Stable `task_id`, when the candidate report supplies one.
- The legacy `(language, docstring, code_stub)` key, for older reports.

When the independent manifest is used:

- Matching is by independent task ID.
- Candidate `language` must equal the task language.
- Candidate `docstring` must equal the task docstring.
- Candidate `code_stub` must equal the task `signature`.
- Candidate `task_id` must equal the task `task_id`.

Failure behavior:

- Missing candidate: task is reported `[MISSING]` and is not evaluated.
- Duplicate candidate task ID: `ValueError`, run aborts.
- Unknown candidate task ID: `ValueError`, run aborts.
- Mismatched language, docstring, code stub, or task ID: `ValueError`, run aborts.

### No silent Phase 11 fallback

- `args.manifest is None` selects `functional/manifest.json`.
- A supplied path is passed directly to `load_manifest()` and is never replaced with the default.
- Candidate IDs from the independent manifest are distinct from Phase 10.
- The intersection of independent `(language, docstring, signature)` keys with the Phase 10 task keys is empty.
- There is no import of `scripts.phase10_tasks` in `functional/evaluate_functional.py`.

### Functional JSON provenance

The functional JSON payload records:

- `candidate_source`
- `manifest`
- `manifest_path`
- `manifest_sha256`
- `manifest_hash`
- `checkpoint_step`
- `seed`
- `safety_probes`
- `sandbox`
- `metrics`
- `task_outcomes`

The manifest path and SHA256 are the actual selected manifest, including the independent manifest.

### Reference-path note

The independent manifest has no `reference_path` field. `functional/evaluate_functional.py` evaluates candidate code against manifest `cases`; it does not load reference solutions. Reference paths are fixed in `functional/independent_v3/reference/validate_references.py` and point to the independent `reference/python_refs.py` and `reference/javascript_refs.js`. This is a documentation clarification, not a blocker for candidate evaluation.

## 5. Sandbox configuration

Unchanged:

- Images: `polycodegpt-sandbox-python:v1`, `polycodegpt-sandbox-node:v1`
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
- Safety probes: 16/16 passed (`reports/phase11_sandbox_security.json`)

## 6. Generation-to-functional data contract

The intended full contract is:

1. Generation reads `functional/independent_v3/manifest.json`.
2. It emits one `per_task_results` entry per task with `task_id`, `language`, `docstring`, and `code_stub = signature`.
3. It sets `syntax_candidate = code_stub + generated_text`.
4. Functional evaluation reads the same manifest with `--manifest`.
5. It indexes candidates by `task_id`.
6. It validates language, docstring, and code stub against the manifest task.
7. It executes `syntax_candidate` for `entry` against `cases` inside Docker.

Static checks confirm:

- 100/100 task IDs match.
- 100/100 languages match.
- 100/100 docstrings match.
- 100/100 code stubs match signatures.
- 50 Python and 50 JavaScript tasks.
- No key overlap with Phase 10 tasks.
- No candidate reordering dependency: matching is by task ID.
- No reference resolution occurs in the functional evaluator.

## 7. Tests and reference evidence

### Unit tests

Command rerun against the current source:

```
C:\Users\UltraPc\Desktop\LLM FROM SCRATCH\.venv\Scripts\python.exe -m pytest tests/unit -q
```

Result: **308 passed** in 128.37s.

`tests/unit/test_phase14_tooling.py` covers:

- Default generation task set remains 100 Phase 10 tasks.
- Independent manifest loads 100 tasks, 50 per language.
- Default functional manifest remains 100 tasks / 420 cases.
- Independent functional manifest loads 100 tasks / 328 cases.
- No implicit fallback from independent_v3 to Phase 11.
- Candidate matching by independent task ID.
- Missing candidate returns none.
- Duplicate, unknown, and mismatched candidates fail safely.
- Sandbox configuration unchanged.

### Reference validation

Recorded in `reports/phase14_tooling_fix_report.md`:

- The updated `functional/evaluate_functional.py` was run with `--manifest functional/independent_v3/manifest.json`.
- Reference solutions were supplied through a temporary generation report; no model generation was performed.
- Result: **100/100 tasks, 328/328 cases**.
- Python: 50/50 tasks, 164/164 cases.
- JavaScript: 50/50 tasks, 164/164 cases.

This is reference validation, not a finalist model result. It must not be reported as either finalist’s score.

### Contamination evidence

`reports/phase12b_contamination_v3.json` reports `contamination_found: false` with no entry-name overlaps, no corpus function-name hits, no reference-file exact-hash hits, and no behavior near-duplicate pairs.

This is a **preliminary lexical contamination check**, not proof of full independence. It does not establish provenance of task ideas, hidden overlap, or semantic duplication beyond those checks.

## 8. Blockers

### Blocker 1 — generation output paths are not safe for finalists

`scripts/evaluate_phase10_benchmark.py` has no `--json-out` or `--md-out` option.

- JSON output is always `reports/phase10_generation_evaluation_<timestamp>.json`.
- Markdown output is always `reports/phase10_evaluation_summary.md`.
- The markdown path is a Phase 10 output and is overwritten on every generation run.
- The markdown heading is also fixed as “Phase 10 Evaluation Summary”, so an independent run would produce a misleading artifact.
- There is no CLI way to guarantee distinct candidate and summary output filenames for Exp B and Exp C.

A backup-and-restore wrapper could preserve the old file content after a successful run, but it would still overwrite the Phase 10 artifact during generation and would risk loss if the process is interrupted. That is not sufficient for a strict “do not overwrite Phase 10 outputs” requirement.

### Blocker 2 — residual Phase 10 module import

The module-level `from scripts.phase10_tasks import ...` remains in `scripts/evaluate_phase10_benchmark.py`. It is behaviorally inert for the independent-manifest path, but it violates the literal no-import condition. The local import inside `_phase10_generation_tasks()` is the only import needed for the default path.

### Non-blocking clarification — reference paths

The functional evaluator does not load reference solutions, and the independent manifest does not contain reference paths. Reference validation uses its own fixed paths in `reference/validate_references.py`. This is safe for candidate evaluation, but the preflight requirement should be read as applying to reference validation, not to `evaluate_functional.py`.

## 9. Exact Windows CMD commands

These are the exact current CLI forms. **Do not run the generation commands until Blocker 1 is fixed**, because the current generator will overwrite `reports\phase10_evaluation_summary.md`.

### A. Exp B generation using the independent manifest

```
cd /d "C:\Users\UltraPc\Desktop\LLM FROM SCRATCH\PolyCodeGPT"
"C:\Users\UltraPc\Desktop\LLM FROM SCRATCH\.venv\Scripts\python.exe" scripts\evaluate_phase10_benchmark.py --device cpu --checkpoint experiments\phase12b\runs\exp_b\step_016260.pt --temperature 0.0 --max-new-tokens 64 --task-manifest functional\independent_v3\manifest.json
```

Planned candidate output after a safe output-path fix/rename:
`reports\phase14_exp_b_generation_independent_v3.json`

### B. Exp C generation using the same independent manifest

```
cd /d "C:\Users\UltraPc\Desktop\LLM FROM SCRATCH\PolyCodeGPT"
"C:\Users\UltraPc\Desktop\LLM FROM SCRATCH\.venv\Scripts\python.exe" scripts\evaluate_phase10_benchmark.py --device cpu --checkpoint experiments\phase12b\runs\exp_c\step_016260.pt --temperature 0.0 --max-new-tokens 64 --task-manifest functional\independent_v3\manifest.json
```

Planned candidate output after a safe output-path fix/rename:
`reports\phase14_exp_c_generation_independent_v3.json`

### C. Exp B functional evaluation

```
cd /d "C:\Users\UltraPc\Desktop\LLM FROM SCRATCH\PolyCodeGPT"
"C:\Users\UltraPc\Desktop\LLM FROM SCRATCH\.venv\Scripts\python.exe" functional\evaluate_functional.py --report reports\phase14_exp_b_generation_independent_v3.json --manifest functional\independent_v3\manifest.json --timeout 10.0 --json-out reports\phase14_exp_b_functional_independent_v3.json --md-out reports\phase14_exp_b_functional_independent_v3.md
```

Outputs:
- `reports\phase14_exp_b_functional_independent_v3.json`
- `reports\phase14_exp_b_functional_independent_v3.md`

### D. Exp C functional evaluation

```
cd /d "C:\Users\UltraPc\Desktop\LLM FROM SCRATCH\PolyCodeGPT"
"C:\Users\UltraPc\Desktop\LLM FROM SCRATCH\.venv\Scripts\python.exe" functional\evaluate_functional.py --report reports\phase14_exp_c_generation_independent_v3.json --manifest functional\independent_v3\manifest.json --timeout 10.0 --json-out reports\phase14_exp_c_functional_independent_v3.json --md-out reports\phase14_exp_c_functional_independent_v3.md
```

Outputs:
- `reports\phase14_exp_c_functional_independent_v3.json`
- `reports\phase14_exp_c_functional_independent_v3.md`

### Candidate verification before functional evaluation

After each generation run, verify the candidate JSON exists and contains exactly 100 unique task IDs, 50 Python and 50 JavaScript tasks, the independent manifest SHA256, and the expected checkpoint SHA256.

Exp B:

```
"C:\Users\UltraPc\Desktop\LLM FROM SCRATCH\.venv\Scripts\python.exe" -c "import json; p=r'reports\phase14_exp_b_generation_independent_v3.json'; d=json.load(open(p,encoding='utf-8')); ids=[t['task_id'] for t in d['per_task_results']]; assert len(ids)==100 and len(set(ids))==100; assert d['python_tasks']==50 and d['javascript_tasks']==50; assert d['task_manifest_sha256']=='2a7fbf3ddc43f6a2fee368d6faa711dbe8fa841343c7cf9507c919cd3baeb46c'; assert d['checkpoint_sha256']=='fc70f5264a97d22c162986e172f8e70881713d5ae235c992e516a91ff67f5db6'; print('Exp B candidate JSON OK:',len(ids),'unique IDs')"
```

Exp C:

```
"C:\Users\UltraPc\Desktop\LLM FROM SCRATCH\.venv\Scripts\python.exe" -c "import json; p=r'reports\phase14_exp_c_generation_independent_v3.json'; d=json.load(open(p,encoding='utf-8')); ids=[t['task_id'] for t in d['per_task_results']]; assert len(ids)==100 and len(set(ids))==100; assert d['python_tasks']==50 and d['javascript_tasks']==50; assert d['task_manifest_sha256']=='2a7fbf3ddc43f6a2fee368d6faa711dbe8fa841343c7cf9507c919cd3baeb46c'; assert d['checkpoint_sha256']=='cf0544099d73f85cdeb40bc1e0a93ce17bb790db4cb7136254be65d16c3b45b4'; print('Exp C candidate JSON OK:',len(ids),'unique IDs')"
```

Do not proceed to C or D unless the corresponding check prints OK.

## 10. Final assessment

The independent suite, finalist checkpoints, functional evaluator, sandbox, tests, and reference evidence are consistent. The data contract can carry all 100 task IDs without reordering or language confusion.

The generation evaluator is not yet safe to run for finalists because it cannot write distinct candidate/summary paths without overwriting a Phase 10 artifact, and it still imports `scripts.phase10_tasks` at module level even when the independent manifest is selected. These are tooling issues, not model or suite issues. No finalist candidates were generated and no finalist evaluation was launched during this preflight.

FIX FIRST — DO NOT GENERATE FINALIST CANDIDATES