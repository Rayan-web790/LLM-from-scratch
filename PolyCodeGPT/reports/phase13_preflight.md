# Phase 13 — Read-only evaluation preflight and model comparison plan

No benchmarks launched. No artifacts modified except this report.

## 1. Frozen checkpoints (verified)

| label | path | size (bytes) | SHA256 |
|---|---|---:|---|
| original (reference) | `checkpoints/polycodegpt_first_pass/step_008130.pt` | 205,612,557 | `5562acc2fd58837f9d4245042b7f0b10de65520aa2ba292999c80045470590bd` |
| control | `experiments/phase12b/runs/control/step_016260.pt` | 207,215,053 | `81a02873fa2f52838ecaf6ca141f9b665f5941d7ee67f325b34d16cb7cb2f1f9` |
| exp_a | `experiments/phase12b/runs/exp_a/step_016260.pt` | 207,215,053 | `65fab1180eff51c37dc7c3d872a8ed4545690ea46d150180eb3b7ed1f6fb6f45` |
| exp_b | `experiments/phase12b/runs/exp_b/step_016260.pt` | 207,215,053 | `fc70f5264a97d22c162986e172f8e70881713d5ae235c992e516a91ff67f5db6` |
| exp_c | `experiments/phase12b/runs/exp_c/step_016260.pt` | 207,215,053 | `cf0544099d73f85cdeb40bc1e0a93ce17bb790db4cb7136254be65d16c3b45b4` |

Hashes match the Phase 12C audit reports (control/exp_a/exp_b/exp_c) and the original reference
(`5562acc2…90bd`). No checkpoint was modified.

### Architecture and tokenizer compatibility

All five checkpoints share identical `model_config`:
`vocab_size=16000, d_model=384, n_layers=6, n_heads=6, ffn_dim=1536, context_length=512,
dropout=0.1, tie_embeddings=True, layer_norm_eps=1e-05, init_std=0.02,
architecture=decoder_only_transformer`; `parameter_count=16,988,160`; `weights_tied=True`;
`format_version=1`. Tokenizer SHA256 identical across all five:
`0459f8556a6f79ca1ba1c3dc3f76c26ae2e55653ebbf97beaa9bd23467358a03`.

No incompatibility. All five are interchangeable with the existing benchmark/functional tooling.

## 2. Existing evaluation tools (inspected)

### `scripts/evaluate_phase10_benchmark.py`

CLI (`--help`):
- `--device {cpu,cuda}` (default `cpu`)
- `--checkpoint CHECKPOINT` (default `step_008130.pt`, resolved against project root)
- `--verify-only` (lightweight, no generation)
- `--temperature TEMPERATURE` (default `0.0`)
- `--max-new-tokens` / `--max_new_tokens` (default `64`)

Fixed constants (not CLI-configurable): `SEED=42`, `N_PYTHON=50`, `N_JAVASCRIPT=50`,
`MAX_NEW_TOKENS=64`, `TEMPERATURE=0.0`. Loads model via `rebuild_model` from the training
checkpoint (falls back to model checkpoint), verifies weights tied, checks finite params.
Tokenizer fixed: `tokenizer/artifacts/polycodegpt_tokenizer_v1_1.json`. Model config fixed:
`configs/model.yaml`. Prompts: 50 Python + 50 JavaScript tasks from `scripts/phase10_tasks.py`,
built via `build_generation_prefix` (no trailing EOS). Generation: greedy (`temperature=0.0`),
`max_new_tokens=64`, EOS stop at token id 2.

Syntax validation: Python `ast.parse` on `code_stub + generated_text`; JavaScript via
`node --check` on a temp file (node v24.12.0 available). Metrics: syntax-valid count/rate,
EOS hits/rate, empty generations, unique-token ratio, repeated-line ratio,
repeated 4-gram ratio (BPE).

Output: timestamped JSON `reports/phase10_generation_evaluation_<YYYYMMDD_HHMMSS>.json`
(timestamped, safe) plus a fixed Markdown summary `reports/phase10_evaluation_summary.md`
(overwritten each run — see risks).

### `scripts/phase10_tasks.py`

Fixed 100-task inventory (50 Python + 50 JavaScript `(docstring, code_stub)` pairs). No CLI.
Task order is the tuple order; must be identical across all five runs (guaranteed by shared module).

### `functional/evaluate_functional.py`

CLI (`--help`):
- `--report REPORT` (default `reports/phase10_generation_evaluation_20261117_153043.json`)
- `--limit LIMIT` (first N tasks, optional)
- `--timeout TIMEOUT` (default `10.0`)
- `--json-out JSON_OUT` (default `reports/phase11_functional_results.json`)
- `--md-out MD_OUT` (default `reports/phase11_functional_results.md`)

Reads the Phase 10 generation report (never regenerates), indexes candidates by
`(language, docstring, code_stub)`, validates syntax, then runs each candidate against its
task's test cases inside the Docker sandbox. Outcomes per task: `syntax_failure`,
`missing_entry`, `runtime_error`, `timeout`, `resource_limit`, `incorrect_output`, `success`.
Requires `reports/phase11_sandbox_security.json` present with all probes passing (16/16).

### `functional/manifest.json`

100 tasks (50 Python + 50 JavaScript), 210 cases each language (116 normal, 92 boundary,
2 error, 6 convention cases). Candidate source expected:
`reports/phase10_generation_evaluation_20261117_153043.json`. For Phase 13, override
`--report` to the freshly generated report for each checkpoint.

### Sandbox (`functional/sandbox/runner.py`)

Docker images present: `polycodegpt-sandbox-python:v1`, `polycodegpt-sandbox-node:v1`.
Containment probes: 16/16 passed (`reports/phase11_sandbox_security.json`).
Limits: memory 128m (swap 128m), 0.50 CPU, 32 pids, nofile 64, fsize 1MB,
timeout 10.0s, network none, read-only root + tmpfs, unprivileged user 10001:10001,
cap-drop ALL, no-new-privileges. Node v24.12.0; Docker 29.0.1.

## 3. Phase 13 evaluation design (held constant across all five)

- Task inputs/order: identical 50 Python + 50 JavaScript from `scripts/phase10_tasks.py`.
- Prompt construction: `build_generation_prefix` (no EOS), canonical tokenizer v1.1.0.
- Tokenizer: `tokenizer/artifacts/polycodegpt_tokenizer_v1_1.json` (SHA `0459f855…8a03`).
- Generation: greedy, `temperature=0.0`, `max_new_tokens=64`, `seed=42`.
- Syntax validators: Python `ast.parse`; JavaScript `node --check`.
- Functional sandbox: Docker images above, 10s timeout, limits as listed.
- No per-checkpoint decoding tuning. No task/prompt changes.

Metrics to report per checkpoint (Python and JavaScript separately):
Generation: syntax-valid count/%, EOS completion count/%, max-token truncation count,
empty output count, repetition metrics (unique-token ratio, repeated-line ratio,
repeated 4-gram ratio). Functional: fully passed tasks/total, passed cases/total,
runtime-error tasks, incorrect-output tasks, syntax-failure tasks.

Original checkpoint included as reference baseline.

## 4. Blockers / risks

- No code change required to support all checkpoints; `--checkpoint` accepts any path and
  `--report` accepts any Phase 10 JSON. No blockers.
- Risk: gen benchmark overwrites `reports/phase10_evaluation_summary.md` each run.
  Preserve prior copy if needed before executing.
- Risk: functional eval default `--report` points at the old Phase 10 report
  (`20261117_153043`). Must override `--report` per checkpoint.
- Risk: functional eval default `--json-out`/`--md-out` overwrite Phase 11 artifacts.
  Use distinct names.
- Risk: gen benchmark seed is not CLI-configurable (fixed 42) — acceptable, but must not
  be altered; all runs use 42.
- Risk: JS syntax check requires `node` on PATH (present, v24.12.0).
- Risk: Docker must be running (present, Linux engine).

## 5. Exact Windows CMD commands (manual execution)

Run from the project root. Replace `<TS>` with the timestamp printed by each gen run.

### Generation benchmark — 5 runs (one per checkpoint)

```
cd "C:\Users\UltraPc\Desktop\LLM FROM SCRATCH\PolyCodeGPT"
"C:\Users\UltraPc\Desktop\LLM FROM SCRATCH\.venv\Scripts\python.exe" scripts\evaluate_phase10_benchmark.py --checkpoint checkpoints\polycodegpt_first_pass\step_008130.pt --device cuda --temperature 0.0 --max-new-tokens 64
"C:\Users\UltraPc\Desktop\LLM FROM SCRATCH\.venv\Scripts\python.exe" scripts\evaluate_phase10_benchmark.py --checkpoint experiments\phase12b\runs\control\step_016260.pt --device cuda --temperature 0.0 --max-new-tokens 64
"C:\Users\UltraPc\Desktop\LLM FROM SCRATCH\.venv\Scripts\python.exe" scripts\evaluate_phase10_benchmark.py --checkpoint experiments\phase12b\runs\exp_a\step_016260.pt --device cuda --temperature 0.0 --max-new-tokens 64
"C:\Users\UltraPc\Desktop\LLM FROM SCRATCH\.venv\Scripts\python.exe" scripts\evaluate_phase10_benchmark.py --checkpoint experiments\phase12b\runs\exp_b\step_016260.pt --device cuda --temperature 0.0 --max-new-tokens 64
"C:\Users\UltraPc\Desktop\LLM FROM SCRATCH\.venv\Scripts\python.exe" scripts\evaluate_phase10_benchmark.py --checkpoint experiments\phase12b\runs\exp_c\step_016260.pt --device cuda --temperature 0.0 --max-new-tokens 64
```

Each run writes `reports\phase10_generation_evaluation_<TS>.json` (timestamped) and
overwrites `reports\phase10_evaluation_summary.md`. Record each `<TS>`.

### Functional evaluation — 5 runs (one per generated report)

```
cd "C:\Users\UltraPc\Desktop\LLM FROM SCRATCH\PolyCodeGPT"
"C:\Users\UltraPc\Desktop\LLM FROM SCRATCH\.venv\Scripts\python.exe" functional\evaluate_functional.py --report reports\phase10_generation_evaluation_<TS_original>.json --json-out reports\phase13_original_functional.json --md-out reports\phase13_original_functional.md
"C:\Users\UltraPc\Desktop\LLM FROM SCRATCH\.venv\Scripts\python.exe" functional\evaluate_functional.py --report reports\phase10_generation_evaluation_<TS_control>.json --json-out reports\phase13_control_functional.json --md-out reports\phase13_control_functional.md
"C:\Users\UltraPc\Desktop\LLM FROM SCRATCH\.venv\Scripts\python.exe" functional\evaluate_functional.py --report reports\phase10_generation_evaluation_<TS_exp_a>.json --json-out reports\phase13_exp_a_functional.json --md-out reports\phase13_exp_a_functional.md
"C:\Users\UltraPc\Desktop\LLM FROM SCRATCH\.venv\Scripts\python.exe" functional\evaluate_functional.py --report reports\phase10_generation_evaluation_<TS_exp_b>.json --json-out reports\phase13_exp_b_functional.json --md-out reports\phase13_exp_b_functional.md
"C:\Users\UltraPc\Desktop\LLM FROM SCRATCH\.venv\Scripts\python.exe" functional\evaluate_functional.py --report reports\phase10_generation_evaluation_<TS_exp_c>.json --json-out reports\phase13_exp_c_functional.json --md-out reports\phase13_exp_c_functional.md
```

No benchmarks launched by this preflight. No artifacts modified except this report.

## PHASE 13 PREFLIGHT READY