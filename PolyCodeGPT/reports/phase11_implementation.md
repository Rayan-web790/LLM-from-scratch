# Phase 11 — Functional Correctness Evaluation Infrastructure

Status: **implemented and executed**. 100 tasks, 420 test cases, 16/16 containment probes
passed, 100/100 reference tasks validated inside the sandbox, and the frozen Phase 10
candidates were evaluated without re-running generation.

All results below come from actual isolated execution. Nothing is claimed that was not run.

## 1. Environment requirements reported before any change

| requirement | status |
|---|---|
| Docker Desktop with Linux containers | already installed; engine was stopped, started with `docker desktop start` |
| WSL2 | present (default version 2, `docker-desktop` distro) |
| Windows configuration changes | **none** |
| Software installed on the host | **none** |
| Container image downloaded | `node:22-alpine` (~50 MB, the only download); `python:3` was already cached |
| Images built locally | `polycodegpt-sandbox-python:v1`, `polycodegpt-sandbox-node:v1` |

The JavaScript sandbox needed a Node base image; none existed locally, so one was pulled
and this was reported. No other network access, package install or system change occurred.

## 2. Sandbox (Part A)

Each candidate runs in a **disposable container**, one per task, with:

```
--network none
--read-only  --tmpfs /tmp:rw,noexec,nosuid,nodev,size=16m
--user 10001:10001
--cap-drop ALL  --security-opt no-new-privileges:true
--memory 128m  --memory-swap 128m  --cpus 0.50  --pids-limit 32
--ulimit nofile=64:64  --ulimit fsize=1048576:1048576
```

No host directory is mounted. The payload travels over stdin; results come back over
stderr prefixed with a per-run random token, so a candidate cannot forge results or
corrupt the protocol with its own output. The host enforces a 10 s wall clock and
`docker kill`s the container; the container is removed after every run (with an inspect
before removal to record `OOMKilled`).

Python runs under `python3 -I` (isolated mode: no user site, no `PYTHON*` env). Node runs
candidate code inside `vm` with a 2 s per-evaluation budget, a second line of defence
behind the container kill.

## 3. Safety verification (Part F) — 16/16 probes passed

`reports/phase11_sandbox_security.md` / `.json`, produced **before** any model-generated
code was executed.

| property | observed |
|---|---|
| network (TCP) | `OSError [Errno 101] Network is unreachable` |
| network (DNS) | `gaierror [Errno -3] Temporary failure in name resolution`; JS `ENETUNREACH` |
| read-only root | `EROFS` / `EACCES` writing `/opt/harness`, `/etc` and `/` |
| host project files | not visible; `FOUND []` for `C:/Users`, `/c/Users`, `/mnt/c/Users`, `/workspace` |
| host mounts | `HOST_MOUNTS 0`; `/` mounted `ro` |
| privileges | `UID 10001`, `EUID 10001`, `CapEff 0000000000000000`, `NoNewPrivs 1` |
| timeout | infinite loops killed (host kill in Python, in-VM budget in Node) |
| memory | 512 MiB request under a 128 MiB cap → OOM-killed (exit 137) |
| processes | `fork` stops at ~6–10 processes with `BlockingIOError` (pids limit) |
| file size | 8 MiB write under a 1 MiB `fsize` → `OSError 27` |

The evaluator refuses to run if this report is missing or any probe failed
(`preflight_safety`), and a unit test asserts the committed report is fully green.

Residual risk is documented in the safety report: containers share the Docker Desktop VM
kernel, no custom seccomp profile or user-namespace remapping is used, and `/tmp` is
writable by design (16 MiB, `noexec`).

## 4. Specification and manifest (Part B)

- `reports/phase11_functional_spec.md` — 100 task specifications (50 Python, 50 JS) with
  entry point, required behaviour, normal cases, boundary cases and error behaviour.
- `functional/manifest.json` — machine-readable 100-task manifest, 420 cases
  (210 per language: 116 normal, 92 boundary, 2 error).
- `functional/spec/python_specs.py`, `functional/spec/javascript_specs.py` — specs are
  **positional** against `scripts/phase10_tasks.py`, so the docstrings/stubs the model saw
  are imported rather than retyped; a unit test proves the match.

Ambiguities are resolved once, recorded per task (`resolution`), and summarised in the
specification. The ones that affect scoring:

1. Miss value for searches → `-1` (Python and JS), flagged as *convention* cases.
2. Ordinal indexing → 1-based, except Fibonacci which is 0-based with F(0)=0.
3. Text predicates → case-sensitive unless stated (`same_word`/`sameWord` are not).
4. Empty/negative inputs → defined only where the docstring gives a basis; otherwise untested.
5. `countdown` returns a list/array, not printed output.
6. Error contracts → `ValueError` (Python), any `Error`/`RangeError`/`TypeError` (JS).
7. Negative remainder (`my_mod`/`myMod`) → untested, because the two languages disagree.
8. Nested `flatten` → fully recursive; `remove_duplicates`/unions → first-occurrence order.

Cases marked *convention* are counted in the primary success rate **and** reported
separately as a sensitivity metric, so the ambiguity is visible rather than hidden.

## 5. Trusted reference implementations (Part C)

`functional/reference/python_refs.py` and `functional/reference/javascript_refs.js`,
written from the specification — not from generated code. `functional/reference/validate_references.py`
executes them in the sandbox against every case:

```
reference validation: 100/100 tasks, 420/420 cases
```

Reference sources live in their own module and are never concatenated with, or shown to,
candidate code.

## 6. Evaluator (Parts D–E)

`functional/evaluate_functional.py` loads
`reports/phase10_generation_evaluation_20261117_153043.json` (created earlier, checkpoint
step 8130, seed 42), uses each task's complete `syntax_candidate`, and assigns exactly one
outcome per task:

| outcome | Python | JavaScript |
|---|---|---|
| Syntax failure (not executed) | 9 | 49 |
| Missing function / entry point | 0 | 0 |
| Runtime exception | 17 | 0 |
| Timeout | 0 | 0 |
| Resource-limit violation | 0 | 0 |
| Incorrect output | 24 | 1 |
| **Complete task success** | **0** | **0** |

Candidates with invalid syntax are never executed and never counted as correct.

### Measured metrics (all 50 tasks per language in the denominator)

| metric | Python | JavaScript |
|---|---|---|
| syntax-valid rate | 82.0% (41/50) | 2.0% (1/50) |
| functional task success rate | 0.0% (0/50) | 0.0% (0/50) |
| test-case pass rate | 6.2% (13/210) | 0.0% (0/210) |
| runtime-error rate | 34.0% | 0.0% |
| timeout rate | 0.0% | 0.0% |
| completion rate | 82.0% | 2.0% |
| edge-case success rate | 10.6% | 0.0% |
| success excluding convention cases | 0.0% | 0.0% |
| resource-limit rate | 0.0% | 0.0% |

Interpretation: the model produces source that parses reasonably often in Python (41/50)
but almost never in JavaScript (1/50), and **no candidate passes all of its task's cases**.
The 13 passing Python cases are coincidences where the generated body returns the input
unchanged and the expected value happens to equal it (e.g. `my_abs(5)`, `reverse_string("")`).

A task counts as successful only when every mandatory case passes; the convention-excluded
sensitivity metric is also 0/50, so no result hinges on an ambiguous convention.

## 7. Tests

| suite | result |
|---|---|
| `tests/unit/test_functional_spec.py` | 20 tests — manifest integrity, positional drift, ambiguity documentation, argument/arity checks, report content |
| `tests/unit/test_functional_evaluator.py` | 23 tests — every outcome category, metric denominators, convention sensitivity, infrastructure exclusion, Part F gate |
| `tests/unit/test_sandbox_runner.py` | 11 tests — container flags (no network, read-only, non-root, no capabilities, no mounts, limits) |
| `tests/integration/test_phase11_sandbox.py` | 16 tests — real container runs: success, wrong output, exception, missing entry, timeout, memory limit, host-write block, network block, non-root, reference oracle |

All 49 unit tests and all 16 integration tests pass.

## 8. Exact Windows CMD commands

```cmd
cd /d "C:\Users\UltraPc\Desktop\LLM FROM SCRATCH\PolyCodeGPT"

REM 1. Docker must be running with the Linux engine
docker desktop start
docker version --format "{{.Server.Os}}/{{.Server.Arch}}"

REM 2. Build both sandbox images (first run pulls node:22-alpine)
..\.venv\Scripts\python.exe functional\sandbox\runner.py --build

REM 3. Harness smoke test (both languages must print statuses=['pass','pass','pass'])
..\.venv\Scripts\python.exe functional\sandbox\runner.py --self-test

REM 4. Part F containment probes - must report 16/16
..\.venv\Scripts\python.exe functional\sandbox\probes.py

REM 5. Validate the 100-task manifest against the Phase 10 inventory
..\.venv\Scripts\python.exe functional\spec\build_manifest.py --check

REM 6. Validate reference implementations inside the sandbox
..\.venv\Scripts\python.exe functional\reference\validate_references.py

REM 7. Evaluate the frozen Phase 10 candidates
..\.venv\Scripts\python.exe functional\evaluate_functional.py

REM 8. Tests
..\.venv\Scripts\python.exe -m pytest tests\unit\test_functional_spec.py tests\unit\test_functional_evaluator.py tests\unit\test_sandbox_runner.py -q
..\.venv\Scripts\python.exe -m pytest tests\integration\test_phase11_sandbox.py -q
```

Step 4 must be run before step 7 in a fresh checkout: the evaluator refuses to execute
candidate code without a fully green safety report.

## 9. Constraints compliance

| constraint | evidence |
|---|---|
| no retraining | no training code executed; only `evaluate_functional.py` ran |
| no checkpoint changes | `checkpoints/polycodegpt_first_pass/step_008130.pt` present, `step == 8130`, file mtime unchanged (Nov 15) |
| no tokenizer changes | tokenizer files untouched; tokenizer not even loaded |
| no dataset modifications | data/ untouched; no stream or manifest written |
| no test-split access | evaluator reads only the Phase 10 report JSON |
| checkpoint step 8130 preserved | `best.pt`, `latest.pt`, `step_008130.pt` all report step 8130 |
| existing reports preserved | all new files; Phase 10/10F reports unmodified |
| no second generation run | no model loaded anywhere in Phase 11 |
| no candidate execution outside the sandbox | all execution through `functional/sandbox/runner.py`; probes confirm containment |

## 10. Implemented vs incomplete

**Implemented**

- Part A: Docker/WSL inventory, isolated environment design, two purpose-built images.
- Part B: 100 specifications, documented ambiguity resolutions, machine-readable manifest.
- Part C: independent reference implementations, validated 420/420 in the sandbox.
- Part D: evaluator over the official report with the full outcome taxonomy.
- Part E: per-language metrics with an explicit denominator policy and sensitivity check.
- Part F: 16 containment probes, run before any candidate execution, enforced by a gate.
- Part G: specification, manifest, references, sandbox config, tests, evaluator, safety
  report, CMD commands, this report.

**Incomplete or limited**

1. **JavaScript metrics rest on one executable candidate.** 49 of 50 JS candidates fail to
   parse, so the JS functional rates are technically correct but statistically meaningless.
   Improving JS generation is a model/training concern, out of scope here.
2. **One sample per task.** Results come from a single Phase 10 run (seed 42, T=0.8,
   top_k=50); no variance estimate is possible without another generation run, which is
   forbidden in this phase.
3. **Style-only requirements are not enforced** (`without len()`, `without abs()`,
   `without %`, `without Math.abs`, `without s.length`, "recursively"). Only observable
   behaviour is judged; a non-recursive `factorial` that returns correct values passes.
4. **Containment is process/namespace level**, sharing the Docker Desktop VM kernel. There
   is no custom seccomp profile, no user-namespace remapping, and Docker Desktop itself
   runs with the host user's privileges.
5. **Convention cases remain a judgement call.** They are reported separately (0/50 either
   way), but a different resolution for search-miss values would change individual task
   verdicts, never the headline result.
6. **No hidden-test generalisation check.** Everything is measured on these 100 tasks; no
   separate held-out functional set was created (the earlier test split was off-limits).
