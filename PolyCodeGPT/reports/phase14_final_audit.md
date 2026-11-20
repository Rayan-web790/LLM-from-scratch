# Phase 14 — Final independent evaluation audit and model selection

Strictly read-only audit. No training, generation, or evaluation was rerun. No frozen artifact, checkpoint, report, manifest, or suite file was modified. The old test partition was not accessed. This report is the only new file.

## Audited reports

| Report | Size | SHA256 |
|---|---:|---|
| `reports/phase14_exp_b_generation.json` | 85,188 bytes | `216f99dcef549d81c921ecc61d75504c0db3394dacc1fbcda202b06782db8da4` |
| `reports/phase14_exp_c_generation.json` | 84,642 bytes | `842c33b75dd39af81ed29b426d52259864507646f47a0ce6ec3770637134bbac` |
| `reports/phase14_exp_b_functional.json` | 132,511 bytes | `c59bfe81f230a8a9e9ed17ab1f286678927ade2be29cf71f26c271557f76461f` |
| `reports/phase14_exp_c_functional.json` | 131,294 bytes | `141a27a50591cae1d568cb71182fec8d4ab05f5f55439804570a9a36d6059176` |

The functional reports identify their candidate sources as:

- Exp B: `reports\phase14_exp_b_generation.json`
- Exp C: `reports\phase14_exp_c_generation.json`

## 1. Independently recomputed results

Results were recomputed directly from each functional report's `task_outcomes` and each task's `cases` array. A task was counted as complete only when every manifest case for that task had status `pass`.

| Metric | Exp B | Exp C | Supplied expectation | Status |
|---|---:|---:|---:|---|
| Python complete tasks | 2/50 | 3/50 | 2/50 and 3/50 | Match |
| JavaScript complete tasks | 0/50 | 1/50 | 0/50 and 1/50 | Match |
| Total complete tasks | 2/100 | 4/100 | 2/100 and 4/100 | Match |
| Python cases passed | 18/164 | 27/164 | 18/164 and 27/164 | Match |
| JavaScript cases passed | 2/164 | 12/164 | 2/164 and 12/164 | Match |
| Total cases passed | 20/328 | 39/328 | 20/328 and 39/328 | Match |

### Complete-task rates

| Language | Exp B | Exp C | Difference |
|---|---:|---:|---:|
| Python | 2/50 = 4.00% | 3/50 = 6.00% | +2.00 pp |
| JavaScript | 0/50 = 0.00% | 1/50 = 2.00% | +2.00 pp |
| Total | 2/100 = 2.00% | 4/100 = 4.00% | +2.00 pp |

### Case-pass rates

| Language | Exp B | Exp C | Difference |
|---|---:|---:|---:|
| Python | 18/164 = 10.98% | 27/164 = 16.46% | +5.49 pp |
| JavaScript | 2/164 = 1.22% | 12/164 = 7.32% | +6.10 pp |
| Total | 20/328 = 6.10% | 39/328 = 11.89% | +5.79 pp |

No discrepancy was found between the raw JSON and the supplied expected counts.

### Task-outcome reconciliation

| Outcome | Exp B Python | Exp B JS | Exp B total | Exp C Python | Exp C JS | Exp C total |
|---|---:|---:|---:|---:|---:|---:|
| `success` | 2 | 0 | 2 | 3 | 1 | 4 |
| `syntax_failure` | 11 | 32 | 43 | 10 | 27 | 37 |
| `runtime_error` | 17 | 1 | 18 | 19 | 0 | 19 |
| `incorrect_output` | 20 | 17 | 37 | 18 | 22 | 40 |
| `timeout` | 0 | 0 | 0 | 0 | 0 | 0 |
| `resource_limit` | 0 | 0 | 0 | 0 | 0 | 0 |
| `missing_entry` | 0 | 0 | 0 | 0 | 0 | 0 |
| Any other task outcome | 0 | 0 | 0 | 0 | 0 | 0 |
| **Total** | **50** | **50** | **100** | **50** | **50** | **100** |

Both reports compute to exactly 50 tasks per language and 100 tasks total. No task-level `infrastructure_error` outcome is present; both reports record zero infrastructure errors.

### Case-status reconciliation

| Case status | Exp B Python | Exp B JS | Exp B total | Exp C Python | Exp C JS | Exp C total |
|---|---:|---:|---:|---:|---:|---:|
| `pass` | 18 | 2 | 20 | 27 | 12 | 39 |
| `fail` | 55 | 54 | 109 | 45 | 61 | 106 |
| `exception` | 55 | 4 | 59 | 61 | 0 | 61 |
| `not_run` | 36 | 104 | 140 | 31 | 91 | 122 |
| **Total** | **164** | **164** | **328** | **164** | **164** | **328** |

The additional categories present at case level are `fail`, `exception`, and `not_run`. They are case-level statuses, not additional task outcomes.

### Success-rule consistency

The recomputed all-cases-pass set matches the `success` outcomes exactly:

- Exp B: `py_v2_01`, `py_v2_21` — 2 tasks.
- Exp C: `py_v2_01`, `py_v2_21`, `py_v2_25`, `js_v2_25` — 4 tasks.

No task marked `success` had a non-passing case. No task with every case passing was marked as anything other than `success`.

## 2. Provenance and integrity

### Checkpoint and tokenizer identities

| Artifact | Actual SHA256 | Exp B embedded | Exp C embedded | Result |
|---|---|---|---|---|
| Exp B checkpoint `experiments/phase12b/runs/exp_b/step_016260.pt` | `fc70f5264a97d22c162986e172f8e70881713d5ae235c992e516a91ff67f5db6` | `fc70f5264a97d22c162986e172f8e70881713d5ae235c992e516a91ff67f5db6` | — | Match |
| Exp C checkpoint `experiments/phase12b/runs/exp_c/step_016260.pt` | `cf0544099d73f85cdeb40bc1e0a93ce17bb790db4cb7136254be65d16c3b45b4` | — | `cf0544099d73f85cdeb40bc1e0a93ce17bb790db4cb7136254be65d16c3b45b4` | Match |
| Tokenizer `tokenizer/artifacts/polycodegpt_tokenizer_v1_1.json` | `0459f8556a6f79ca1ba1c3dc3f76c26ae2e55653ebbf97beaa9bd23467358a03` | `0459f8556a6f79ca1ba1c3dc3f76c26ae2e55653ebbf97beaa9bd23467358a03` | `0459f8556a6f79ca1ba1c3dc3f76c26ae2e55653ebbf97beaa9bd23467358a03` | Match |

Both generation reports record checkpoint step 16260, seed 42, device `cuda`, tokenizer version `v1.1.0`, and the correct tokenizer path.

### Independent manifest

| Check | Value | Result |
|---|---|---|
| Actual manifest SHA256 | `2a7fbf3ddc43f6a2fee368d6faa711dbe8fa841343c7cf9507c919cd3baeb46c` | Match |
| Exp B `manifest_sha256` | `2a7fbf3ddc43f6a2fee368d6faa711dbe8fa841343c7cf9507c919cd3baeb46c` | Match |
| Exp C `manifest_sha256` | `2a7fbf3ddc43f6a2fee368d6faa711dbe8fa841343c7cf9507c919cd3baeb46c` | Match |
| Manifest path in both functional reports | `functional\independent_v3\manifest.json` | Match |

The manifest contains 100 tasks, 100 unique task IDs, 50 Python tasks, 50 JavaScript tasks, 164 Python cases, 164 JavaScript cases, and 328 cases total.

### Frozen suite files

The freeze manifest SHA256 is:

`ec36c8803f5c22692a9b789dda4315a4f1c9901e08dc80ac09e435d3a065997e`

All seven listed files hash-match:

| Frozen file | Result |
|---|---|
| `functional/independent_v3/spec/python_specs.py` | OK |
| `functional/independent_v3/spec/javascript_specs.py` | OK |
| `functional/independent_v3/spec/build_manifest.py` | OK |
| `functional/independent_v3/manifest.json` | OK |
| `functional/independent_v3/reference/python_refs.py` | OK |
| `functional/independent_v3/reference/javascript_refs.js` | OK |
| `functional/independent_v3/reference/validate_references.py` | OK |

### Candidate-to-task matching

For both finalists:

- The functional report contains 100 task outcomes with 100 unique IDs.
- Task IDs exactly match the independent manifest: no missing, duplicate, unknown, or extra IDs.
- Every task's language, entry, and docstring matches its manifest task.
- Every task has the same number of cases as its manifest task.
- Every `candidate_source` exactly matches the corresponding generation report's `syntax_candidate`.
- Every recomputed SHA256 of `candidate_source` matches the functional report's `candidate_sha256`.
- Candidate source report paths match the corresponding Exp B or Exp C generation report.

### Sandbox evidence

- Existing containment evidence: **16/16 probes passed**.
- Exp B functional report safety probes: **16/16 passed**.
- Exp C functional report safety probes: **16/16 passed**.
- Both reports record `network: none`, 128 MB memory, 0.50 CPUs, 32 PIDs, nofile 64, fsize 1,048,576, and a 10.0-second timeout.

## 3. Selection rationale

Exp C is the leading candidate based on the primary functional metrics.

### Primary comparison

| Metric | Exp B | Exp C | Direction |
|---|---:|---:|---|
| Python complete tasks | 2/50 | 3/50 | Exp C +1 |
| JavaScript complete tasks | 0/50 | 1/50 | Exp C +1 |
| Total complete tasks | 2/100 | 4/100 | Exp C +2 |
| Python cases passed | 18/164 | 27/164 | Exp C +9 |
| JavaScript cases passed | 2/164 | 12/164 | Exp C +10 |
| Total cases passed | 20/328 | 39/328 | Exp C +19 |

Exp C improves both language tracks:

- Python complete tasks rise from 2 to 3, and Python cases rise from 18 to 27.
- JavaScript complete tasks rise from 0 to 1, and JavaScript cases rise from 2 to 12.

### Failure-category comparison

| Category | Exp B | Exp C | Change |
|---|---:|---:|---:|
| Syntax failures | 43 | 37 | −6 |
| Runtime errors | 18 | 19 | +1 |
| Incorrect outputs | 37 | 40 | +3 |

Exp C reduces syntax failures by six overall, including one fewer Python syntax failure and five fewer JavaScript syntax failures. It has one more runtime error and three more incorrect outputs overall. Those regressions do not outweigh its larger gains in complete tasks and passed cases.

### Generation metrics as supporting evidence

| Generation metric | Exp B | Exp C |
|---|---:|---:|
| Syntax-valid generations | 57/100 | 63/100 |
| Python syntax-valid | 39/50 | 40/50 |
| JavaScript syntax-valid | 18/50 | 23/50 |
| EOS hits | 50/100 | 59/100 |
| Mean unique-token ratio | 0.4708 | 0.5121 |
| Mean repeated-line ratio | 0.1144 | 0.0995 |
| Mean repeated-4-gram ratio | 0.1138 | 0.1108 |

Exp C produces more syntax-valid generations, more EOS-terminated generations, higher token diversity, and fewer repeated lines. These generation-level indicators are consistent with Exp C's stronger functional results.

### Recommendation

**Exp C is selected as the leading candidate.** It improves both Python and JavaScript complete-task counts and passed-case counts over Exp B. The improvement is consistent across the primary metrics and supported by better syntax validity and generation-diversity indicators.

This is a relative selection between two finalists. It is not a claim of release readiness. Functional success remains low at 4/100 complete tasks and 39/328 passed cases.

## 4. Limitations

- The contamination assessment was preliminary lexical evidence, not proof of full dataset independence. It does not establish semantic independence or rule out all hidden overlap.
- The benchmark contains 100 tasks and is not proof of general programming ability.
- No training or tuning was performed using Phase 14 results.
- This audit is static and did not rerun generation or functional evaluation. It verifies the recorded results and their provenance, not reproducibility of the execution environment.
- No existing report, checkpoint, tokenizer, manifest, suite file, or sandbox configuration was modified.

## Final status

All supplied counts reconcile exactly with the raw JSON. Provenance, manifest identity, frozen-suite hashes, candidate matching, and sandbox evidence are consistent. Exp C leads on both language tracks and on total complete tasks and passed cases.

PHASE 14 COMPLETE — EXPC SELECTED AS LEADING CANDIDATE