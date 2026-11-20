# Phase 13 — Final comparison and candidate recommendation

Strictly read-only report. No training, checkpoint, tokenizer, dataset, manifest, script, evaluation report, or log was modified. Only this report was created. Reserved evaluation data (`functional/independent_v3/` and the old test partition) was not accessed.

## 1. Executive summary

All ten required JSON reports exist, are readable, and correspond to the correct checkpoint and task set (see §2). Independently recomputed metrics match the provisional terminal tables exactly; no discrepancies were found.

Generation and functional results remain weak overall, but the two trained branches split the top positions:

- **Exp B** has the most fully passed functional tasks: **2/50 Python, 0/50 JavaScript** (2/100 total).
- **Exp C** has the most passed functional cases: **34/420** (16/210 Python + 18/210 JavaScript), the only JavaScript fully passed task, and the strongest aggregate generation metrics (56% syntax validity, 45% EOS completion, 0.4623 unique-token ratio).
- **Exp A** leads Exp B in aggregate generation syntax validity (53% vs. 51%) but has **0 fully passed tasks** and fewer total passed cases (16/420) than Exp B (24/420) and Exp C (34/420).
- **Control** and **Original** are lower on both functional and generation metrics.

Under the stated selection hierarchy (§8), no checkpoint dominates both complete-task success and case-level success across both languages. **Exp B and Exp C are recommended as finalists for Phase 14.** Neither is release-ready, and no Phase 14 evaluation was run.

## 2. Checkpoint identity and evaluation provenance

All functional reports use the same 100 task identifiers (`py_01`…`py_50`, `js_01`…`js_50`): 100 unique IDs per report, identical sets across all five reports and the `functional/manifest.json` task set, with 50 Python + 50 JavaScript tasks. Candidate hashes in every functional report match the corresponding generation report's `syntax_candidate` hashes for all 100 tasks. All safety probes passed (16/16) and sandbox limits were identical (Python/Node images, 128m memory, 0.50 CPU, 32 pids, 10.0s timeout, no network).

| Label | Checkpoint | Checkpoint step | Generation report | Functional report | Candidate source in functional JSON |
|---|---|---:|---|---|---|
| Original | `checkpoints/polycodegpt_first_pass/step_008130.pt` | 8130 | `phase10_generation_evaluation_20261118_031153.json` | `phase13_original_functional.json` | `phase10_generation_evaluation_20261118_031153.json` |
| Control | `experiments/phase12b/runs/control/step_016260.pt` | 16260 | `phase10_generation_evaluation_20261118_031234.json` | `phase13_control_functional.json` | `phase10_generation_evaluation_20261118_031234.json` |
| Exp A | `experiments/phase12b/runs/exp_a/step_016260.pt` | 16260 | `phase10_generation_evaluation_20261118_031316.json` | `phase13_exp_a_functional.json` | `phase10_generation_evaluation_20261118_031316.json` |
| Exp B | `experiments/phase12b/runs/exp_b/step_016260.pt` | 16260 | `phase10_generation_evaluation_20261118_031356.json` | `phase13_exp_b_functional.json` | `phase10_generation_evaluation_20261118_031356.json` |
| Exp C | `experiments/phase12b/runs/exp_c/step_016260.pt` | 16260 | `phase10_generation_evaluation_20261118_031548.json` | `phase13_exp_c_functional.json` | `phase10_generation_evaluation_20261118_031548.json` |

Generation settings (identical for all five): device `cuda`, tokenizer v1.1.0, seed 42, temperature 0.0, max new tokens 64, 50 Python + 50 JavaScript tasks, greedy decoding, no EOS-appended prompt. Functional settings (identical for all five): 10.0s sandbox timeout, 16/16 safety probes, no infrastructure errors.

## 3. Generation comparison

Aggregate repetition ratios are means over all 100 tasks (not language-specific; no per-language repetition metrics were recorded). Max-token truncation is a derived count of tasks with `generated_token_count = 64` and no EOS hit; it is not a field recorded in the JSON.

| Label | Syntax valid (n/100) | Python syntax (n/50) | JS syntax (n/50) | EOS hits (n/100) | Python EOS (n/50) | JS EOS (n/50) | Empty (n/100) | Max-token truncations (derived, n/100) | Unique-token ratio (mean) | Repeated-line ratio (mean) | Repeated 4-gram ratio (mean) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original | 42 (42%) | 41 | 1 | 29 (29%) | 28 | 1 | 0 | 71 | 0.3910 | 0.1384 | 0.1265 |
| Control | 42 (42%) | 35 | 7 | 25 (25%) | 17 | 8 | 0 | 75 | 0.3833 | 0.1313 | 0.1376 |
| Exp A | 53 (53%) | 42 | 11 | 33 (33%) | 22 | 11 | 0 | 67 | 0.4184 | 0.1240 | 0.1203 |
| Exp B | 51 (51%) | 41 | 10 | 38 (38%) | 28 | 10 | 0 | 62 | 0.4351 | 0.1194 | 0.1029 |
| Exp C | 56 (56%) | 41 | 15 | 45 (45%) | 30 | 15 | 0 | 55 | 0.4623 | 0.1307 | 0.1107 |

Measured observations: all checkpoints are far stronger in Python syntax than JavaScript syntax. Exp C has the highest aggregate syntax validity, EOS completion, unique-token ratio, and fewest max-token truncations. Exp B has the lowest repeated-line and repeated 4-gram ratios. Exp A exceeds Exp B in aggregate syntax validity but trails in EOS completion, unique-token ratio, repetition, and truncation.

## 4. Functional task-success comparison

A task counts as successful only if every required test case passes. The raw JSON `success` outcome counts are used; no case-level partial success is treated as task success.

| Label | Python fully passed (n/50) | JS fully passed (n/50) | Total fully passed (n/100) | Python cases passed (n/210) | JS cases passed (n/210) | Total cases passed (n/420) |
|---|---:|---:|---:|---:|---:|---:|
| Original | 0 | 0 | 0 | 13 | 0 | 13 |
| Control | 0 | 0 | 0 | 7 | 3 | 10 |
| Exp A | 0 | 0 | 0 | 7 | 9 | 16 |
| Exp B | 2 | 0 | 2 | 19 | 5 | 24 |
| Exp C | 0 | 1 | 1 | 16 | 18 | 34 |

## 5. Functional case-pass comparison

Case pass rates (passed/total): Original 13/420 (3.10%), Control 10/420 (2.38%), Exp A 16/420 (3.81%), Exp B 24/420 (5.71%), Exp C 34/420 (8.10%). Python/JavaScript rates: Original 13/210 (6.19%) / 0/210 (0.00%); Control 7/210 (3.33%) / 3/210 (1.43%); Exp A 7/210 (3.33%) / 9/210 (4.29%); Exp B 19/210 (9.05%) / 5/210 (2.38%); Exp C 16/210 (7.62%) / 18/210 (8.57%).

Measured observations: Exp C is the only checkpoint with a JavaScript fully passed task and the highest JavaScript case count. Exp B has the highest Python case count and the only Python fully passed tasks. No checkpoint has both strong Python and JavaScript functional performance.

## 6. Error-category comparison

Outcome counts are task-level counts from the raw JSON. All 500 task outcomes across the five reports fall into the seven predefined categories; there are no timeout, resource-limit, or missing-entry outcomes, and no infrastructure errors.

| Label | Language | Syntax failure | Runtime error | Incorrect output | Timeout | Resource limit | Missing entry | Success |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Original | Python | 9 | 17 | 24 | 0 | 0 | 0 | 0 |
| Original | JavaScript | 49 | 0 | 1 | 0 | 0 | 0 | 0 |
| Control | Python | 15 | 25 | 10 | 0 | 0 | 0 | 0 |
| Control | JavaScript | 43 | 1 | 6 | 0 | 0 | 0 | 0 |
| Exp A | Python | 8 | 25 | 17 | 0 | 0 | 0 | 0 |
| Exp A | JavaScript | 39 | 0 | 11 | 0 | 0 | 0 | 0 |
| Exp B | Python | 9 | 20 | 19 | 0 | 0 | 0 | 2 |
| Exp B | JavaScript | 40 | 2 | 8 | 0 | 0 | 0 | 0 |
| Exp C | Python | 9 | 19 | 22 | 0 | 0 | 0 | 0 |
| Exp C | JavaScript | 35 | 0 | 14 | 0 | 0 | 0 | 1 |

Measured observations: JavaScript failures are dominated by syntax failures (70–98% of tasks). Python failures are split among runtime errors (38–50% of tasks) and incorrect outputs. Exp C has the lowest JavaScript syntax-failure count (35) and no JavaScript runtime errors; Exp B has slightly fewer Python runtime errors than Exp C (20 vs. 19) but more incorrect outputs (19 vs. 22 is the reverse: Exp B has fewer incorrect outputs than Exp C). Exp A has the highest Python runtime-error count (25) despite the highest Python syntax validity among the trained models.

## 7. Python-versus-JavaScript trade-offs

- All models are strongly Python-biased. Even Exp C, the best JavaScript model, reaches only 15/50 JavaScript syntax validity and 18/210 JavaScript case passes.
- Exp B is the strongest Python model by task completion (2 fully passed Python tasks, 19/210 Python cases) but weak in JavaScript (0 fully passed tasks, 5/210 cases).
- Exp C is the strongest JavaScript model (1 fully passed JavaScript task, 18/210 cases) and has the highest total case passes (34/420), but weaker Python case completion than Exp B (16/210 vs. 19/210).
- Exp A has the highest aggregate generation syntax validity among Exp A/B/C? No: Exp C leads aggregate syntax validity (56%), Exp A is second (53%). Exp A does lead Exp B in aggregate syntax validity (53% vs. 51%) and Python syntax (42 vs. 41), but has no fully passed tasks and fewer total passed cases than both Exp B and Exp C.
- Generation quality does not map directly to functional success: Control has the lowest aggregate syntax validity among trained models (42%) but the lowest total case passes (10/420); Exp A has higher syntax validity (53%) but only 16/420 case passes.
- Ranking changes by criterion: by fully passed tasks, Exp B (2) > Exp C (1) > all others (0); by total cases, Exp C (34) > Exp B (24) > Exp A (16) > Original (13) > Control (10); by JavaScript alone, Exp C leads every metric. The language imbalance must not be hidden in an aggregate score.

## 8. Model-selection criterion and rationale

Applied hierarchy:

1. **Fully passed functional tasks (Python and JavaScript separately):** Exp B leads (2 Python). Exp C is second (1 JavaScript). All others: 0. Exp B leads.
2. **Total passed functional cases (language-specific):** Exp C leads (34/420: 16 Python + 18 JavaScript). Exp B second (24/420: 19 + 5). Exp A 16, Original 13, Control 10.
3. **Runtime-error, incorrect-output, and syntax-failure rates:** Exp C has the lowest JavaScript syntax-failure rate (70% vs. 78–98%) and no JavaScript runtime errors. Exp B has fewer Python runtime errors than Exp C (20 vs. 19 is effectively tied; Exp C is one lower) and fewer incorrect outputs than Exp C (19 vs. 22). Exp A has the highest Python runtime-error rate among trained models. Mixed; no clear dominance.
4. **Generation syntax validity, EOS completion, and repetition metrics:** Exp C leads aggregate syntax validity (56%), EOS completion (45%), unique-token ratio (0.4623), and fewest max-token truncations (55). Exp B leads repetition ratios (0.1194 repeated-line, 0.1029 repeated 4-gram). Exp A is between them on syntax (53%) but lower on EOS/repetition/truncation.
5. **Validation loss (supporting evidence only):** Original overall validation loss 2.4462 (combined Python 2.4449, combined JavaScript 2.5293). Trained checkpoints: Exp A overall 2.2955 (Python 2.2724, JavaScript 2.4376); Exp B overall 2.3131 (Python 2.3473, JavaScript 2.2825); Exp C overall 2.3620 (Python 2.3679, JavaScript 2.4161); Control best sampled validation loss 2.22899 (no full per-language table reported). All trained checkpoints improve over Original, but Control has the lowest reported validation loss while being weakest on the development benchmark. Validation loss therefore does not resolve the finalist choice and is treated as supporting evidence only.

Rationale: Exp B owns the only complete-task successes in Python; Exp C owns the only complete-task success in JavaScript and the largest case-level total. Because neither dominates both criteria and the languages are imbalanced, carrying both forward preserves the trade-off for a final decision on the same untouched independent set.

## 9. Phase 14 recommendation

**Recommend carrying both Exp B and Exp C forward as finalists to Phase 14.** Phase 14 should evaluate both on the same untouched independent evaluation set before a final selection, using the same decoding, tokenizer, prompt construction, validators, and sandbox settings. Do not claim either model is release-ready. Do not begin Phase 14 or run any independent evaluation from this report.

Exp A is not recommended as a finalist under the stated hierarchy: although it exceeds Exp B in aggregate generation syntax validity, it has zero fully passed functional tasks and fewer total passed cases than both Exp B and Exp C. Control and Original are not recommended.

## 10. Limitations

- Results are from the Phase 10/11 development benchmark only (100 tasks, 420 cases). They are not estimates of independent-evaluation performance.
- Functional results depend on the fixed task definitions, case sets, sandbox images, and 10.0s timeout; generated code was executed only inside the existing Docker sandbox.
- Max-token truncation counts are derived from `generated_token_count = 64` with no EOS hit; the JSON does not record a separate truncation field.
- Repetition metrics are aggregate means over 100 tasks, not language-specific.
- Control has no full per-language validation-loss table in its audit; only the sampled best validation loss (2.22899) is reported.
- The provisional terminal tables matched the raw JSON exactly; no discrepancies were found.
- Reserved data was not accessed: `functional/independent_v3/` and the old test partition were not read, listed, inspected, or executed. No artifact other than this report was modified.

PHASE 13 COMPLETE — EXP_B AND EXP_C ADVANCE AS FINALISTS