# Phase 12B — Final Experiment Preflight and Evaluation Freeze

Phase 12A preparation is complete; this phase verifies comparability and freezes
the independent benchmark. No training has been started. No checkpoint, dataset,
tokenizer or report has been modified.

## 1. Configuration comparison (Part A)

All four branches: base checkpoint step 8130 (sha256 `5562acc2…90bd`), AdamW,
lr base 3e-4 with a fresh cosine schedule (reset_schedule, warmup 163, 8130 steps),
micro-batch 4 x 8 (effective 32), grad clip 1.0, context 512, seed 42, deterministic
flag set. Optimizer moments are carried over (`reset_optimizer_step: false`); only the
LR curve is reset. No config silently resets optimizer or scheduler state.
Full audit: `reports/phase12b_config_audit.md`.

| branch | config | JS data | Python tokens | JS tokens |
|---|---|---|---:|---:|
| control | control.yaml | original 102,169 | 98,807,136 | 31,538,801 |
| A | exp_a_curated_js_ratio.yaml | curated, oversampled | 98,807,136 | 31,538,801 |
| B | exp_b_language_balance.yaml | original, balanced | 65,172,968 | 65,172,968 |
| C | exp_c_curated_balanced.yaml | curated, balanced | 65,172,968 | 65,172,968 |

Token budget per branch: 130,345,937 (measured). Derived sampling figures:
`reports/phase12b_token_sampling.json`. Exp A revisits each curated JS record
~4.82x; Exp C ~9.96x; unique JS coverage is all 35,658 curated records in both.

## 2. Language exposure verification (Part B)

- Control: original distribution, Python:JS token ratio 3.13:1.
- Exp A: curated JS oversampled with replacement to the original JS token volume;
  ratio preserved at 3.13:1.
- Exp B: original records resampled to 50/50 tokens (Python subsampled to 0.66 of
  its records by token, JS oversampled 2.07x).
- Exp C: curated JS oversampled 9.96x to half the budget, Python subsampled to 0.66.
- Effective epochs: Python ~0.66 in B/C; JS ~2.07 (B) / ~4.82 (A) / ~9.96 (C).
- Class/repository distribution: curated set spans 8,341 repos (top repo 1.6%); no
  single repo dominates.

## 3. Curated JavaScript quality (Part C)

Audit: `reports/phase12b_curated_quality.json`. 35,658 records, 8,341 repos.
- Parser validity: 100% (parse-invalid records were excluded at curation time).
- Exact duplicates: 0; near-duplicates: 0 (72 were removed during curation).
- Feature usage: arrow functions 12.9%, array methods 16.6%, promises 3.5%,
  async 2.3%, try/catch 4.0%, classes 0.3%, exports 0.01%.
- Descriptions: all plain natural language (CodeSearchNet JS style); no JSDoc.
- Over-restrictive rules: filtering `this` removed 9,388 records and ecosystem
  dependencies 3,934 records (require 1,980, host globals 10,658). These are
  defensible for self-contained function completion but may discard legitimate
  module-level helpers; flagged as an unresolved risk, not changed (dataset is
  frozen).

## 4. Independent benchmark freeze (Part D)

Two tiers are now frozen:

- **v1 (pilot)** — `functional/independent/`: 20 tasks (10 Python, 10 JavaScript),
  88 cases. Classified as a PILOT: function names are disjoint from Phase 11 and
  the corpus (`reports/phase12b_contamination.json`, `contamination_found: false`)
  and refs validate 20/20, 88/88, but its scale is too small for a final claim.
  Hashed in `functional/independent/freeze_manifest.json`. Retained for continuity.

- **v2 (superseded/provisional)** — `functional/independent_v2/`: 100 tasks (50 Python, 50 JavaScript),
  328 cases, each with >=2 normal cases plus a boundary/error case. Entry names,
  behaviors, and reference code are disjoint from Phase 11 and the training corpus
  (contamination check: `reports/phase12b_contamination_v2.json`,
  `contamination_found: false`). Reference solutions validated in the sandbox:
  100/100 tasks, 328/328 cases. Frozen artifacts hashed in
  `functional/independent_v2/freeze_manifest.json` (manifest, spec, refs, validator).
  Not run against checkpoint 8130 or any experimental model during selection.

- **v3 (final)** — `functional/independent_v3/`: 100 tasks (50 Python, 50 JavaScript),
  328 cases, Python and JavaScript case lists are distinct (0 of 50 identical).
  Entry names, behaviors, and reference code are disjoint from Phase 11 and the
  training corpus (`reports/phase12b_contamination_v3.json`,
  `contamination_found: false`). Reference solutions validated in the sandbox:
  100/100 tasks, 328/328 cases. Frozen with SHA-256 in
  `functional/independent_v3/freeze_manifest.json`.

The v3 benchmark is the final independent benchmark for Phase 14.

## 5. Resource readiness (Part E)

See `reports/phase12b_resource_readiness.md`. Per branch: 2-4 GB VRAM, ~1 GB disk,
0.5-2.0 h GPU. All four: 2-8 h sequential, ~4 GB checkpoints. Estimates only.

## 6. Unresolved risks
- Deterministic-algorithm flags are not encoded in the YAML; the run entry point
  must set them.
- `this`/ecosystem filtering may be over-restrictive (see §3).
- Oversampling with replacement (Exp A/C) duplicates curated records; unique
  coverage is preserved but effective epochs rise to ~4.8 (A) and ~10 (C).
- The v1 pilot (20 tasks) is retained for continuity only; v2 (100 tasks) is the
  final independent benchmark.
- Validation loss uses the fixed 64-block subset; no per-task functional signal
  during training (by design).

## 7. Exact Windows CMD commands for future training runs (do not run now)

```cmd
cd "C:\Users\UltraPc\Desktop\LLM FROM SCRATCH\PolyCodeGPT"

:: Verify baseline before any run
..\..venv\Scripts\python.exe experiments\phase12a\baseline_manifest.py

:: Control branch
..\..venv\Scripts\python.exe scripts\train_first_pass.py ^
  --resume checkpoints\polycodegpt_first_pass\step_008130.pt ^
  --max-steps 8130 --total-steps 8130 ^
  --config configs\experiments\control.yaml ^
  --checkpoint-dir checkpoints\phase12b_control

:: Experiment A
..\..venv\Scripts\python.exe scripts\train_first_pass.py ^
  --resume checkpoints\polycodegpt_first_pass\step_008130.pt ^
  --max-steps 8130 --total-steps 8130 ^
  --config configs\experiments\exp_a_curated_js_ratio.yaml ^
  --checkpoint-dir checkpoints\phase12b_exp_a

:: Experiment B
..\..venv\Scripts\python.exe scripts\train_first_pass.py ^
  --resume checkpoints\polycodegpt_first_pass\step_008130.pt ^
  --max-steps 8130 --total-steps 8130 ^
  --config configs\experiments\exp_b_language_balance.yaml ^
  --checkpoint-dir checkpoints\phase12b_exp_b

:: Experiment C
..\..venv\Scripts\python.exe scripts\train_first_pass.py ^
  --resume checkpoints\polycodegpt_first_pass\step_008130.pt ^
  --max-steps 8130 --total-steps 8130 ^
  --config configs\experiments\exp_c_curated_balanced.yaml ^
  --checkpoint-dir checkpoints\phase12b_exp_c

:: Re-validate the independent benchmark (no model execution)
..\..\venv\Scripts\python.exe functional\independent_v2\reference\validate_references.py
..\..\venv\Scripts\python.exe functional\independent_v2\spec\build_manifest.py --check
..\..\venv\Scripts\python.exe experiments\phase12b\contamination_check_v3.py
```

Note: `scripts/train_first_pass.py` does not currently accept a `--config` flag;
the flags above document the intended invocation. The run entry point must be
updated (or a thin wrapper added) before launch. No training is performed in
Phase 12B.

## Conclusion
The four experiments are demonstrably comparable (identical optimizer, batch,
clip, seed, token budget, schedule shape; only the data distribution differs).
The independent benchmark v3 (100 tasks, 328 cases) is frozen and contamination-free,
and all four branches validate OK via `validate_config.py`. Phase 12B is complete.
No training was started.
