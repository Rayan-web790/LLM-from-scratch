# Phase 12C Preflight — CUDA Smoke Test (Phase 12B completion)

Status: **Phase 12C is safe to start.** The runtime preflight passed. No full training
run was started; no checkpoint was written; the immutable checkpoint is byte-identical.

## 1. Dataset verification (Part B completion)

All four prepared datasets re-verified after preparation: sha256, token count, block
count, loader batch count, update count and language-token sums all match their
manifests. Sources are the phase-1 training partition only.

| branch | tokens | sha256 | blocks | updates (full + partial) | languages (tokens / records / epochs) |
|---|---:|---|---:|---|---|
| control | 133,192,957 | `6b8ffbf653e8a3b3…e021` | 260,142 | 8,129 + 1 | Python 98,807,136 / 290,463 / 1.000; JS 31,538,801 / 102,169 / 1.000; English 2,847,020 / 500 / 1.000 |
| exp_a | 133,192,950 | `37692f87f8c6…6150` | 260,142 | 8,129 + 1 | Python 98,807,136 / 290,463 / 1.000; JS 31,538,794 / 35,658 / 4.821; English 2,847,020 / 500 / 1.000 |
| exp_b | 133,192,915 | `eab16b96373c…895a` | 260,142 | 8,129 + 1 | Python 65,172,938 / 191,985 / 0.661; JS 65,172,957 / 102,169 / 2.065; English 2,847,020 / 500 / 1.000 |
| exp_c | 133,192,925 | `99c7077c570f…ffc55` | 260,142 | 8,129 + 1 | Python 65,172,938 / 191,985 / 0.661; JS 65,172,967 / 35,658 / 9.961; English 2,847,020 / 500 / 1.000 |

Sources per branch (from each `dataset_manifest.json`):

* control: `_python.bin`, `_javascript_original.bin`, `_english_docs.bin`
* exp_a: `_python.bin`, `_javascript_curated.bin`, `_english_docs.bin`
* exp_b: `_python.bin`, `_javascript_original.bin`, `_english_docs.bin`
* exp_c: `_python.bin`, `_javascript_curated.bin`, `_english_docs.bin`

No source path contains `test`, `valid`, or `independent_v3`. The old test split and
`independent_v3` were never opened. The 44 Python records present in
`codesearchnet_python_filtered/train.jsonl` but absent from
`data/processed/polycodegpt_corpus/train.jsonl` (67,996 characters) are excluded by
construction.

## 2. Smoke test mode

`scripts/run_phase12_experiment.py` gained `--smoke --confirm-smoke`, with
`--experiment control` fixed. It is gated exactly like `--train` and cannot be
combined with it. It builds the *existing* `Trainer` with `additional_steps=1`,
`complete_pass=False`, `checkpoint_dir=None` and no validation loader, while keeping
the production model, optimizer, precision, batch geometry (4 × 8 = 32 sequences),
and the 8,130-step schedule shape (warmup 163). It performs exactly one optimizer
update and writes no file.

## 3. Smoke test result (CUDA, BF16)

| field | value |
|---|---|
| date | 2026-11-17 |
| device | NVIDIA GeForce RTX 5060 Laptop GPU (8.5 GB), cc 12.0 |
| torch | 2.15.0+cu132, CUDA 13.2 |
| precision | BF16 (auto), scaler disabled |
| elapsed (cold single update) | 7.064 s |
| peak VRAM allocated | 1,148.4 MiB |
| peak VRAM reserved | 1,318.0 MiB |
| update | 8 micro-batches, 32 sequences, 16,384 tokens |
| training loss | 2.454036 |
| gradient norm (pre-clip) | 0.811556 |
| learning rate used | 1.840491e-06 |
| parameters changed | 100 / 100 (max \|Δ\| 5.944e-06) |
| optimizer step | 8130 → 8131 |
| global step | 8130 → 8131 |
| stage step | 0 → 1 |
| scheduler last_step | 0 → 1 |
| tokens_seen (cumulative) | 133,192,704 → 133,209,088 |
| checkpoint sha256 before | `5562acc2fd58837f9d4245042b7f0b10de65520aa2ba292999c80045470590bd` |
| checkpoint sha256 after | `5562acc2fd58837f9d4245042b7f0b10de65520aa2ba292999c80045470590bd` |

All 14 checks passed:

```
[PASS] checkpoint_sha_unchanged
[PASS] checkpoint_dir_unchanged
[PASS] no_checkpoint_written
[PASS] exactly_one_update
[PASS] loss_finite
[PASS] grad_norm_finite
[PASS] parameters_updated
[PASS] optimizer_state_restored
[PASS] optimizer_state_advanced
[PASS] adamw_moments_updated
[PASS] global_step_advanced
[PASS] stage_step_advanced
[PASS] accumulation_used
[PASS] scheduler_advanced
```

A second identical run produced the same loss (2.454036), gradient norm (0.8116) and
peak VRAM (1,148.4 MiB): on this GPU and build the single update is bitwise
reproducible with the pinned seeds. The unavoidable-nondeterminism caveat in the
determinism report is therefore a known risk rather than a measured failure here.

## 4. Evidence files

* `reports/phase12b_smoke_test_cuda.json` — full machine-readable report.
* `reports/phase12b_final_integration.md` — Phase 12B integration record.

## 5. Tests run (venv interpreter)

```
"C:\Users\UltraPc\Desktop\LLM FROM SCRATCH\.venv\Scripts\python.exe" -m pytest tests/unit/test_experiment_runner.py -q
20 passed in 36.65s
```

New tests: `test_smoke_requires_its_own_confirmation`,
`test_smoke_and_train_cannot_be_combined`,
`test_smoke_refuses_a_non_control_dataset`,
`test_smoke_settings_are_single_update_and_cannot_write`,
`test_smoke_run_on_cpu_performs_exactly_one_verified_update`.

A CPU smoke run (FP32) also passed in 22.4 s with loss 2.4543 / grad 0.835, which
confirms the mode is device-independent.

## 6. Exact smoke-test command

```
"C:\Users\UltraPc\Desktop\LLM FROM SCRATCH\.venv\Scripts\python.exe" ^
  scripts\run_phase12_experiment.py --experiment control --smoke --confirm-smoke
```

Add `--device cpu` to run it without CUDA. No `--train` flag is involved.

## 7. Conclusion

Phase 12C is safe to start: the immutable checkpoint, original datasets, tokenizer and
independent benchmark v3 are unchanged; all four experiment datasets are materialized
and verified; the real engine runs one correct, finite, deterministic update on the
real data at 1,148 MiB peak VRAM; and the stage accounting (8,130 updates,
16,384 tokens/update, global step 8130 → 16,260) is unchanged. Training has not
been started.
