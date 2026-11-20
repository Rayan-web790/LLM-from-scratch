# Phase 5B — Training Engine and 200-Step Pilot Report

**Project:** PolyCodeGPT
**Run started:** 2026-11-15T18:22:41Z
**Command:**

```
C:/Users/UltraPc/Desktop/LLM FROM SCRATCH/.venv/Scripts/python.exe scripts/train.py --mode pilot --max-steps 200
```

**Verdict:** **PILOT TRAINED AND VERIFIED**

This is a 200-optimizer-step pilot (2.46 % of one pass over the training split), not a
pretraining run. No pretrained weights were loaded, no synthetic examples were used, and
`test.bin` was never opened.

---

## 1. Model architecture

Decoder-only Transformer implemented from scratch (`src/polycodegpt/model/`), trained from
random initialisation:

| Property | Value |
|---|---|
| Layers | 6 pre-LayerNorm blocks (`ln1` → attention → residual, `ln2` → FFN → residual) |
| Attention | 6 heads, head dim 64, causal scaled dot-product, from scratch |
| d_model | 384 |
| FFN | 1536, GELU |
| Context length | 512 |
| Vocabulary | 16,000 (tokenizer v1.1.0, verified SHA-256) |
| Positional encoding | learned absolute embeddings (512 × 384) |
| Output | `lm_head` tied to `token_embedding` (verified before and after every save/reload) |
| Dropout | 0.1 (embeddings, attention, residual, FFN) |
| Init | normal(0, 0.02); residual projections scaled by 0.02/√(2·6); biases 0; LayerNorm 1/0 |
| Params | 16,988,160 |

Parameter breakdown: token embeddings 6,144,000 · position embeddings 196,608 · 6 × 1,774,464
blocks (attention 591,360 + FFN 1,181,568 + LayerNorms 1,536) · final LayerNorm 768.

## 2. Actual parameter count

```
model: 16,988,160 parameters (analytical 16,988,160)
```

Counted from `model.parameters()` and cross-checked against `ModelConfig.estimate_parameters()`.
Optimizer grouping: **16,957,440 decay** params in 38 tensors + **30,720 no-decay** params in
62 tensors = 16,988,160 (the tied embedding/projection is counted once).

## 3. Training configuration

From `configs/training.yaml` (values passed through, no hardcoded constants):

| Setting | Value |
|---|---|
| Optimizer | AdamW, betas (0.9, 0.999), eps 1e-8, `zero_grad(set_to_none=True)` |
| Learning rate | 3e-4 (peak) |
| Weight decay | 0.1 on dim ≥ 2 weights; 0.0 on biases and LayerNorm |
| Gradient clip | 1.0 (global norm, applied after unscaling) |
| Micro-batch | 4 sequences × 512 tokens |
| Gradient accumulation | 8 micro-batches |
| Planned schedule | 8,129 optimizer steps, warmup 163 steps (2 %), cosine to 10 % of base (3e-5) |
| Pilot length | 200 optimizer steps |
| Seed | 42 (recorded; also used for the train loader shuffle and validation subset) |
| Validation | fixed subset of 64 blocks (32,768 tokens), seed 42, at steps 0/50/100/150/200 |
| Checkpoints | steps 50/100/150/200 + `latest.pt` + `best.pt` |

## 4. Precision

`torch.cuda.is_bf16_supported()` → True, so **BF16 autocast** was selected automatically
(requested `auto`). No `GradScaler` in the BF16 path; the FP16 + scaler path exists and is
unit-tested but was not needed. FP32 was not used for the pilot. There was no CPU fallback:
`require_cuda()` raises rather than silently switching device.

## 5. Effective batch size

```
4 sequences × 8 accumulation steps = 32 sequences per optimizer update
32 × 512 = 16,384 supervised tokens per optimizer update
```

The serialized stream uses `<|eos|>` as a separator, so **no loss masking** is applied: all
16,384 targets per step are supervised.

## 6. Actual GPU memory usage

Full-context forward + backward on real tokenized blocks, before training:

| Attempt | micro | accum | result | allocated | peak |
|---|---|---:|---|---:|---:|
| 1 | 4 | 8 | OK | 196.77 MiB | 949.11 MiB |

No OOM occurred, so no fallback rung (2×16, 1×32) was needed. During the 200-step run the
peak allocated memory was **1,210.7 MiB** (including the CUDA context and allocator
fragmentation); reported per-row in `logs/training/pilot_metrics.csv`.

Device: NVIDIA GeForce RTX 5060 Laptop GPU, 7.96 GiB, sm_120, torch 2.15.0+cu132 / CUDA 13.2.

## 7. Successful optimizer steps

```
200 optimizer steps, 0 skipped, 1,600 micro-batches
```

Steps are counted only after `optimizer.step()` actually updated the parameters and the
scheduler advanced. `scheduler.last_step == step == 200` at the end. No micro-batch count is
reported as a step anywhere.

## 8. Processed tokens

```
3,276,800 tokens = 200 × 16,384
6,400 blocks (sequences) = 2.46 % of one pass over the training split (133,192,957 tokens)
```

Each optimizer step consumed 32 blocks; every block was 512 supervised tokens.

## 9. Initial training loss

```
step 1  training loss 9.774015
mean of steps 1-10     9.613421
```

Consistent with a uniform-ish random model over 16,000 tokens (ln 16000 = 9.680).

## 10. Final training loss

```
step 200 training loss 5.665087
mean of steps 191-200  5.748412
```

Decrease of **4.109 (−42.0 %)** from step 1. The last-step value sits ~0.08 below the
10-step mean, which is normal step-to-step noise; both are reported rather than cherry-picking
the lower number.

## 11. Initial validation loss

```
9.791127  (perplexity 17,874.44)
```

Measured before any optimizer step, on the fixed 64-block subset (32,768 tokens).

## 12. Validation loss at each evaluation

Same 64 blocks (identical indices, identical order) at every evaluation:

| Step | Validation loss | Perplexity |
|---:|---:|---:|
| 0 | 9.791127 | 17,874.44 |
| 50 | 8.181973 | 3,575.90 |
| 100 | 6.756120 | 859.30 |
| 150 | 6.179685 | 482.84 |
| 200 | 5.873116 | 355.35 |

Monotonic decrease at every checkpoint. Training loss fell slightly faster than validation
loss (42.0 % vs 40.0 % on the subset), as expected when 2.46 % of the data has been seen; no
overfitting is visible yet at this scale.

## 13. Final complete validation loss

Full pass over **all 15,150 validation blocks** (7,756,800 tokens, 1,894 batches):

```
validation loss 5.815269   (token-weighted, 67.9 s)
```

Token-weighted: Σ(loss_batch × valid_tokens) / Σ(valid_tokens), not a mean of batch means.
The full-split result (5.815) is marginally below the 64-block subset at the same step
(5.873); both are within run-to-run noise on different data. `test.bin` was not touched.

## 14. Validation perplexity

```
PPL = exp(5.815269) = 335.38        (full validation split)
PPL = exp(5.873116) = 355.35        (fixed subset, step 200)
PPL = 17,874.44                     (same subset, step 0)
```

## 15. Gradient health

* 200/200 gradient norms finite; `train_step` raises `FloatingPointError` on a non-finite
  loss or a non-finite gradient norm, so no checked value could be inf/NaN.
* Norm range **0.667 – 4.772**, mean **1.258**, clip 1.0 applied every step (pre-clip norms
  recorded in the CSV).
* 0 optimizer updates skipped (BF16 path has no scaler-driven skip).
* All 100 parameter tensors finite after training; no NaN/Inf in any recorded loss.

## 16. Checkpoints

Directory `checkpoints/polycodegpt_pilot/`, atomic writes (`.partial` + `os.replace`, no
leftover temporary files):

| File | Step | Bytes | SHA-256 |
|---|---:|---:|---|
| `step_000050.pt` | 50 | 204,017,293 | `7942a44ee5caf72c7a859ace4ec7377cd99a7397041839cdcbc9c8701b70ce65` |
| `step_000100.pt` | 100 | 204,027,085 | `2d3ab842210aff1a21e65d1f3177198cb392aa4485913a7f0ce8a4e7a2e6918a` |
| `step_000150.pt` | 150 | 204,036,813 | `0ebc9cbb10df6f350fe1e2c34feadacdc063eb94e01ae89d3487a9f543f35450` |
| `step_000200.pt` | 200 | 204,046,605 | `b7c8898811d503e130fda8bddaca325236d9c7920ea2cf6e7b66e7fa12577b17` |
| `latest.pt` | 200 | 204,044,619 | `5ccce4ca69efa98252f4581c290204ec2da7bd6a0c0871a25e4230fc0e03d008` |
| `best.pt` | 200 (lowest val) | 204,043,223 | `ef16a53a2cd0ed29a0b4cc6e30ec0c9c3927a602e33185c186f866cc48535931` |

Each checkpoint holds: model `state_dict`, optimizer `state_dict` (AdamW moments), scheduler
`state_dict`, scaler state (empty for BF16), global step, tokens seen, micro-batches,
model config, training settings, tokenizer SHA-256, data-manifest SHA-256, seeds and RNG
states (torch/python/numpy/CUDA/loader), metrics, validation history and the best loss.

Loading uses `torch.load(..., weights_only=True)`; reload verified:

```
checkpoint reload: weights_match=True tied=True
```

`model.lm_head.weight is model.token_embedding.weight` holds after restoration (checked in the
run and again in the unit/integration tests).

**Resume verified on the real checkpoint:** restoring `latest.pt` into a fresh model continued
from step 200 with the scheduler at step 200 and lr 2.99985e-4, then reached step 202 with
losses 5.689 / 5.614 and finite parameters — without writing into the pilot directory.

## 17. Training duration

```
training loop (200 updates, incl. 5 validations): 144.5 s
final full validation (7.76 M tokens):             67.9 s
setup, probing and integrity hashing:              ~18 s
total wall clock (18:22:41Z → ~18:26:31Z):         ~230 s
```

## 18. Training throughput

```
training:   22,677 tokens/s  (1.38 optimizer steps/s, 11.1 micro-batches/s)
validation: 114,286 tokens/s (forward only, batch 8 × 512)
```

Measured on this machine, BF16, batch 4 × 512 with 8-step accumulation. DataLoader uses
`num_workers=0`, so this number includes blocking host-to-device copies; it is a starting
figure, not an optimised one.

## 19. Unit and integration test results

```
python -m pytest -q   →   152 passed in 23.11 s   (0 failed, 0 skipped)
```

124 pre-existing tests + **28 new** (24 unit in `tests/unit/test_training.py`, 4 integration in
`tests/integration/test_training_step.py`, including the CUDA mixed-precision step, which ran
because a GPU was present). Required coverage:

| # | Requirement | Test(s) |
|---:|---|---|
| 1 | AdamW parameter grouping | `test_parameter_groups_split_weights_and_biases` |
| 2 | Weight decay exclusion | `test_weight_decay_excludes_biases_and_layer_norms` |
| 3 | Gradient accumulation arithmetic | `test_accumulation_gradients_match_the_large_batch` |
| 4 | Loss scaling | `test_step_reports_mean_micro_batch_loss` (loss/accum must equal the full-batch gradient) |
| 5 | Gradient clipping | `test_gradient_clipping_limits_the_total_norm` |
| 6 | Scheduler warmup | `test_scheduler_warmup_is_linear` |
| 7 | Scheduler cosine decay | `test_scheduler_cosine_decay_is_monotonic` |
| 8 | Validation no-grad behaviour | `test_validation_runs_without_grad` |
| 9 | Validation determinism | `test_validation_is_deterministic`, `test_validation_uses_the_same_blocks_every_time` |
| 10 | Token-weighted validation loss | `test_validation_loss_is_token_weighted` |
| 11 | Checkpoint save/load | `test_checkpoint_roundtrip_is_weights_only` |
| 12 | Optimizer state restoration | `test_optimizer_state_is_restored` |
| 13 | Scheduler restoration | `test_scheduler_state_is_restored` |
| 14 | Weight tying after restoration | `test_weight_tying_survives_checkpoint_restoration` |
| 15 | Resume from checkpoint | `test_resume_matches_an_uninterrupted_run` (3+3 steps == 6 uninterrupted, bitwise) |
| 16 | RNG state restoration | `test_rng_state_is_restored` |
| 17 | CUDA mixed-precision selection | `test_precision_selection_follows_gpu_capability`, `test_training_refuses_cpu_fallback`, `test_cuda_mixed_precision_step_updates_parameters` |
| 18 | Effective batch calculation | `test_effective_batch_arithmetic` |

Plus the end-to-end integration run: 3 optimizer steps over real token streams with validation,
CSV and checkpoints, and a metrics-column contract test.

## 20. OOM attempts and other warnings

**OOM:** none. The probe at 4 × 8 succeeded (peak 949 MiB) and no fallback was triggered, so
there is no failed-run history to reconcile.

**Warnings / limitations:**

1. **200 steps ≈ 2.46 % of one epoch.** The loss curve proves the engine learns, not that the
   model is trained. No claim of complete pretraining is made.
2. **Validation is on a fixed 64-block subset** during training (32,768 of 7,756,800 tokens,
   0.42 %). The full-split number at step 200 (5.815) is the comparable one for later runs.
3. **Loss is not masked at `<|eos|>`** — by design, the stream uses EOS as a separator and
   windows may cross example boundaries, so a small fraction of targets are structural.
4. **Checkpoints are ~195 MiB each** (FP32 weights + AdamW moments). Six exist (1.22 GB).
   Only `step_000200.pt`, `latest.pt` and `best.pt` are needed to continue.
5. **`num_workers=0`** on Windows; throughput (22.7 k tokens/s) is unoptimised.
6. **Schedule is 8,129 steps but only 200 ran.** `total_steps` stays 8,129 in the checkpoint,
   so a resume continues the intended schedule rather than jumping to the cosine floor.
7. **BF16 without a scaler** relies on the gradient-norm guard instead of scaler-driven
   skipping; a single non-finite step aborts the run loudly rather than silently skipping.
8. **Determinism is best-effort.** Same seed, same code and same device reproduce the run;
   CUDA kernels are not bitwise deterministic across different hardware.

## 21. Data and tokenizer integrity

Checked before training and re-checked afterwards:

| Artifact | SHA-256 | Status |
|---|---|---|
| `tokenizer/artifacts/polycodegpt_tokenizer_v1_1.json` | `0459f8556a6f79ca…` | unchanged, hash verified at load |
| `data/processed/tokenized_v1_1/train.bin` | `6b8ffbf653e8a3b3…` | unchanged, matches manifest |
| `data/processed/tokenized_v1_1/valid.bin` | `ee55019be57e1b53…` | unchanged, matches manifest |
| `data/processed/tokenized_v1_1/test.bin` | `4ce477e357c2b3ab…` (manifest) | **never opened**; size 14,895,646 B and mtime unchanged |
| `data/processed/polycodegpt_corpus/train.jsonl` | `2a9c71607a189371…` | unchanged |
| `…/valid.jsonl` | `2eb02406d2210171…` | unchanged |
| `…/test.jsonl` | `93f624f6e1f24d8d…` | unchanged |
| `configs/model.yaml` | equals `manifest.model_config` | architecture unchanged |
| data manifest | v1.1.0, `367636ca923c8e9b…` | unchanged |
| pretrained weights | — | none downloaded; all parameters came from seed 42 |

Initial weights digest `48cccdc478c76552…` → final `1466e7a1a491227f…`; **100/100** parameter
tensors changed, so the optimizer genuinely updated every part of the model.

---

## Artifacts produced

| Path | Content |
|---|---|
| `src/polycodegpt/training/optimizer.py` | AdamW with decay/no-decay groups |
| `src/polycodegpt/training/scheduler.py` | warmup + cosine schedule, checkpointable |
| `src/polycodegpt/training/checkpoint.py` | atomic save, `weights_only` load, RNG capture/restore |
| `src/polycodegpt/training/evaluate.py` | fixed subsets, token-weighted loss, perplexity |
| `src/polycodegpt/training/engine.py` | precision selection, accumulation loop, metrics, checkpoints |
| `src/polycodegpt/training/__init__.py` | public API |
| `scripts/train.py` | pilot CLI: integrity, memory probe, run, full validation |
| `configs/training.yaml` | schedule / evaluation / checkpoint configuration |
| `tests/unit/test_training.py`, `tests/integration/test_training_step.py` | 28 tests |
| `logs/training/pilot_metrics.csv` | 200 metric rows, 17 columns |
| `logs/training/pilot_validation.json` | validation history |
| `logs/training/pilot_summary.json` | full run summary (settings, probe, integrity, reload) |
| `logs/training/pilot_console.log` | console transcript |
| `checkpoints/polycodegpt_pilot/*.pt` | six checkpoints |

Training outputs are git-ignored (`checkpoints/*`, `logs/*`, `*.pt`).

## Verdict

**PILOT TRAINED AND VERIFIED**

200 real optimizer updates on the finalised training stream, BF16 on the RTX 5060, 3,276,800
tokens, training loss 9.774 → 5.665, validation loss 9.791 → 5.873 (subset) / 5.815 (full
split), perplexity 17,874 → 335, 200/200 finite gradient norms, 100/100 parameters updated,
checkpoints resumable with weight tying intact, and no data, tokenizer or test-split
contamination. The next step would be the full 8,129-step schedule — not started.
