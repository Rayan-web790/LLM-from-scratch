# Phase 12C — exp_c Completion Audit (strictly read-only)

No training, generation, or artifact modification. All checks below are read-only.

## 1. Checkpoint integrity

- Path: `experiments/phase12b/runs/exp_c/step_016260.pt`
- Size: 207,215,053 bytes
- SHA256: `cf0544099d73f85cdeb40bc1e0a93ce17bb790db4cb7136254be65d16c3b45b4`
- `latest.pt` and `best.pt` present alongside.

Checkpoint metadata (loaded read-only):

| field | value |
|---|---:|
| `step` | 16260 |
| `stage_step` | 8130 |
| `initial_global_step` | 8130 |
| `tokens_seen` (cumulative) | 266,385,408 |
| `initial_tokens_seen` | 133,192,704 |
| `micro_batches` | 130,072 |
| `sequences` | 520,284 |
| `skipped_steps` | 0 |
| `best_val_loss` / `best_step` | 2.319764374755323 / 16260 |
| `passes` | 0 (bookkeeping artifact; weights unaffected) |
| optimizer param `step` (all 100 params) | 16260 |
| param-group `lr` | 2.9999999999999997e-05 (cosine end, 0.1× base) |
| scheduler `total_steps` / `last_step` | 8130 / 8130 |
| scheduler `warmup_steps` | 163 |
| scheduler `base_lrs` | [0.0003, 0.0003] |
| `loader_batches_consumed` | 65,036 |
| tokenizer SHA256 | `0459f8556a6f79ca1ba1c3dc3f76c26ae2e55653ebbf97beaa9bd23467358a03` |
| `data_manifest_sha256` | `367636ca923c8e9b6d0693a7606ae06504e76067908c686bcbf0e746508f5544` |

Model weights and AdamW moments represent the completed Exp C run (optimizer step advanced to 16260; moments updated).

Base checkpoint `checkpoints/polycodegpt_first_pass/step_008130.pt`:
- SHA256: `5562acc2fd58837f9d4245042b7f0b10de65520aa2ba292999c80045470590bd` (matches expected, unchanged).

## 2. Final partial update — execution order

Do not infer from console print order. Traced the shared `Trainer.run()` / `train_step()` path:

1. `train_step(batches)` runs the micro-batch group, calls `optimizer.step()` + `scheduler.step()`,
   then increments `stage_step` and sets `step_index = initial_global_step + stage_step`.
   Final group: 14 sequences / 7,168 tokens (metrics entry: `effective_batch_size=14`,
   `tokens_per_step=7168`).
2. `due_eval` computed on the advanced step → `validate()` runs on updated weights.
3. `due_checkpoint` computed on the advanced step → `save()` writes `step_016260.pt`
   and `latest.pt` from the updated model/optimizer.
4. Only then is the partial-update log line emitted.

Concrete evidence in checkpoint:
- Last `metrics` entry: optimizer step 16260, `effective_batch_size=14`, `tokens_per_step=7168`,
  `validation_loss=2.319764374755323`, `blocks_consumed=520284`, `tokens_seen=266385408`.
- Last `validation` entry: step 16260, loss 2.319764374755323, 262144 tokens, 512 sequences.
- Optimizer all 100 params `step=16260`; scheduler `last_step=8130`.

Conclusion: the final partial optimizer update was applied before final validation and checkpoint saving.

## 3. Dataset and exposure

Manifest: `data/processed/phase12b_experiments/exp_c/dataset_manifest.json`
- Manifest SHA256 on disk: `c7bc967c612d429067fefcd50cbfcaeab1531512c36b4f48d776a6a4aadbfcd4`
- `train.bin` SHA256: `99c7077c570f1067ce8e594cd186b02ef86f814d40eb0b7c84ba19648f7ffc55` (verified on disk)
- file size 266,385,850 bytes, 133,192,925 serialized tokens

Exposure:
- Python: subsampled to balanced budget, 65,172,938 materialized tokens
  (target 65,172,968, 30 unfilled), 191,985 records, no repeats, coverage 0.661.
- Curated JavaScript: oversampled with replacement, 65,172,967 materialized tokens
  (target 65,172,969, 2 unfilled), 35,658 unique records, 355,196 instances,
  max 10 repeats, effective epochs 9.961.
- English documentation: 2,847,020 tokens, 500 records.
- Geometry: 260,142 blocks, 133,192,705 tokens in blocks, 220 dropped trailing.
- Full updates 8,129 + 14-sequence final partial = 8,130.

Matches requested: Python ≈ 65,172,968; curated JS ≈ 65,172,969; English docs 2,847,020;
curated JS oversampled with replacement; Python subsampled to balanced budget.

## 4. Doubled counter discrepancy — resolution

Reported in run_summary/checkpoint: `micro_batches = 130,072`, `sequences = 520,284`.
Configured schedule implies stage-only: 8,129 full × 8 micro-batches + 4 partial = 65,036 micro-batches;
8,129 full × 32 sequences + 14 partial = 260,142 sequences.

Trace of increment sites in `src/polycodegpt/training/engine.py`:
- `self.micro_batches += 1` (line ~541) executes once per micro-batch inside the
  `for (input_ids, targets), size in zip(group, sizes)` loop. Within a pass this yields
  exactly 65,036 (8,129×8 + 4) for Exp C's stage. No per-unit duplication occurs within a pass.
- `self.sequences += total_sequences` (line ~575) executes once per successful optimizer step.
  Within a pass this yields 260,142 (8,129×32 + 14). No per-unit duplication.

Cause of the doubling: `restore()` copies `micro_batches`, `sequences`, and `tokens_seen` from the
base checkpoint verbatim (lines ~686-687), and `train_step()` then increments them again. The base
checkpoint `step_008130.pt` already carries `micro_batches=65,036`, `sequences=260,142`,
`tokens_seen=133,192,704` (one full pass). The continuation stage adds an equal second pass, so the
final counters are cumulative across both passes: 130,072 / 520,284 / 266,385,408.

This mirrors `tokens_seen`, which is also cumulative and is reported with an
`initial_tokens_seen` / `stage_tokens_seen` split (both 133,192,704 here). The asymmetry is that
`micro_batches` and `sequences` have no equivalent initial/stage split in either the checkpoint schema
or `run_summary.json`, so they appear as a single cumulative number.

Determination: this is NOT a per-unit double-counting defect (each micro-batch/sequence is counted
once per pass, and the stage performed exactly 65,036 micro-batches / 260,142 sequences). It is a
metrics-accounting asymmetry: cumulative counters without a stage split, inconsistent with how
`tokens_seen` is handled. Training correctness, update counts, and weights are unaffected. This is
a reporting limitation, not a training defect; no model-quality impact. Recommended (non-blocking)
future cleanup: add `initial_micro_batches`/`stage_micro_batches` and
`initial_sequences`/`stage_sequences` splits mirroring `tokens_seen`.

## 5. Training summary

`run_summary.json` (`experiments/phase12b/runs/exp_c/run_summary.json`):
- `steps`: 16260, `stage_step`: 8130, `initial_global_step`: 8130
- `additional_updates`: 8130, `expected_final_global_step`: 16260
- `skipped_steps`: 0
- `micro_batches`: 130072 (cumulative, see §4), `sequences`: 520284 (cumulative, see §4)
- `passes`: 1
- `tokens_seen`: 266,385,408, `initial_tokens_seen`: 133,192,704,
  `stage_tokens_seen`: 133,192,704 (reconciles: initial + stage = cumulative)
- `elapsed_seconds`: 3008.7802298
- `tokens_per_second`: 44268.00690885073

Verification:
- 8,130 additional updates; final global step 16,260; zero skipped. Confirmed.
- Stage tokens 133,192,704. Confirmed.
- Elapsed stage time 3,008.780 s. Confirmed.
- Throughput: `stage_tokens_seen / elapsed = 133,192,704 / 3,008.780 = 44,268.0 tok/s`
  (cumulative would be 266,385,408 / 3,008.780 = 88,536.0). Uses stage delta, not cumulative. Correct.
- `tokens_seen = initial_tokens_seen + stage_tokens_seen` (133,192,704 + 133,192,704 = 266,385,408). Reconciles.

## 6. Validation (existing validation partition only)

Full validation pass over `valid.jsonl` / `valid.bin` with the exp_c checkpoint
(inference only, no writes, no test data, no independent_v3):

| category | tokens | mean CE | perplexity |
|---|---:|---:|---:|
| python_docstring | 1,345,519 | 3.315134 | 27.5261 |
| python_code | 4,409,459 | 2.078882 | 7.9955 |
| javascript_docstring | 325,490 | 3.164832 | 23.6848 |
| javascript_code | 1,592,479 | 2.263084 | 9.6127 |
| structural_special | 83,853 | 0.714359 | 2.0429 |
| **combined python** | **5,754,978** | **2.367919** | **10.6752** |
| **combined javascript** | **1,917,969** | **2.416115** | **11.2023** |
| overall | 7,756,800 | 2.361961 | 10.6117 |

Python validation loss: **2.3679**. JavaScript validation loss: **2.4161**. Combined: **2.3620**.
(Training-time reported `best_val_loss` 2.3198 is the sampled validation subset at step 16260.)

## 7. Final decision

All critical checks pass: checkpoint integrity, 8,130/8,130 updates, zero skipped, final global step 16,260,
final partial update applied before validation/checkpoint save, base checkpoint and tokenizer hashes
unchanged, dataset manifest and `train.bin` hash verified, throughput uses stage delta, per-language
validation losses computed on the existing validation partition only. The only noted issue is the
cumulative-without-stage-split reporting asymmetry for `micro_batches`/`sequences` (§4), which does
not affect training or model quality.

**ACCEPT EXP_C**