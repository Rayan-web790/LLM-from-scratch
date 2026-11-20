# Phase 12C — exp_a Completion Audit (read-only)

No training launched, no checkpoint/dataset modified. All checks below are read-only.

## 1. Final checkpoint

- Path: `experiments/phase12b/runs/exp_a/step_016260.pt`
- Size: 207,215,053 bytes
- SHA256: `65fab1180eff51c37dc7c3d872a8ed4545690ea46d150180eb3b7ed1f6fb6f45`
- `latest.pt` and `best.pt` also present in the same directory.

## 2. Checkpoint metadata (loaded read-only)

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
| `best_val_loss` / `best_step` | 2.2567871687933803 / 16260 |
| `passes` | 0 (bookkeeping artifact; weights unaffected, same as control) |
| optimizer param `step` (all 100 params) | 16260 |
| param-group `lr` | 2.9999999999999997e-05 (cosine end, 0.1× base) |
| scheduler `total_steps` / `last_step` | 8130 / 8130 |
| scheduler `warmup_steps` | 163 |
| scheduler `base_lrs` | [0.0003, 0.0003] |
| `loader_batches_consumed` | 65,036 |
| tokenizer SHA256 | `0459f8556a6f79ca1ba1c3dc3f76c26ae2e55653ebbf97beaa9bd23467358a03` |
| `data_manifest_sha256` | `367636ca923c8e9b6d0693a7606ae06504e76067908c686bcbf0e746508f5544` |

All 8,130 stage updates (including the final partial group) were applied and persisted.

## 3. Final partial update ordering

Shared `Trainer.run()` / `train_step()` path (identical to control):

1. `train_step(batches)` runs the micro-batch group, calls `optimizer.step()` + `scheduler.step()`,
   then increments `stage_step` and sets `step_index = initial_global_step + stage_step`.
   For the final group this is the 14-sequence / 7,168-token partial update
   (metrics entry: `effective_batch_size=14`, `tokens_per_step=7168`).
2. `due_eval` computed on the advanced step → `validate()` runs on updated weights.
3. `due_checkpoint` computed on the advanced step → `save()` writes `step_016260.pt`
   and `latest.pt` from the updated model/optimizer.
4. Only then is the partial-update log line emitted.

Evidence in checkpoint: last `validation` entry is at step 16260 with loss 2.2567871687933803;
last `metrics` entry is optimizer step 16260 with `validation_loss=2.2567871687933803`,
`effective_batch_size=14`, `tokens_per_step=7168`. The partial update preceded both
validation and checkpoint saving.

## 4. Base checkpoint unchanged

- `checkpoints/polycodegpt_first_pass/step_008130.pt`
- SHA256: `5562acc2fd58837f9d4245042b7f0b10de65520aa2ba292999c80045470590bd` (matches expected).

## 5. Dataset (curated JavaScript, oversampled with replacement)

Manifest: `data/processed/phase12b_experiments/exp_a/dataset_manifest.json`
- `train.bin` SHA256: `37692f87f8c66a9c36623f70c1dc3d356c67d50d028306bab836d9b944086150` (verified on disk)
- file size 266,385,900 bytes, 133,192,950 tokens
- Python: 98,807,136 tokens, 290,463 records, all original, coverage 1.0
- JavaScript: curated `js_v1` source, 35,658 unique records, oversampled with replacement
  to 31,538,794 tokens (target 31,538,801, 7 unfilled), 171,920 instances,
  max 5 repeats, effective epochs 4.8214, unique coverage 1.0
- English docs: 2,847,020 tokens, 500 records
- Blocks: 260,142; full updates 8,129 + 14-sequence final partial = 8,130

JavaScript is curated and oversampled with replacement to the original JavaScript token
volume (31,538,801), preserving the Python:JavaScript exposure ratio and total budget
identical to control. Manifest and `train.bin` hashes recorded above.

## 6. Update counters

`run_summary.json` (`experiments/phase12b/runs/exp_a/run_summary.json`):
- `steps`: 16260, `stage_step`: 8130, `initial_global_step`: 8130
- `additional_updates`: 8130, `expected_final_global_step`: 16260
- `skipped_steps`: 0
- `micro_batches`: 130072, `sequences`: 520284
- `passes`: 1
- `tokens_seen`: 266,385,408, `initial_tokens_seen`: 133,192,704,
  `stage_tokens_seen`: 133,192,704

8,130 additional updates, zero skipped, final global step exactly 16,260.

## 7. Throughput accounting

- `elapsed_seconds`: 3094.8891219
- `tokens_per_second`: 43036.341126893516
- Verification: `stage_tokens_seen / elapsed = 133,192,704 / 3,094.889 = 43,036.34`
  (cumulative would be 266,385,408 / 3,094.889 = 86,072.7). Uses stage delta, not cumulative. Correct.

## 8. Validation losses (existing validation partition only)

Full validation pass over `valid.jsonl` / `valid.bin` with the exp_a checkpoint
(inference only, no writes, no test data, no independent_v3):

| category | tokens | mean CE | perplexity |
|---|---:|---:|---:|
| python_docstring | 1,345,519 | 3.132213 | 22.9246 |
| python_code | 4,409,459 | 2.010015 | 7.4634 |
| javascript_docstring | 325,490 | 3.161532 | 23.6067 |
| javascript_code | 1,592,479 | 2.289585 | 9.8708 |
| structural_special | 83,853 | 0.628828 | 1.8754 |
| **combined python** | **5,754,978** | **2.272386** | **9.7025** |
| **combined javascript** | **1,917,969** | **2.437559** | **11.4451** |
| overall | 7,756,800 | 2.295460 | 9.9290 |

Python validation loss: **2.2724**. JavaScript validation loss: **2.4376**.
(Training-time reported `best_val_loss` 2.2568 is the sampled validation subset at step 16260.)

## 9. Conclusion

- **exp_a is accepted.** All counters exact (8,130/8,130, zero skipped, final step 16,260),
  final partial update applied and persisted in `step_016260.pt`, base checkpoint and
  tokenizer hashes unchanged, dataset manifest and `train.bin` hash verified, throughput
  correctly uses stage delta, and per-language validation losses computed on the existing
  validation partition only.
- **exp_b is safe to start**, subject to the same pre-run verification pattern:
  confirm its dataset manifest and `train.bin` hashes, then run with the verified
  continuation engine. No blocker identified from exp_a's results. No training launched
  by this audit.