# Phase 12C — exp_b Completion Audit (read-only)

No training launched, no checkpoint/dataset modified. All checks below are read-only.

## 1. Final checkpoint

- Path: `experiments/phase12b/runs/exp_b/step_016260.pt`
- Size: 207,215,053 bytes
- SHA256: `fc70f5264a97d22c162986e172f8e70881713d5ae235c992e516a91ff67f5db6`
- `latest.pt` and `best.pt` also present.

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
| `best_val_loss` / `best_step` | 2.2751977536827326 / 16260 |
| `passes` | 0 (bookkeeping artifact; weights unaffected) |
| optimizer param `step` (all 100 params) | 16260 |
| param-group `lr` | 2.9999999999999997e-05 (cosine end, 0.1× base) |
| scheduler `total_steps` / `last_step` | 8130 / 8130 |
| scheduler `warmup_steps` | 163 |
| scheduler `base_lrs` | [0.0003, 0.0003] |
| `loader_batches_consumed` | 65,036 |
| tokenizer SHA256 | `0459f8556a6f79ca1ba1c3dc3f76c26ae2e55653ebbf97beaa9bd23467358a03` |
| `data_manifest_sha256` | `367636ca923c8e9b6d0693a7606ae06504e76067908c686bcbf0e746508f5544` |

All 8,130 stage updates applied and persisted.

## 3. Final partial update ordering

Shared `Trainer.run()` / `train_step()` path (identical to control/exp_a):

- Final metrics entry: optimizer step 16260, `effective_batch_size=14`,
  `tokens_per_step=7168`, `validation_loss=2.2751977536827326`,
  `tokens_seen=266,385,408`.
- Final validation entry: step 16260, loss 2.2751977536827326.
- `train_step()` runs the 14-sequence partial group, calls `optimizer.step()` +
  `scheduler.step()`, advances `stage_step` to 8130, then `due_eval`/`due_checkpoint`
  fire on the advanced step, so validation and `save()` (step_016260.pt + latest.pt)
  occur after the update. Confirmed by checkpoint contents.

## 4. Base checkpoint unchanged

- `checkpoints/polycodegpt_first_pass/step_008130.pt`
- SHA256: `5562acc2fd58837f9d4245042b7f0b10de65520aa2ba292999c80045470590bd` (matches expected).

## 5. Dataset (language-balanced, original data only)

Manifest: `data/processed/phase12b_experiments/exp_b/dataset_manifest.json`
- `train.bin` SHA256: `eab16b96373c07022f7725af234f367126ecd7a12feb5e6df190274ddf4d895a` (verified on disk)
- file size 266,385,830 bytes, 133,192,915 serialized tokens
- Python: subsampled to budget, 65,172,938 materialized tokens (target 65,172,968, 30 unfilled), 191,985 records, no repeats, coverage 0.661
- JavaScript: original data oversampled with replacement, 65,172,957 materialized tokens (target 65,172,969, 12 unfilled), 210,980 instances, 102,169 unique, max 4 repeats, effective epochs 2.065
- English docs: 2,847,020 tokens, 500 records
- Geometry: 260,142 blocks, 133,192,705 tokens in blocks, 210 dropped trailing
- Full updates 8,129 + 14-sequence final partial = 8,130

Python exposure ≈ 65,172,968 (materialized 65,172,938); JavaScript exposure ≈ 65,172,969 (materialized 65,172,957); English docs 2,847,020. Manifest SHA on disk: `08d6ee214a2da6d305ee8e85eb7b9d8c94f2ec6341c2137d0b100795a8028416`.

## 6. Update counters

`run_summary.json` (`experiments/phase12b/runs/exp_b/run_summary.json`):
- `steps`: 16260, `stage_step`: 8130, `initial_global_step`: 8130
- `additional_updates`: 8130, `expected_final_global_step`: 16260
- `skipped_steps`: 0
- `micro_batches`: 130072, `sequences`: 520284
- `passes`: 1
- `tokens_seen`: 266,385,408, `initial_tokens_seen`: 133,192,704,
  `stage_tokens_seen`: 133,192,704

8,130 additional updates, zero skipped, final global step exactly 16,260.

## 7. Throughput accounting

- `elapsed_seconds`: 4176.8763228
- `tokens_per_second`: 31888.112959665803
- Verification: `stage_tokens_seen / elapsed = 133,192,704 / 4,176.876 = 31,888.11`
  (cumulative would be 266,385,408 / 4,176.876 = 63,776.2). Uses stage delta, not cumulative. Correct.

## 8. Validation losses (existing validation partition only)

Full validation pass over `valid.jsonl` / `valid.bin` with the exp_b checkpoint
(inference only, no writes, no test data, no independent_v3):

| category | tokens | mean CE | perplexity |
|---|---:|---:|---:|
| python_docstring | 1,345,519 | 3.284013 | 26.6826 |
| python_code | 4,409,459 | 2.061429 | 7.8572 |
| javascript_docstring | 325,490 | 3.064850 | 21.4312 |
| javascript_code | 1,592,479 | 2.122610 | 8.3529 |
| structural_special | 83,853 | 0.667564 | 1.9495 |
| **combined python** | **5,754,978** | **2.347271** | **10.4570** |
| **combined javascript** | **1,917,969** | **2.282513** | **9.8013** |
| overall | 7,756,800 | 2.313100 | 10.1057 |

Python validation loss: **2.3473**. JavaScript validation loss: **2.2825**.
(Training-time reported `best_val_loss` 2.2752 is the sampled validation subset at step 16260.)

## 9. Failed zero-update evidence

- `reports/phase12b_failed_zero_update_run/run_summary.json` intact
  (SHA256 `8ed2943528955abeb8d802580e69bcfc362b15cacad4941a4a35660b763230ee`,
  520 bytes): steps 8130, stage_step 0, additional_updates 0, tokens_seen 133,192,704.
  Preserved unchanged.

## 10. Conclusion

- **exp_b is accepted.** All counters exact (8,130/8,130, zero skipped, final step 16,260),
  final partial update applied and persisted in `step_016260.pt`, base checkpoint and
  tokenizer hashes unchanged, dataset manifest and `train.bin` hash verified, throughput
  correctly uses stage delta, and per-language validation losses computed on the existing
  validation partition only.
- **exp_c is safe to start**, subject to the same pre-run verification pattern:
  confirm its dataset manifest and `train.bin` hashes, then run with the verified
  continuation engine. exp_c's dataset is already staged under
  `data/processed/phase12b_experiments/exp_c/`; no run directory exists yet.
  No blocker identified from exp_b's results. No training launched by this audit.