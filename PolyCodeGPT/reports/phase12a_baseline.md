# Phase 12A — Baseline Freeze and Verification

Recorded 2026-11-17T16:21:56+00:00. Read-only: every hash below was computed from the existing files, and nothing was written outside `reports/`.

## Baseline checkpoint

| field | value |
|---|---|
| file | `checkpoints/polycodegpt_first_pass/step_008130.pt` |
| sha256 | `5562acc2fd58837f9d4245042b7f0b10de65520aa2ba292999c80045470590bd` |
| optimizer step | 8130 |
| tokens seen | 133,192,704 |
| best validation loss | 2.392860 (step 8130) |
| precision | bf16 |
| seed | 42 |

## Training configuration recorded in the checkpoint

| setting | value |
|---|---|
| micro_batch_size | 4 |
| gradient_accumulation_steps | 8 |
| effective_batch_size | 32 |
| tokens_per_step | 16384 |
| max_steps | 8130 |
| total_steps | 8130 |
| warmup_fraction | 0.02 |
| warmup_steps | 163 |
| min_lr_ratio | 0.1 |
| learning_rate | 0.0003 |
| weight_decay | 0.1 |
| gradient_clip | 1.0 |
| precision | auto |
| seed | 42 |
| context_length | 512 |
| eval_steps | [] |
| checkpoint_steps | [] |
| eval_every | 100 |
| checkpoint_every | 500 |
| complete_pass | True |
| partial_final_update | True |
| total steps | 8130 |
| warmup steps | 163 |
| min lr ratio | 0.1 |

## Model configuration

```json
{
  "vocab_size": 16000,
  "d_model": 384,
  "n_layers": 6,
  "n_heads": 6,
  "ffn_dim": 1536,
  "context_length": 512,
  "dropout": 0.1,
  "tie_embeddings": true,
  "layer_norm_eps": 1e-05,
  "init_std": 0.02,
  "architecture": "decoder_only_transformer"
}
```

## Verification checks

| check | result | detail |
|---|---|---|
| `step_is_8130` | PASS | `{"value": 8130}` |
| `tokenizer_hash_matches` | PASS | `{"checkpoint": "0459f8556a6f79ca1ba1c3dc3f76c26ae2e55653ebbf97beaa9bd23467358a03", "on_disk": "0459f8556a6f79ca1ba1c3dc3f76c26ae2e55653ebbf97beaa9bd23467358a...` |
| `data_manifest_hash_matches` | PASS | `{"checkpoint": "367636ca923c8e9b6d0693a7606ae06504e76067908c686bcbf0e746508f5544", "on_disk": "367636ca923c8e9b6d0693a7606ae06504e76067908c686bcbf0e746508f55...` |
| `model_state_loads_strictly` | PASS | `{"missing": 0, "unexpected": 0, "parameter_count": 16988160, "checkpoint_parameter_count": 16988160}` |
| `optimizer_state_restores` | PASS | `{"parameter_groups": 2, "tensors_with_moments": 100, "parameters": 100, "adam_step_values": [8130], "group_weight_decay": [0.1, 0.0]}` |
| `scheduler_state_restores` | PASS | `{"total_steps": 8130, "warmup_steps": 163, "min_lr_ratio": 0.1, "last_step": 8130, "base_lr": 0.0003, "next_step_lr": 3e-05}` |
| `rng_state_present` | PASS | `{"keys": ["cuda", "loader_generator", "numpy", "python", "torch"], "loader_epoch_state_shape": [5056], "loader_epoch_state_dtype": "torch.uint8"}` |
| `token_accounting` | PASS | `{"tokens_seen": 133192704, "sequences_x_context": 133192704, "sequences": 260142, "micro_batches": 65036, "partial_final_group_sequences": 14, "partial_final...` |

All checks passed: **True**

Optimizer state can be restored consistently: the checkpoint carries AdamW moments for every trainable tensor, a single shared Adam step counter, both decay/no-decay groups, the scheduler's step counters, and Torch/CUDA/Python/NumPy/loader RNG states plus the loader epoch state tensor. Experiment branches will resume from a **copy** of `step_008130.pt`; the original is never opened for writing.

## Protected artifacts

| kind | files |
|---|---|
| checkpoint | 3 |
| tokenizer | 6 |
| data_manifest | 14 |
| config | 2 |
| report | 36 |

Full per-file hashes are in `reports/phase12a_baseline.json`.
