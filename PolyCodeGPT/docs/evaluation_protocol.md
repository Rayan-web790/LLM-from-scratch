# PolyCodeGPT — Evaluation Protocol

## Overview

Phase 6 provides two evaluation entry points:

1. **Held-out test evaluation** (quantitative): `scripts/evaluate_test.py`
2. **Qualitative generation samples**: `scripts/generate_samples.py`

Both are inference-only. Neither trains, fine-tunes, or modifies any checkpoint,
tokenizer, or dataset. Neither opens `train.bin` or `valid.bin`.

## Prerequisites

- Final checkpoint: `checkpoints/polycodegpt_first_pass/latest.pt` (step 8,130)
- Tokenizer: `tokenizer/artifacts/polycodegpt_tokenizer_v1_1.json`
- Manifest: `data/processed/tokenized_v1_1/manifest.json`
- Model config: `configs/model.yaml`
- Python interpreter: `C:/Users/UltraPc/Desktop/LLM FROM SCRATCH/.venv/Scripts/python.exe`
- CUDA device available (the scripts fail clearly if CUDA is unavailable; no CPU fallback).

## Test split geometry (verified against manifest)

| Quantity | Value |
|---|---:|
| Complete blocks | 14,546 |
| Target tokens per block | 512 |
| Evaluated target tokens | 7,447,552 |
| Trailing tokens excluded | 270 |
| Context window | 512 tokens |
| Stride | 512 (overlap by 1 token) |

## Command 1 — Held-out test evaluation

Run from the project root (`PolyCodeGPT/`):

```
C:/Users/UltraPc/Desktop/LLM FROM SCRATCH/.venv/Scripts/python.exe scripts/evaluate_test.py
```

Options:
- `--checkpoint PATH` (default: `checkpoints/polycodegpt_first_pass/latest.pt`)
- `--batch-size N` (default: 8)
- `--report PATH` (default: `reports/phase6_test_evaluation_report.md`)
- `--max-batches N` (optional smoke-test cap; omit for full evaluation)

The script:
1. Loads the checkpoint with `weights_only=True`.
2. Verifies checkpoint, tokenizer, model config, and manifest consistency.
3. Verifies tied embeddings survive loading and all parameters are finite.
4. Opens only `test.bin` (read-only memmap) and evaluates all 14,546 blocks.
5. Computes token-weighted mean cross-entropy and perplexity.
6. Records test-data access (size/mtime before and after).
7. Writes `reports/phase6_test_evaluation_report.md`.

**Status until run:** `NOT RUN`. Freebuff does not execute this; the user runs it.

## Command 2 — Qualitative generation

```
C:/Users/UltraPc/Desktop/LLM FROM SCRATCH/.venv/Scripts/python.exe scripts/generate_samples.py
```

Options:
- `--checkpoint PATH` (default: `checkpoints/polycodegpt_first_pass/latest.pt`)
- `--max-new-tokens N` (default: 128)
- `--report PATH` (default: `reports/phase6_generation_samples.md`)

Generates greedy completions (temperature=0) for 8 hand-written prompts
(4 Python, 4 JavaScript) using the project's serializer format and special tokens.
Outputs are qualitative examples, not formal correctness scores.

## Safety and integrity

- No training, fine-tuning, or optimizer updates.
- No gradients computed during evaluation.
- `train.bin` and `valid.bin` are never opened by these scripts.
- `test.bin` is opened read-only (memmap) only by `evaluate_test.py`, and its
  size/mtime are recorded before and after.
- No hyperparameter tuning based on test results.
- Validation data is never used as test data.
- All model parameters verified finite; tied embeddings verified after load.
- Fail clearly if CUDA is unavailable (no CPU fallback).

## Limitations

- French and Arabic competence is NOT claimed: those languages were reserved in
  the tokenizer (IDs 9, 10) but not represented in the described training corpus.
- Generation samples are qualitative and may be incoherent for a model trained on
  a single epoch over a small corpus.
- Perplexity is computed over complete 512-token blocks only; 270 trailing tokens
  are excluded by the established complete-block geometry.
