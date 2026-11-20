# Phase 4 — Transformer Implementation Report

**Project:** PolyCodeGPT
**Date:** 2026-11-15
**Scope:** Implement the decoder-only Transformer from random initialization.
**Status:** Implementation complete, unit-tested, CUDA smoke test passed. No pretraining performed.

---

## 1. Files created or modified

| Path | Status | Purpose |
|---|---|---|
| `src/polycodegpt/model/__init__.py` | modified | Package exports (`GPT`, `ModelConfig`, checkpoint helpers) |
| `src/polycodegpt/model/config.py` | created | `ModelConfig`: YAML loading, validation, parameter estimate, tokenizer-vocab check |
| `src/polycodegpt/model/attention.py` | created | `CausalSelfAttention` (manual Q/K/V, causal mask, dropout, out-projection) |
| `src/polycodegpt/model/layers.py` | created | `FeedForward` (GELU) and `TransformerBlock` (pre-LayerNorm) |
| `src/polycodegpt/model/gpt.py` | created | `GPT`: embeddings, blocks, final LN, tied head, loss, `generate`, init metadata, parameter breakdown |
| `src/polycodegpt/model/checkpoint.py` | created | Weights-only save/load with plain-dict metadata |
| `tests/unit/test_model_config.py` | created | 5 tests |
| `tests/unit/test_attention.py` | created | 4 tests |
| `tests/unit/test_transformer_block.py` | created | 3 tests |
| `tests/unit/test_gpt_model.py` | created | 17 tests |
| `scripts/test_model.py` | created | One-step CUDA smoke test |
| `reports/phase4_transformer_implementation_report.md` | created | This report |

`configs/model.yaml` was inspected and left unchanged — it already specified the locked dimensions.
No dataset, tokenizer artifact, or tokenizer source file was modified.

---

## 2. Actual architecture

Decoder-only Transformer, implemented entirely with `torch.nn` primitives:

- Token embedding `[B,T] -> [B,T,384]`, learned absolute positional embedding added, embedding dropout.
- 6 pre-LayerNorm blocks, each:
  - `x = x + Attention(LN1(x))`
  - `x = x + FeedForward(LN2(x))`
- Attention: separate `q_proj`/`k_proj`/`v_proj`/`out_proj` (all biased), reshape to `[B, 6, T, 64]`,
  `softmax(QK^T / sqrt(64) + causal_mask)V`, attention dropout, output projection.
  Causal mask is an upper-triangular boolean mask (`triu(diagonal=1)`) filled with `-inf`.
- FFN: `Linear(384,1536) -> GELU -> Dropout -> Linear(1536,384) -> Dropout`.
- Final `LayerNorm(384, eps=1e-5)`, then `Linear(384,16000, bias=False)` tied to the token embedding.
- `nn.MultiheadAttention` and `nn.Transformer*` are not used anywhere; asserted by a test that
  AST-parses every model source file for banned attributes and banned third-party imports.

Initialization: `normal(0, 0.02)` for Linear and Embedding weights, zeros for biases, ones/zeros for
LayerNorm affine parameters, residual projections (`out_proj.weight`, `fc2.weight`) re-scaled to
`0.02 / sqrt(2*6) = 0.0057735`. No checkpoint is loaded. Metadata recorded in every checkpoint.

---

## 3. Exact parameter count

```
16,988,160 trainable parameters
```

This is the value produced by `sum(p.numel() for p in model.parameters())` on the real configuration,
and it equals `ModelConfig.estimate_parameters()`. The expected figure in the task brief matches
exactly; no discrepancy to report.

---

## 4. Parameter breakdown

| Component | Parameters |
|---|---|
| Token embeddings (16000 x 384) | 6,144,000 |
| Positional embeddings (512 x 384) | 196,608 |
| Attention projections (6 layers x 591,360) | 3,548,160 |
| Feed-forward layers (6 layers x 1,181,568) | 7,089,408 |
| LayerNorm parameters (6 x 1,536 + final 768) | 9,984 |
| Final vocabulary projection | 0 (tied) |
| **Total** | **16,988,160** |

Per-layer detail: attention = 3 x (384x384 + 384) + (384x384 + 384) = 591,360;
FFN = (384x1536 + 1536) + (1536x384 + 384) = 1,181,568; two LayerNorms = 1,536.

Weight tying: the output projection contributes **no** extra parameters because
`model.lm_head.weight is model.token_embedding.weight` (verified by object identity, in the live
model and after reload from a checkpoint). The 6,144,000 embedding parameters serve both roles.

---

## 5. Special-token compatibility

Tokenizer artifact `polycodegpt_tokenizer_v1_1.json` loads with `get_vocab_size() == 16000`, which
matches `ModelConfig.vocab_size`; `validate_tokenizer_vocab` accepts it and rejects mismatches.
All 11 special tokens (IDs 0–10: pad, unk, eos, code, docstring, text, lang:python, lang:javascript,
lang:en, lang:fr, lang:ar) fit inside the 16,000-entry softmax, and the smoke test tokenizes real
serialized training records with this tokenizer. Serializer version 1.1.0 confirmed in metadata.

---

## 6. Unit test results

Command: `python -m pytest -q` (project `.venv`)

```
82 passed in 2.66s
```

Previous suites (tokenizer/serializer, 53 tests) still pass; the 29 new tests are:

| File | Tests | Coverage |
|---|---|---|
| `test_model_config.py` | 5 | YAML load of the locked dims, invalid dimensions rejected, tokenizer vocab compatibility, parameter estimate |
| `test_attention.py` | 4 | Output shapes `[B,T,d_model]`, causal masking (perturbing the last token leaves earlier outputs unchanged), finiteness under dropout, determinism under a fixed seed |
| `test_transformer_block.py` | 3 | Residual shape preservation, finiteness, FFN shape |
| `test_gpt_model.py` | 17 | Forward shapes, finite CE loss, `ignore_index=-100`, backward gradients present and finite, weight-tying identity, over-length sequence rejected, generation shape, greedy determinism in eval mode, real EOS stopping (forced-logit model stops after one token), long-context cropping, future-token perturbation invariance of earlier logits, checkpoint save/reload identical eval logits, no banned/pretrained dependencies, parameter breakdown sums to total, generation preserves train mode, generation does not mutate parameters, full-size 16K/6-layer forward smoke |

Unit tests use small dimensions (`ModelConfig.small()`: vocab 32, d_model 16, 4 heads, 2 layers);
one integration test builds the real 16,988,160-parameter model and checks a forward pass.
Synthetic tensors are used only as test inputs; nothing is trained.

---

## 7. CUDA smoke test results

Command: `python scripts/test_model.py` — `SMOKE TEST PASSED`

```
device: cuda
tokenizer vocab_size: 16000
total params: 16,988,160
estimated params: 16,988,160
logits shape: (4, 31, 16000)
initial loss: 9.750381
all grads finite: True
weight tying: OK
checkpoint reload: OK
```

Batch of 4 real serialized training records, 32 tokens each (31 inputs / 31 shifted targets, pad
positions masked with `-100`), one AdamW step at lr 3e-4, weight decay 0.1, gradient clipping 1.0.

---

## 8. Initial loss value

**9.750381** — within 0.07 nats of `ln(16000) = 9.680344`, as expected at random initialization
(uniform-over-vocabulary prediction).

---

## 9. GPU memory consumption

```
allocated after optimizer step: 268.98 MB
peak allocated:                 334.34 MB
```

Measured with `torch.cuda.reset_peak_memory_stats()` / `memory_allocated()` on the RTX 5060 Laptop
GPU (8 GB). Well inside budget; real training will raise this as batch size and sequence length grow.

---

## 10. Checkpoint verification

`checkpoints/smoke.pt` (68 MB, Git-ignored) stores a single plain dict:

- `state_dict` — 101 tensors, all `torch.Tensor` (no pickled Python objects); saved/loaded with
  `weights_only=True`
- `config` — plain dict from `ModelConfig.to_dict()`
- `tokenizer_version` = `"1.1.0"`, `tokenizer_sha256` = `0459f855...58a03`
- `param_count` = 16,988,160
- `init_metadata` — scheme, `init_std`, residual scaling, LayerNorm eps, `pretrained: False`,
  `tie_embeddings: True`

Unit test confirms save → load into a fresh model reproduces eval logits exactly
(`torch.allclose`), and tying survives the round trip. The smoke script additionally performs this
round trip after the optimizer step on CPU. Note: after `optimizer.step()` the checkpoint reflects
one updated step, not the pristine random initialization.

---

## 11. Dataset and tokenizer integrity

| Artifact | State |
|---|---|
| `tokenizer/artifacts/polycodegpt_tokenizer_v1_1.json` | unchanged, sha256 `0459f855...58a03` |
| `tokenizer/artifacts/tokenizer_metadata_v1_1.json` | unchanged |
| `data/processed/polycodegpt_corpus/train.jsonl` | unchanged, sha256 `2a9c7160...45c74` — matches the hash recorded in tokenizer metadata |
| `data/processed/polycodegpt_corpus/valid.jsonl` | not opened by model code or smoke test |
| `data/processed/polycodegpt_corpus/test.jsonl` | not opened by model code or smoke test |

The smoke test streams only `train.jsonl`. GPU availability re-verified after the run
(`torch.cuda.is_available() == True`, RTX 5060 Laptop GPU). All generated artifacts
(`checkpoints/`, `*.pt`) are covered by the existing `.gitignore`.

---

## 12. Warnings and limitations

1. **Interpreter matters.** The project environment is `../.venv` (torch 2.15.0+cu132, CUDA 13.2).
   The system `Python311` has torch 2.6.0+cu124, whose kernels do not support `sm_120`, so any run
   under that interpreter fails with *"no kernel image is available for execution on the device"*.
   All results above were produced with the project `.venv` interpreter.
2. **Generation quality is meaningless here.** The model is randomly initialized; `generate` is
   verified for shape, determinism, cropping, EOS stopping, and parameter immutability only.
3. **Absolute positions do not extrapolate.** Learned absolute positional embeddings cap the model
   at 512 tokens; `generate` crops the context rather than shifting positions, so tokens falling out
   of the window are dropped from the model's view (standard for this design).
4. **No attention-pattern inspection.** Causal correctness is established by the future-token
   perturbation tests, not by visualizing attention matrices.
5. **Checkpoint stores no optimizer state.** Only the model `state_dict`; resuming real training will
   need an optimizer-state format extension in the training phase.
6. **One batch, one step.** The smoke test proves the CUDA path works end-to-end — it is not evidence
   about training dynamics, throughput, or memory at 512-token contexts.
7. **Untied configuration is supported but untested at scale.** `tie_embeddings: false` is handled in
   config, parameter estimate, and breakdown, but the locked configuration enables tying.

---

## Verdict

**TRANSFORMER IMPLEMENTED AND VERIFIED**

No pretraining, instruction tuning, or chatbot interface was started. Awaiting review before the
training phase.
