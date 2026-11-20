# Phase 8 — Validation Loss Decomposition and Token Distribution Analysis

Inference-only diagnostic. No training, no checkpoint/tokenizer/dataset modification,
no test data accessed.

## 1. Data boundaries

- Validation records: `data/processed/polycodegpt_corpus/valid.jsonl` (validation partition only).
- Validation stream: `data/processed/tokenized_v1_1/valid.bin`.
- Training records (distribution only): `data/processed/polycodegpt_corpus/train.jsonl`.
- Checkpoint: `C:\Users\UltraPc\Desktop\LLM FROM SCRATCH\PolyCodeGPT\checkpoints\polycodegpt_first_pass\step_008130.pt`.
- Tokenizer: `tokenizer/artifacts/polycodegpt_tokenizer_v1_1.json` (v1.1.0).
- Model config: `configs/model.yaml`; training config: `configs/training.yaml` (precision auto).
- Test partition files were never opened or referenced.

## 2. Reconstruction verification

| field | value |
|---|---|
| reconstructed tokens | 7757005 |
| valid.bin tokens | 7757005 |
| exact match | True |
| first mismatch index | none |
| valid.jsonl sha256 | 2eb02406d22101716214a6509314e9d7e1596e1cc6640d0eb4a886c00fb715af |
| valid.bin sha256 | ee55019be57e1b53a0a74b232ae6d37b5e2935c4b6af95f70221e714fa7e52df |
| manifest valid tokens | 7757005 |

Reconstruction serializes each record with `serializer.serialize_code_record`,
encodes with `tokenizer.encode(..., add_special_tokens=False)` in batched order,
and concatenates. No separators are inserted between records (matches prepare.py).

## 3. Window geometry and target counting

| field | value |
|---|---|
| context_length | 512 |
| window_length | 513 |
| stride | 512 |
| overlap | 1 |
| blocks | 15150 |
| tokens in blocks | 7756801 |
| dropped trailing tokens | 204 |
| target tokens (blocks * context_length) | 7756800 |

Every target token is counted exactly once: target ranges tile
`1 .. blocks*stride` contiguously; position 0 is context only.

## 4. Category assignment method

Each serialized record has the fixed marker layout
`<|lang:X|><|docstring|>{doc}<|code|>{code}<|eos|>`. Token IDs between
`<|docstring|>` and `<|code|>` are docstring content; IDs between
`<|code|>` and `<|eos|>` are code content; the four markers are
structural/special. Language is taken from the leading marker. Categories
are assigned by marker offsets in the complete serialized stream, never by
independently tokenizing fragments.

## 5. Loss decomposition

| category | tokens | token share | loss sum | mean CE | perplexity |
|---|---:|---:|---:|---:|---:|
| python_docstring | 1345519 | 17.3463% | 4492629.241702 | 3.338956 | 28.1897 |
| python_code | 4409459 | 56.8464% | 9577991.652205 | 2.172147 | 8.7771 |
| javascript_docstring | 325490 | 4.1962% | 1069570.143759 | 3.286031 | 26.7365 |
| javascript_code | 1592479 | 20.5301% | 3781622.770419 | 2.374677 | 10.7475 |
| structural_special | 83853 | 1.0810% | 52733.160481 | 0.628876 | 1.8755 |
| combined python | 5754978 | 74.1927% | 14070620.893907 | 2.444948 | 11.5299 |
| combined javascript | 1917969 | 24.7263% | 4851192.914178 | 2.529339 | 12.5452 |
| combined docstring | 1671009 | 21.5425% | 5562199.385462 | 3.328647 | 27.9006 |
| combined code | 6001938 | 77.3765% | 13359614.422624 | 2.225883 | 9.2617 |
| overall | 7756800 | 100.0000% | 18974546.968566 | 2.446182 | 11.5442 |

### Consistency with existing full-validation result

- Existing full-validation loss (step 8130): 2.446182.
- Recomputed overall loss: 2.446182.
- Difference: 0.000000.
Any non-zero difference is attributable to floating-point reduction order (per-token `reduction='none'` summed in float64 vs. the batched mean in the established evaluator) and to autocast accumulation; model weights are unchanged.

## 6. Token distribution analysis

### train.jsonl

| metric | value |
|---|---:|
| records | 392632 |
| total tokens | 130345937 |
| python docstring tokens | 20345263 |
| javascript docstring tokens | 5490813 |
| python code tokens | 77300021 |
| javascript code tokens | 25639312 |
| total code tokens | 102939333 |
| median code length (tokens) | 158.00 |
| p90 code length (tokens) | 520.00 |
| p99 code length (tokens) | 1606.00 |
| max code length (tokens) | 624832 |
| longest 1% records | 3926 |
| token mass from longest 1% (code tokens) | 12.4032% |
| bundle/minified-like records | 18 |
| token mass from bundle/minified-like (code tokens) | 0.7132% |
| high-repetition records | 41403 |
| token mass from high-repetition (code tokens) | 36.2671% |

### valid.jsonl

| metric | value |
|---|---:|
| records | 20965 |
| total tokens | 7757005 |
| python docstring tokens | 1345519 |
| javascript docstring tokens | 325521 |
| python code tokens | 4409459 |
| javascript code tokens | 1592646 |
| total code tokens | 6002105 |
| median code length (tokens) | 161.00 |
| p90 code length (tokens) | 573.00 |
| p99 code length (tokens) | 1995.44 |
| max code length (tokens) | 20355 |
| longest 1% records | 209 |
| token mass from longest 1% (code tokens) | 13.3055% |
| bundle/minified-like records | 0 |
| token mass from bundle/minified-like (code tokens) | 0.0000% |
| high-repetition records | 2380 |
| token mass from high-repetition (code tokens) | 40.1793% |

Heuristics (documented):
- bundle/minified-like: code token count >= 512 AND newline count <= max(1, code_tokens // 200).
- high repetition: unique-token ratio in code tokens < 0.30.
- longest 1%: top 1% of records by code token count.
- token mass percentages are relative to total code tokens of that split.

## 7. Model-weight integrity

- digest before: n/a
- digest after: n/a
- weights changed: n/a

## 8. Reproduction command (Windows CMD, project venv)

```cmd
cd /d "C:\Users\UltraPc\Desktop\LLM FROM SCRATCH\PolyCodeGPT"
"C:\Users\UltraPc\Desktop\LLM FROM SCRATCH\.venv\Scripts\python.exe" scripts\evaluate_validation_loss_breakdown.py --device cuda
```

Use `--verify-only` to run reconstruction + distribution without GPU.

## 9. Limitations

- Loss decomposition requires a CUDA GPU and the step-8130 checkpoint; on CPU-only hosts the
  script writes reconstruction and distribution results and marks loss as pending.
- Per-token `reduction='none'` aggregation may differ from the established batched mean by
  a small floating-point epsilon; differences are reported, not adjusted.
- Token distribution heuristics are documented approximations, not ground-truth labels.
- No test data was accessed.

Generated in 290.9 s.
