# PolyCodeGPT Tokenizer Training Report (Phase 3)

**Date:** 2026-11-15
**Verdict:** `TOKENIZER TRAINED WITH WARNINGS` (warnings are limited to determinism documentation and tokenizer coverage gaps, not failures)

---

## 1. Input sources

| Input | Path | Records / files | SHA-256 |
|---|---|---|---|
| Combined training partition | `data/processed/polycodegpt_corpus/train.jsonl` | **392,632** records | `2a9c71607a1893711cd7f2963ec172784415f838ec1c633c988949b45e345c74` |
| Python documentation | 500 files listed in `data/manifests/python_docs_training_corpus_inventory.csv` | **500** files | per-file SHA-256 recorded in the manifest |
| Evaluation partitions | `valid.jsonl`, `test.jsonl` | **not opened** | — |

Inputs were verified against their recorded hashes before a single byte was tokenized:
`train.jsonl` matched the combined-corpus manifest, and all 500 documents matched the
inventory (`Doc/license.rst` is absent from the inventory and is additionally refused by a
guard in `train.py`).

## 2. Training sample counts

| Kind | Count | Serialized characters |
|---|---|---|
| CodeSearchNet records (Python + JavaScript) | 392,632 | 524,737,257 |
| Python documentation files (raw RST) | 500 | 11,109,475 |
| **Total sequences seen by the trainer** | **393,132** | **535,846,732** |

Both counts were measured during the run (`counters` in the streaming generator) and asserted
against the expected values after fitting. Language split of the training partition:
290,463 Python + 102,169 JavaScript records.

## 3. Documentation file count

**500** — exactly the `ELIGIBLE_CANDIDATE` RST files in the verified inventory
(501 `.rst` files minus `Doc/license.rst`). No asset, binary, licence file, root `LICENSE`, or
non-inventory document was included. Raw documentation was read only; nothing was moved,
stripped or modified.

## 4. Serializer (version 1.0.0)

Implemented once in `src/polycodegpt/tokenization/serializer.py` and shared by tokenizer
training, future dataset encoding and model training.

```
Python record      : <|lang:python|><|docstring|>{docstring}<|code|>{code}<|eos|>
JavaScript record  : <|lang:javascript|><|docstring|>{docstring}<|code|>{code}<|eos|>
Documentation file : <|lang:en|><|text|>{raw RST}<|eos|>
```

* Docstrings are **not** labelled `en` — their natural language is unverified.
* Nothing is truncated, stripped, normalised or rewritten; whitespace and indentation are
  preserved byte-for-byte.
* Records whose `language` disagrees with `source_dataset` raise instead of being mislabelled.

## 5. Tokenizer architecture

| Component | Setting |
|---|---|
| Model | `tokenizers.models.BPE(unk_token="<|unk|>")` |
| Pre-tokenizer | `ByteLevel(add_prefix_space=False, use_regex=True)` |
| Decoder | `ByteLevel()` |
| Normalizer | **none** (no NFC/NFKC, no lowercasing) |
| Initial alphabet | `ByteLevel.alphabet()` — all **256** bytes |
| Trainer | `BpeTrainer(vocab_size=16000, min_frequency=2, special_tokens=11, initial_alphabet=256 bytes)` |
| Device | CPU (no CUDA allocation) |
| Library | `tokenizers==0.22.2`, Python 3.11.9 |

Vocab composition: **16,000** IDs = 11 special tokens + 256 single-byte tokens + 15,738
recorded merges (a few merge entries reproduce single-character units, so the merge count is
not exactly 16,000 − 267; the total vocabulary is exact).

## 6. Vocabulary size

| Requested | Achieved |
|---|---|
| 16,000 | **16,000** |

## 7. Special tokens and IDs

| Token | ID | Role |
|---|---|---|
| `<|pad|>` | 0 | padding (loss-masked) |
| `<|unk|>` | 1 | unreachable with byte fallback; kept for safety |
| `<|eos|>` | 2 | end of sequence |
| `<|code|>` | 3 | code-role marker |
| `<|docstring|>` | 4 | docstring-role marker |
| `<|text|>` | 5 | prose/documentation marker |
| `<|lang:python|>` | 6 | language marker |
| `<|lang:javascript|>` | 7 | language marker |
| `<|lang:en|>` | 8 | language marker |
| `<|lang:fr|>` | 9 | **reserved**, never trained on |
| `<|lang:ar|>` | 10 | **reserved**, never trained on |

All 11 registered as added/special tokens and verified to have unique IDs; each encodes to
exactly one ID. No French or Arabic examples were manufactured.

## 8. Streaming and memory

* `train_from_iterator()` over a generator that yields one serialized record at a time;
  `length=393,132` provided for progress reporting.
* No `readlines()`, no list of documents, no corpus materialised in memory.
* Documents are opened one at a time and closed immediately.
* CPU-only: `torch.cuda` is never imported during training, no CUDA memory allocated.

## 9. Training duration and peak RAM

| Metric | Measured |
|---|---|
| Wall clock, whole run (verify + manifest + fit + save) | **34.48 s** |
| Fit only (`train_from_iterator`) | 32.51 s |
| Peak working set (RSS) | **1,052.9 MB** |
| GPU memory | 0 MB |

Peak RSS is read from the Win32 `PROCESS_MEMORY_COUNTERS_EX.PeakWorkingSetSize` of the
training process itself.

## 10. Input manifest

`tokenizer/artifacts/training_input_manifest.csv` — 501 data rows + header
(1 combined JSONL + 500 documents), 101,700 B, every row `partition = train`,
`serializer_version = 1.0.0`. Columns: `input_kind, source_path, source_sha256, record_count,
partition, serializer_version, collection_source`. The manifest contains no evaluation file.

## 11. Test results

`tests/unit/test_tokenizer.py` — **13 passed, 0 failed** (0.31 s).

| Test | Scope | Result |
|---|---|---|
| A | All 11 special tokens registered as added tokens | PASS |
| B | Unique IDs, fixed order 0–10, one ID per special token | PASS |
| C | Python code encode/decode | PASS |
| D | JavaScript code encode/decode | PASS |
| E | Indentation, tabs, leading/trailing whitespace preserved | PASS |
| F | Punctuation and operators (`->`, `=>`, `!==`, `**`, `//`, `??`, template literals) | PASS |
| G | English text | PASS |
| H | French accents (no `<|unk|>`) | PASS |
| I | Arabic text (no `<|unk|>`) | PASS |
| J | Emoji, combining marks, CJK, ZWJ sequences | PASS |
| K | Round-trip reconstruction of serialized Python/JS/doc records | PASS |
| L | Empty input → `[]`, decodes to `""` | PASS |
| M | Long input (≈150k chars) round-trips, no truncation | PASS |

## 12. Example tokenization

Record `ageitgey/face_recognition:examples/face_recognition_knn.py` (Python), 3,591 chars → 958 tokens:

```
<|lang:python|><|docstring|>Trains a k-nearest neighbors classifier for face recognition.
... :param train_dir: directory that contains a sub-directory for each known person ...
```

| | |
|---|---|
| IDs (first 32) | `[6, 4, 3787, 1466, 282, 903, 23, 13773, 8405, 12043, 357, 5639, 10533, 682, 24, 372, 347, 405, 4831, 73, 897, 36, 1546, 572, 2311, 282, 936, 23, 3285, 357, 1104, 4630]` |
| Tokens (first 32) | `['<\|lang:python\|>', '<\|docstring\|>', 'Tra', 'ins', 'Ġa', 'Ġk', '-', 'nearest', 'Ġneighbors', 'Ġclassifier', 'Ġfor', 'Ġface', 'Ġrecogn', 'ition', '.', 'ĊĊĠĠĠ', 'Ġ:', 'param', 'Ġtrain', '_', 'dir', ':', 'Ġdirectory', 'Ġthat', 'Ġcontains', 'Ġa', 'Ġsub', '-', 'directory', 'Ġfor', 'Ġeach', 'Ġknown']` |
| `decode(encode(x)) == x` | **True** |

Second sample (same repository, 2,977 chars) → 819 tokens, round-trip exact.

Compression measured on the first 2,000 training records: **4.26 chars/token** mean
(median 4.28, min 2.56, max 6.05). Indentation is learned as merged units (`ĊĊĠĠĠ` = two
newlines + three spaces), so code layout is cheap to encode. These are measurements, not
projections.

## 13. Saved artifacts

| Artifact | Size | SHA-256 |
|---|---|---|
| `tokenizer/artifacts/polycodegpt_tokenizer.json` | 1,095,495 B | `f1feba02fcb9f5c85b8f90b610f5ccb350ba2950ecc1fd3847a74562ca2a6e79` |
| `tokenizer/artifacts/tokenizer_metadata.json` | 40,668 B | `401e8ab6b0ed0a6f983184c75881490971651790850681280c5f2e565e84675e` |
| `tokenizer/artifacts/training_input_manifest.csv` | 101,700 B | `b00ff0ee1f012520679533ada2cdde5ebfa4685af2c20ae582d0275e38c008b4` |

Code: `src/polycodegpt/tokenization/{__init__,serializer,train}.py`,
`scripts/train_tokenizer.py`, `tests/unit/test_tokenizer.py`, `tests/conftest.py`.

Metadata records algorithm, requested/actual vocab size, special tokens and IDs, library
version, every training parameter, input counts and hashes, durations, peak RAM, date,
serializer version, seed and the determinism note.

## 14. Limitations

1. **No RNG seeding in `tokenizers`.** `BpeTrainer` exposes no seed and performs no random
   sampling; it counts n-grams in stream order and breaks ties deterministically. Seed 1234 is
   recorded for the wider pipeline but has no effect on fitting. Determinism holds for identical
   input order and identical `tokenizers` version.
2. **Single training run** — the tokenizer was fitted once; no ablation of vocab size or
   `min_frequency` was performed.
3. **Coverage is empirical, not guaranteed semantic.** French/Arabic/emoji round-trip because of
   the byte-level alphabet, not because the vocabulary learned their morphology; the reserved
   `<|lang:fr|>`/`<|lang:ar|>` tokens are untrained and currently carry no meaning.
4. **Docstrings are unlabelled** for natural language (only programming-language markers), so the
   model cannot distinguish English from other prose in docstrings at this stage.
5. **No padding/truncation policy yet** — sequence packing and max-length are model-phase
   decisions; the tokenizer itself never truncates.
6. **Artifacts are Git-ignored** (`tokenizer/artifacts/*`), including the 100 KB input manifest;
   this report carries the hashes needed to audit which inputs were used.
7. **Source provenance unchanged**: the CodeSearchNet inputs are Zenodo v1; S3 `v2` parity
   remains unverified.

## 15. Reproducibility

```bash
cd "C:\Users\UltraPc\Desktop\LLM FROM SCRATCH\PolyCodeGPT"
.venv\Scripts\python.exe scripts\train_tokenizer.py            # 34 s, CPU only
.venv\Scripts\python.exe -m pytest tests/unit/test_tokenizer.py -v
```

The script re-verifies every input hash before training, rewrites
`tokenizer/artifacts/training_input_manifest.csv`, and atomically replaces the tokenizer and
metadata files. Re-running on the same inputs with `tokenizers==0.22.2` reproduces the same
vocabulary.

## 16. Validation summary

| Check | Result |
|---|---|
| Only the combined train partition was read | ✅ `train.jsonl` atime updated during the run; guard rejects `valid.jsonl`/`test.jsonl` by name |
| `valid.jsonl` / `test.jsonl` never opened by the trainer | ✅ atime at 16:52 and 17:03 local only: the 16:52 access (before training) and the 17:03:37 access (after training ended 17:02:23) are SHA-256 integrity checks, not training; the trainer has no code path that opens them |
| Exactly 500 eligible documents considered | ✅ 500 manifest rows, all `partition=train` |
| No excluded document included | ✅ `Doc/license.rst` absent and hard-refused |
| Every input hash matches its source manifest | ✅ `train.jsonl` + 500/500 documents |
| No synthetic data | ✅ only corpus records and raw RST |
| No pretrained tokenizer/model downloaded | ✅ built from an empty BPE model, `initial_alphabet` only |
| 11 special tokens with distinct IDs | ✅ 0–10 |
| Saved tokenizer reloads | ✅ `Tokenizer.from_file` → vocab 16,000 |
| Tokenizer tests | ✅ 13/13 passed |
| PyTorch / CUDA | ✅ 2.15.0+cu132, CUDA 13.2, RTX 5060 Laptop available |
| Original datasets unchanged | ✅ all three corpus SHA-256 still match the manifest |
| Git ignore status | ✅ all three artifacts IGNORED by `tokenizer/artifacts/*` |

**Verdict: `TOKENIZER TRAINED WITH WARNINGS`** — the tokenizer is trained, reloadable and
fully tested; the warnings concern documented determinism limits, single-run validation, and
Git-ignored artifacts rather than any correctness failure.
