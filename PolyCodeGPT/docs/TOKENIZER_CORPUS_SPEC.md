# Tokenizer Corpus Specification (PolyCodeGPT)

**Status:** implemented and retrained with serializer 1.1.0 — see
`reports/tokenizer_finalization_v1_1_report.md` for the measured run
(16,000 vocab, 35.41 s, 1,051.4 MB peak RSS).
**Serializer version:** 1.1.0 (`src/polycodegpt/tokenization/serializer.py`), adds reversible
control-token escaping on top of 1.0.0.
**Tokenizer library:** `tokenizers==0.22.2`

---

## 1. Inputs

| Corpus | Path | Records | Role |
|---|---|---|---|
| Combined training partition | `data/processed/polycodegpt_corpus/train.jsonl` | 392,632 (290,463 Python + 102,169 JavaScript) | model **and** tokenizer training |
| Python documentation (English) | 500 files from `data/manifests/python_docs_training_corpus_inventory.csv` | 500 docs, 11,112,983 serialized chars | model **and** tokenizer training (train-only) |
| Validation partition | `data/processed/polycodegpt_corpus/valid.jsonl` | 20,965 | **model evaluation only** |
| Test partition | `data/processed/polycodegpt_corpus/test.jsonl` | 21,080 | **model evaluation only** |

Hard rules (enforced in code, not just policy):

1. **Only the training partition may train the tokenizer.** `valid.jsonl` and `test.jsonl` are
   never opened; `train.py` refuses any path named `valid.jsonl`/`test.jsonl` before opening it.
2. Repository-level partition assignment is global (`TEST > VALID > TRAIN`), so no repository
   contributes to two partitions.
3. The documentation corpus is train-only and never enters validation or test.
4. Inputs are hash-verified against their manifests before training; a mismatch aborts the run.

## 2. Fields that will be tokenized

| Field | Content | Tokenized? |
|---|---|---|
| `code` | full function source, indentation and line breaks preserved | **yes** |
| `docstring` | raw docstring text as published | **yes** |
| `path`, `url`, `repo`, `func_name`, `sha`, `code_tokens`, `docstring_tokens` | metadata | **no** |

Both fields are emitted; unlike the first draft of this spec, the docstring is **not** dropped
when it also appears inside `code`. Duplication is accepted because the `<|docstring|>` /
`<|code|>` role markers make the two segments distinguishable, and dropping text would have
made the stream depend on a substring test.

No record is truncated or concatenated at the tokenizer stage; packing happens later, in the
dataset loader.

## 3. Serialized training streams (exact, version 1.1.0)

```
Python record      : <|lang:python|><|docstring|>{docstring}<|code|>{code}<|eos|>
JavaScript record  : <|lang:javascript|><|docstring|>{docstring}<|code|>{code}<|eos|>
Documentation file : <|lang:en|><|text|>{raw RST}<|eos|>
```

* There is **no** `<|bos|>` token: a sequence begins with its `<|lang:…|>` marker.
* `<|docstring|>` is always emitted, even when the docstring is empty.
* Newlines inside `code`, `docstring` and RST are preserved verbatim (`\n`); indentation is
  never normalised.
* One logical unit per record; documentation is one unit per file.
* Records whose `language` field disagrees with `source_dataset` raise instead of being
  silently relabelled.

### 3.1 Content escaping (1.1.0)

Content fields are escaped before insertion so a literal control token in code or prose can
never be confused with a structural marker, and the tokenizer's added-token matcher can never
emit a structural ID from user content:

| Rule | Example |
|---|---|
| `\` → `\\` | `r'\d+'` becomes `r'\\d+'` |
| `<|` of a registered control token → `<\|` | `<|eos|>` becomes `<\|eos|>` |

`unescape_content` reverses both in one pass (`\\`→`\`, `\|`→`|`). The mapping is bijective,
so `parse_serialized(serialize(x))` returns the original `docstring`/`code`/`text` exactly.
Cost on the training corpus: 243,801 backslashes (0.05% of characters) and 0 control-token
literals, i.e. +243,801 characters in the code stream and +3,508 in the documentation stream.

## 4. Programming-language markers

| Marker | Assigned to |
|---|---|
| `<|lang:python|>` | every CodeSearchNet Python record |
| `<|lang:javascript|>` | every CodeSearchNet JavaScript record |
| `<|lang:en|>` | the English documentation corpus |
| `<|lang:fr|>` | reserved — no training data, never emitted |
| `<|lang:ar|>` | reserved — no training data, never emitted |

Markers are assigned **per record**, not per document. Docstrings are deliberately **not**
marked as English: their natural language has not been independently verified.

## 5. Docstring representation

* Emitted as the raw `docstring` field (CodeSearchNet has already dedented it), preceded by
  `<|docstring|>`.
* No quotes, prefixes, trailing whitespace, paraphrasing, normalisation or translation is added.
* Empty docstrings are kept as an empty segment so "undocumented function" is a learnable pattern.

## 6. Special tokens (11, fixed IDs)

| Token | ID | Purpose |
|---|---|---|
| `<|pad|>` | 0 | batch padding (loss-masked) |
| `<|unk|>` | 1 | unreachable via byte fallback; kept for safety |
| `<|eos|>` | 2 | sequence end |
| `<|code|>` | 3 | code-role marker |
| `<|docstring|>` | 4 | docstring-role marker |
| `<|text|>` | 5 | prose/documentation-role marker |
| `<|lang:python|>` | 6 | language marker |
| `<|lang:javascript|>` | 7 | language marker |
| `<|lang:en|>` | 8 | language marker |
| `<|lang:fr|>` | 9 | reserved |
| `<|lang:ar|>` | 10 | reserved |

Vocabulary size: **16,000** tokens total = 256 byte-alphabet tokens + 15,738 merge rules − 5
merges that rebuild an existing special token + 11 special tokens.

## 7. Algorithm and normalization

* **Byte-level BPE**: `models.BPE(unk_token="<|unk|>")` with
  `pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)` and
  `decoders.ByteLevel()`.
* `initial_alphabet = pre_tokenizers.ByteLevel.alphabet()` → all 256 byte values, giving
  complete UTF-8 coverage and the required byte fallback (no `<|unk|>` at encode time).
* **No normalizer**: no NFC/NFKC (would rewrite characters inside string literals), no
  lowercasing (Python and JavaScript are case-sensitive).
* `BpeTrainer(vocab_size=16000, min_frequency=2, special_tokens=<the 11>, show_progress=True)`.
* Training on CPU only; `train_from_iterator(..., length=393,132)` for bounded memory.
* **Determinism:** `BpeTrainer` exposes no RNG seed and performs no random sampling — it counts
  n-grams in stream order and breaks frequency ties deterministically. Seed `1234` is recorded in
  the metadata for the wider pipeline (sampling/init) but has no effect on fitting. Reproducible
  for identical input order and identical `tokenizers` version.

## 8. What the tokenizer must handle (and does)

| Requirement | Mechanism / evidence |
|---|---|
| Python syntax (`def`, `:`, indentation, docstrings) | learned from 290,463 Python functions; 4.26 chars/token on training samples |
| JavaScript syntax (`=>`, `function`, template literals) | learned from 102,169 JS functions |
| English prose | docstrings + 11.1M chars of documentation |
| Future French / Arabic | `<|lang:fr|>` / `<|lang:ar|>` reserved; byte alphabet already covers Latin-1 accents and Arabic script (round-trip tested, no `<|unk|>`) |
| UTF-8 byte fallback | 256-byte base alphabet; tested with emoji, ZWJ sequences, CJK, combining marks |

Byte-level encodability of French and Arabic is a vocabulary property, **not** evidence of
language-model proficiency.

## 9. Train / valid / test discipline

| Stage | Reads `train.jsonl` | Reads `valid.jsonl` | Reads `test.jsonl` | Documentation |
|---|---|---|---|---|
| Tokenizer training | ✅ yes (392,632 records) | ❌ never | ❌ never | ✅ 500 files |
| Model training | ✅ yes | ❌ no | ❌ no | ✅ train-only |
| Model validation | ❌ no | ✅ yes | ❌ no | ✅ optional holdout |
| Final evaluation | ❌ no | ❌ no | ✅ once | ✅ optional holdout |

The training run writes `tokenizer/artifacts/training_input_manifest_v1_1.csv` with one row per input
(1 JSONL + 500 documents), each `partition = train`, each with its SHA-256 — the audit trail
proving what was fitted on.

## 10. Artefacts produced

| Artefact | Contents |
|---|---|
| `tokenizer/artifacts/polycodegpt_tokenizer_v1_1.json` | full tokenizer (model + pre-tokenizer + decoder + added tokens); byte-identical copy at `polycodegpt_tokenizer.json` |
| `tokenizer/artifacts/tokenizer_metadata_v1_1.json` | algorithm, vocab requested/actual, base entries, merge count, unk config, special tokens + IDs, library versions, training parameters, input counts/hashes, serializer version + SHA-256, durations, peak RAM, date, seed note |
| `tokenizer/artifacts/training_input_manifest_v1_1.csv` | per-input audit manifest (kind, path, SHA-256, record count, partition, serializer version + SHA-256, collection source, documentation eligibility) |
| `tokenizer/artifacts/backups/phase3_before_serializer_fix/` | pre-1.1.0 tokenizer, metadata, manifest and serializer source |
| `src/polycodegpt/tokenization/serializer.py` | the single shared serializer (version 1.1.0) with escaping and `parse_serialized` |
| `scripts/train_tokenizer.py` | executable entry point (CPU only) |
| `tests/unit/test_tokenizer.py` | 13 tests (registration, IDs, Python/JS, whitespace, operators, English, French, Arabic, emoji, round-trip, empty, long) |

## 11. Non-goals

* No synthetic or pretrained content: the tokenizer was built from an empty BPE model.
* No token counts are estimated in any report; all reported figures are measured from the run.
* The Transformer and language model are out of scope for this phase.
