# PolyCodeGPT Tokenizer Finalization Audit (Phase 3.1)

**Date:** 2026-11-15
**Artifact under audit:** `tokenizer/artifacts/polycodegpt_tokenizer.json`
**No downloads, no installs, no raw-data changes, no Transformer work.**
**Verdict:** `TOKENIZER FINALIZED WITH WARNINGS`

---

## 1. Actual tokenizer inspection

Loaded with `tokenizers.Tokenizer.from_file` (library 0.22.2, Python 3.11.9).

| Property | Value |
|---|---|
| `tokenizer.get_vocab()` size | **16,000** |
| Model vocabulary entries (`model.vocab`) | **16,000** |
| of which base (non-special) entries | **15,989** |
| Added tokens | **11** (the special tokens, IDs 0–10, also present in `model.vocab`) |
| Merge rules (`model.merges`) | **15,738** (all unique pairs) |
| Token IDs | contiguous 0 … 15,999 |
| Model type | `BPE` |
| `unk_token` | **`<|unk|>`** (ID 1) ✅ consistent |
| `fuse_unk` / `byte_fallback` / `ignore_merges` | `false` / `false` / `false` |
| `continuing_subword_prefix` / `end_of_word_suffix` | `null` / `null` |
| Normalizer | **`null`** (none) |
| Pre-tokenizer | `ByteLevel` (`add_prefix_space=false`, `use_regex=true`) |
| Decoder | `ByteLevel` |
| Post-processor | `null` |
| Truncation / padding | not configured (none) |

### Vocabulary arithmetic (exact)

```
256  single-byte tokens (ByteLevel initial alphabet, one per byte value)
+15,738 merge rules learned by BpeTrainer
-     5 merge rules whose product is already a special-token string
+    11 special tokens
= 16,000 model vocabulary entries
```

Verification details:

* 256 base entries have length 1 → exactly the byte alphabet; no byte is missing.
* 15,738 merge products are all multi-character, all distinct, and all present in
  `model.vocab` (0 products missing).
* **5 merges reproduce a special-token string**, so they map onto an existing entry instead of
  adding one: `['<|eos','|>']→<|eos|>`, `['<|code','|>']→<|code|>`,
  `['<|docstring','|>']→<|docstring|>`, `['<|lang:python','|>']→<|lang:python|>`,
  `['<|lang:javascript','|>']→<|lang:javascript|>` (merge indices 163, 167, 168, 222, 496).
  These rules are **inert**: added tokens are matched before BPE runs, so they can never fire.
* Arithmetic check: 256 + 15,738 − 5 + 11 = **16,000** ✅
* Equivalently: 15,989 base entries = 256 bytes + 15,733 distinct learned tokens
  (15,738 merge rules − 5 collisions).

## 2. Special tokens

| Token | ID | Single ID when encoded? |
|---|---|---|
| `<|pad|>` | 0 | ✅ |
| `<|unk|>` | 1 | ✅ |
| `<|eos|>` | 2 | ✅ |
| `<|code|>` | 3 | ✅ |
| `<|docstring|>` | 4 | ✅ |
| `<|text|>` | 5 | ✅ |
| `<|lang:python|>` | 6 | ✅ |
| `<|lang:javascript|>` | 7 | ✅ |
| `<|lang:en|>` | 8 | ✅ |
| `<|lang:fr|>` | 9 | ✅ (reserved, never emitted) |
| `<|lang:ar|>` | 10 | ✅ (reserved, never emitted) |

All 11 are unique IDs, registered in `added_tokens` with `special: true`, and each encodes to
exactly one token (`encode("<|eos|>").ids == [2]`, etc.).

## 3. Serializer audit

`src/polycodegpt/tokenization/serializer.py` **version 1.0.0** — executable output verified
against the required formats, not just documentation:

| Kind | Required | Actual output | Match |
|---|---|---|---|
| Python | `<|lang:python|><|docstring|>{docstring}<|code|>{code}<|eos|>` | identical | ✅ |
| JavaScript | `<|lang:javascript|><|docstring|>{docstring}<|code|>{code}<|eos|>` | identical | ✅ |
| Documentation | `<|lang:en|><|text|>{rst}<|eos|>` | identical, raw RST byte-for-byte | ✅ |

Checked cases: non-empty docstring, empty docstring (`<|docstring|><|code|>` still emitted),
docstring identical to code (kept twice — no silent deduplication), documentation file
`Doc/README.rst` reproduced verbatim (4,889 chars), and the 500-file inventory
(`partition_assignment = train_only`, no `Doc/license.rst`).

**The serializer was already correct — no correction was required, so the version stays 1.0.0
and no retraining was performed.**

## 4. Unknown token consistency

`model.unk_token == "<|unk|>"` and `<|unk|>` has ID 1 in both `model.vocab` and
`added_tokens`. Because the ByteLevel alphabet covers all 256 byte values, `unk` is never
emitted: 0 occurrences across 5,000 sampled training records and across the Arabic, French,
emoji, CJK and combining-mark tests. No change needed.

## 5. Tests

Full suite: **28 passed, 0 failed** (0.38 s).

* `tests/unit/test_tokenizer.py` — 13 tests (A–M: special tokens, unique IDs, Python/JS,
  whitespace, operators, English, French, Arabic, emoji/Unicode, round-trip, empty, long).
* `tests/unit/test_serializer.py` — 15 tests added in this audit:
  1. Python serialization exact string
  2. JavaScript serialization exact string
  3. Documentation serialization exact string
  4. Empty docstring keeps both markers
  5. Docstring == code is not deduplicated
  6. Every code marker appears exactly once, in order
  7. Reserved `<|lang:fr|>`/`<|lang:ar|>` never emitted
  8. Special-token table and serializer version fixed
  9. Inventory document serialized verbatim (500 files, no `license.rst`)
  10. Marker boundaries at token-ID level (lang = id[0], docstring = id[1], eos = last,
      `<|code|>` exactly once, docstring/code spans decode back to the original fields)
  11. Documentation markers at token-ID level (`<|lang:en|>`, `<|text|>`, `<|eos|>`)
  12–14. Unicode payloads (French accents, Arabic, emoji/ligature/CJK/combining) round-trip
  15. Save/reload parity (`from_str` and `from_file` produce identical vocabulary and IDs)

Additional checks run directly against the artifact: 5,000 training records, **0 structural
anomalies, 0 round-trip failures, 0 `unk` occurrences** (3.985 chars/token on that sample).

## 6. Training inputs and non-interference

* No retraining was performed, so no new input manifest was needed and no backup was required;
  the existing `tokenizer/artifacts/training_input_manifest.csv` still describes the only
  training run, with 501 rows (1 combined JSONL + 500 documents), all `partition = train`.
* `train.jsonl` SHA-256 still `2a9c71607a1893711cd7f2963ec172784415f838ec1c633c988949b45e345c74`.
* `valid.jsonl` and `test.jsonl` were **not read by this audit**; their last access (17:03:37
  local) is the post-training SHA-256 integrity check, after the training window
  (17:01:49–17:02:23 local). The trainer contains a hard guard that refuses any path named
  `valid.jsonl`/`test.jsonl`.

## 7. Artifact hashes (unchanged by this audit)

| Artifact | Size | SHA-256 |
|---|---|---|
| `tokenizer/artifacts/polycodegpt_tokenizer.json` | 1,095,495 B | `f1feba02fcb9f5c85b8f90b610f5ccb350ba2950ecc1fd3847a74562ca2a6e79` |
| `tokenizer/artifacts/tokenizer_metadata.json` | 40,668 B | `401e8ab6b0ed0a6f983184c75881490971651790850681280c5f2e565e84675e` |
| `tokenizer/artifacts/training_input_manifest.csv` | 101,700 B | `b00ff0ee1f012520679533ada2cdde5ebfa4685af2c20ae582d0275e38c0084b4` |

## 8. Findings and warnings

1. **Five inert merge rules** shadow special tokens (indices 163, 167, 168, 222, 496). They
   cannot fire because added tokens are matched first; they only explain the arithmetic
   (16,005 nominal vs 16,000 actual). Removing them would change the artifact hash without
   changing any encoding, so they were left in place.
2. **Nine marker-fragment tokens** (`<|`, `|>`, `<|eos`, `<|code`, `<|lang`, `<|lang:`,
   `<|docstring`, `<|lang:python`, `<|lang:javascript`) occupy vocabulary slots, involved in 12
   merges (0.08% of 15,738). Cause: the serializer's own control strings were present in the
   training text and the BPE trainer counted their fragments. Cost is negligible; no retraining
   warranted.
3. **Control-token literals in raw text would be interpreted as markers.** The serializer does
   not escape `<|…|>` sequences. The training corpus contains **0** literal occurrences of any
   of the 11 special tokens (verified across all 392,632 records; 88 records contain `<|` or
   `|>` in other contexts, none forming a control token). This remains a theoretical risk for
   unseen valid/test text, mitigable at dataset-encoding time.
4. **No RNG seed in `tokenizers.BpeTrainer`** — determinism holds for identical input order and
   library version; seed 1234 is metadata only.
5. **Artifacts are Git-ignored** (`tokenizer/artifacts/*`), including the 100 KB input
   manifest; hashes are preserved in this report and in the training report.
6. **Provenance limitation carried forward**: CodeSearchNet inputs are Zenodo v1
   (`10.5281/zenodo.7908468`); S3 `v2` parity remains unverified.

## 9. Final verdict

`TOKENIZER FINALIZED WITH WARNINGS`

The 16,000-token byte-level BPE tokenizer is structurally correct, fully accounted for
(256 + 15,738 − 5 + 11 = 16,000), consistent with `<|unk|>`, matched to an exact serializer
(version 1.0.0), reload-stable, and covered by 28 passing tests. Warnings are cosmetic or
documentary: 5 inert merges, 9 marker-fragment slots, unescaped control-token literals in
unseen text, seed documentation, Git-ignored artifacts, and inherited Zenodo provenance.
