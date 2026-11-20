# PolyCodeGPT — Serialization Correction and Tokenizer Finalization (Phase 3.2, v1.1.0)

**Date:** 2026-11-15
**Scope:** serializer 1.1.0 with reversible control-token escaping, retrained tokenizer,
versioned artifacts and manifest. No model/Transformer work.
**Verdict:** `TOKENIZER FINALIZED WITH WARNINGS`

---

## 1. Backup location

`tokenizer/artifacts/backups/phase3_before_serializer_fix/`

| File | Size | SHA-256 |
|---|---|---|
| `polycodegpt_tokenizer.json` (v1.0.0) | 1,095,495 B | `f1feba02fcb9f5c85b8f90b610f5ccb350ba2950ecc1fd3847a74562ca2a6e79` |
| `tokenizer_metadata.json` (v1.0.0) | 40,668 B | `401e8ab6b0ed0a6f983184c75881490971651790850681280c5f2e565e84675e` |
| `training_input_manifest.csv` (v1.0.0) | 101,700 B | `b00ff0ee1f012520679533ada2cdde5ebfa4685af2c20ae582d0275e38c0084b4` |
| `serializer_v1_0_0.py` (source snapshot) | 6,231 B | `0d07accfa8dc46e2c66c0381dab784647bf2aaf5c0fc010d707f6c847dd29f40` |
| `BACKUP_MANIFEST.json` | — | lists the four files above |

Nothing was overwritten in the backup directory.

## 2. New serializer version

**1.1.0** — `src/polycodegpt/tokenization/serializer.py`,
SHA-256 `dd4d3712541fe9969142840ddb5ec1a0af16a1ff7199b3307cbeea217fe8a3fa`.

Public API:

| Function | Purpose |
|---|---|
| `serialize_python_record(record)` | Python stream |
| `serialize_javascript_record(record)` | JavaScript stream |
| `serialize_documentation(rst)` | documentation stream |
| `serialize_code_record(record)` | language-dispatched convenience entry point |
| `escape_content(text)` / `unescape_content(text)` | reversible escaping |
| `parse_serialized(text)` | recovers `docstring`/`code` or documentation `text`; raises on malformed streams |

**Escaping rules** (applied to content only, never to structural markers):

| Rule | Example |
|---|---|
| `\` → `\\` | `r'\d+'` → `r'\\d+'` |
| opening `<\|` of a registered control token → `<\\\|` | `<\|eos\|>` → `<\\\|eos\|>` |

The second rule *breaks the control-token literal*, so the tokenizer's added-token matcher can
never produce a structural ID from user content — verified at token level, not just text level.
Unescaping is a single left-to-right pass (`\\`→`\`, `\|`→`|`).

## 3. Actual serialization examples

```
<|lang:python|><|docstring|>Match '\\d+' literally; mentions <\|eos|>.<|code|>import re
PATTERN = re.compile(r'\\d+')

def f(s, other):
    return s | other
<|eos|>
```

(Rendered above with doubled backslashes, exactly as stored.)

| Check | Result |
|---|---|
| `parse_serialized(...)["docstring"]` equals source | ✅ |
| `parse_serialized(...)["code"]` equals source (byte-identical) | ✅ |
| Marker IDs in the encoded stream | docstring 1×, code 1×, eos 1× |

Empty docstring: `<|lang:python|><|docstring|><|code|>…<|eos|>` (both markers kept).
Documentation: `<|lang:en|><|text|>{raw RST}<|eos|>`.

## 4. Special token ID mapping

| Token | ID | Encodes to one ID |
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
| `<|lang:fr|>` | 9 | ✅ (reserved) |
| `<|lang:ar|>` | 10 | ✅ (reserved) |

BPE `unk_token = "<|unk|>"` (ID 1), `fuse_unk=false`, `byte_fallback=false`; the 256-byte
alphabet means `<|unk|>` is never emitted (0 occurrences in 5,000 sampled records and all
Unicode/escaping tests).

## 5. Vocabulary accounting

```
256   byte-alphabet tokens
+15,738 merge rules
-     5 merges whose product is an existing special-token string
+    11 special tokens
= 16,000 model vocabulary entries (IDs contiguous 0…15,999)
```

Requested 16,000 → achieved **16,000**. Base (non-special) entries 15,989 = 256 + 15,733
distinct learned tokens. The same 5 inert merges as in v1.0.0 reappear
(`['<|eos','|>']`, `['<|code','|>']`, `['<|docstring','|>']`, `['<|lang:python','|>']`,
`['<|lang:javascript','|>']`); they cannot fire because added tokens are matched first.

## 6. Input manifest and hashes

`tokenizer/artifacts/training_input_manifest_v1_1.csv` — 150,586 B, **502 data rows**,
SHA-256 `9322655c4ecc65fb4ca3e95664eb7572baa39230170586084fbab98a5f2bfcc7`.

| input_kind | Rows | partition | SHA-256 |
|---|---|---|---|
| `combined_train_jsonl` | 1 | train | `2a9c71607a1893711cd7f2963ec172784415f838ec1c633c988949b45e345c74` |
| `serializer_source` | 1 | train | `dd4d3712541fe9969142840ddb5ec1a0af16a1ff7199b3307cbeea217fe8a3fa` |
| `documentation_rst` | 500 | train | 500 individual hashes (all match the inventory) |

Columns: `input_kind, source_path, source_sha256, record_count, partition, serializer_version,
serializer_sha256, collection_source, documentation_eligibility`. Every row has
`partition = train`, `serializer_version = 1.1.0`; documentation rows carry
`ELIGIBLE_CANDIDATE (train_only)`. No validation or test input appears.

## 7. Tokenizer training time

**35.41 s** wall clock (fit 32.5 s), CPU only, 393,132 sequences (392,632 records + 500 docs).

Training text after escaping: 524,981,058 code-stream chars (+243,801) and 11,112,983
documentation chars (+3,508) — a +0.05% expansion from backslash doubling.

## 8. Peak RAM

**1,051.4 MB** working set (`PROCESS_MEMORY_COUNTERS_EX.PeakWorkingSetSize`); 0 MB GPU.

## 9. New artifact sizes and hashes

| Artifact | Size | SHA-256 |
|---|---|---|
| `tokenizer/artifacts/polycodegpt_tokenizer_v1_1.json` | 1,095,741 B | `0459f8556a6f79ca1ba1c3dc3f76c26ae2e55653ebbf97beaa9bd23467358a03` |
| `tokenizer/artifacts/tokenizer_metadata_v1_1.json` | 41,093 B | `71dd044cb775accf3863890307a4e4700047f137c1534e7fb81751861cabef99` |
| `tokenizer/artifacts/training_input_manifest_v1_1.csv` | 150,586 B | `9322655c4ecc65fb4ca3e95664eb7572baa39230170586084fbab98a5f2bfcc7` |

Byte-identical unversioned copies (`polycodegpt_tokenizer.json`, `tokenizer_metadata.json`,
`training_input_manifest.csv`) point downstream consumers at the current artifact.

Metadata records: actual/requested vocab, base entries, merge count, unk configuration,
special token IDs, per-input counts and hashes, serializer version + SHA-256, training
duration, peak RAM, library versions (`tokenizers 0.22.2`, Python 3.11.9), training date, and
the determinism limitation note.

## 10. Test results

**53 passed, 0 failed** (0.48 s): 13 tokenizer tests + 40 serializer tests.

| # | Required area | Where | Result |
|---|---|---|---|
| 1 | Python serialization | `test_python_serialization_exact` | PASS |
| 2 | JavaScript serialization | `test_javascript_serialization_exact` | PASS |
| 3 | Documentation serialization | `test_documentation_serialization_exact`, `test_inventory_document_is_serialized_verbatim` | PASS |
| 4 | Explicit marker ordering | `test_marker_ordering_is_fixed`, `test_every_code_marker_present_once` | PASS |
| 5 | Empty docstrings | `test_empty_docstring_keeps_both_markers` | PASS |
| 6 | Whitespace preservation | tokenizer E, `test_long_input_not_truncated` | PASS |
| 7 | Python round-trip | tokenizer C, `test_parse_serialized_recovers_code_records` | PASS |
| 8 | JavaScript round-trip | tokenizer D | PASS |
| 9 | French accents | tokenizer H, `test_unicode_payload_roundtrips` | PASS |
| 10 | Arabic UTF-8 | tokenizer I, `test_unicode_payload_roundtrips` | PASS |
| 11 | Emoji / Unicode | tokenizer J, `test_unicode_payload_roundtrips` | PASS |
| 12 | Literal special-token escaping | `test_literal_control_token_is_escaped_in_stream`, `test_escaped_content_does_not_leak_control_token_ids` | PASS |
| 13 | Backslash escaping | `test_backslash_and_marker_escaping_is_reversible` (14 cases), `test_escaping_is_injective` | PASS |
| 14 | Long input without truncation | `test_long_input_not_truncated`, tokenizer M | PASS |
| 15 | Save/reload consistency | `test_save_reload_consistency` | PASS |
| 16 | Unique special token IDs | `test_special_token_ids_unique` | PASS |
| 17 | Unknown token configuration | `test_unknown_token_configuration` | PASS |

Stream-level invariant: escaped content yields exactly one `<|docstring|>`, one `<|code|>` and
one `<|eos|>` ID, and zero `<|pad|>`/`<|lang:fr|>` IDs even when the code literally contains
those strings.

Sample check across 5,000 training records: 0 structural anomalies, 0 code/docstring recovery
failures, 0 `unk`, 3.987 chars/token.

## 11. Data integrity results

| Check | Result |
|---|---|
| `python.zip` unchanged (940,909,997 B, `7223c646…`) | ✅ |
| `javascript.zip` unchanged (1,664,713,350 B, `fdc743f5…`) | ✅ |
| Filtered Python JSONL (train/valid/test) | ✅ all hashes match their manifest |
| Filtered JavaScript JSONL (train/valid/test) | ✅ all hashes match their manifest |
| Combined corpus train/valid/test | ✅ all hashes match `polycodegpt_corpus_manifest.csv` |
| 500 documentation files | ✅ 500/500 SHA-256 match; `Doc/license.rst` absent |
| Validation/test used for training | ✅ never — guarded; reads were SHA-256 stream checks only |
| GPU operational | ✅ torch 2.15.0+cu132, CUDA 13.2, RTX 5060 Laptop, matmul OK |

## 12. Remaining warnings

1. **Backslashes are doubled in the model stream** (243,801 additional characters, 0.05%).
   This is the cost of unambiguous escaping; decoding must call `unescape_content`.
2. **Five inert merges** still rebuild special-token strings; harmless (added tokens match
   first) but they explain why 15,738 merge rules produce only 15,733 new entries.
3. **Nine marker-fragment tokens** (`<|`, `<|code`, `<|eos`, …) remain in the vocabulary
   because the serializer's control strings appear in the training text; ≤12 of 15,738 merges
   involve them (0.08%).
4. **Escaping is a serializer-level guarantee**, not a tokenizer-level one: any consumer that
   builds streams without `serialize_*` forfeits the protection.
5. **No RNG seed in `tokenizers.BpeTrainer`**; determinism is bounded by input order and
   library version (seed 1234 is metadata only).
6. **Artifacts are Git-ignored** (`tokenizer/artifacts/*`), including the 150 KB manifest and
   the backup directory; hashes are preserved in this report.
7. **Inherited provenance limitation**: CodeSearchNet inputs are Zenodo v1
   (`10.5281/zenodo.7908468`); S3 `v2` parity unverified.

## 13. Final verdict

`TOKENIZER FINALIZED WITH WARNINGS`

Serializer 1.1.0 produces the exact required structures, escapes control-token literals
reversibly at both text and token level, and round-trips content exactly. The tokenizer was
retrained on the original training inputs only (392,632 records + 500 documents), reaches
16,000 vocabulary with the documented accounting, and passes all 53 tests. Warnings are the
documented escaping expansion, five inert merges, nine marker-fragment slots, seed/determinism
documentation, ignored artifacts, and inherited Zenodo provenance.
