# Phase 18 — Dataset Pipeline and PolyCodeGPT v2 Corpus

**Report date:** 2026-11-19  
**Scope:** dataset engineering and corpus construction only. No training, fine-tuning, benchmark execution, or model evaluation was performed.  
**Authoritative roadmap:** `reports/phase17_data_strategy.md`  
**Measured baseline:** `reports/phase16_technical_diagnosis.md`

## 1. Result

The Phase 18 pipeline produced a repository-isolated, syntax-gated, license-gated, deduplicated 500M-token Scenario B corpus at:

`data/processed/polycodegpt_v2/scenario_b/`

The selected training split contains **499,999,912 tokens**, split by language as:

| Language | Target | Selected |
|---|---:|---:|
| Python | 250,000,000 | 249,999,981 |
| JavaScript | 225,000,000 | 224,999,986 |
| English documentation | 25,000,000 | 24,999,945 |
| **Total** | **500,000,000** | **499,999,912** |

The validation split contains **30,894,000 tokens** and **46,761 records** across **13,919 repositories**. It contains code only, because documentation is deliberately train-only. Train and validation have **zero repository overlap**.

The full eligible pool before the deterministic 500M cap is also preserved at `data/processed/polycodegpt_v2/train.jsonl` and `valid.jsonl`: **1,189,261 train records / 780,997,244 train tokens** and **46,761 validation records / 30,894,000 validation tokens**. The cap discarded eligible records above the target; it did not lower any quality or license gate.

## 2. Preservation

The following were preserved and were not overwritten or regenerated:

- v0.1.0 release archive and checkpoints
- original v1 corpus and old train/validation partition
- frozen v1 tokenizer
- historical manifests and reports
- old test partition

`data/manifests/v2/preservation_manifest.csv` records **17 frozen artifact hashes**. The test partition was not read. The consumed historical benchmark at `functional/independent_v3/` was not read or used. No test or benchmark material entered extraction, deduplication, tokenizer training, or tokenization.

## 3. Sources and license decisions

The machine-readable registry is `data/manifests/v2/source_registry_v2.csv`; acquisition hashes and versions are in `data/manifests/v2/source_preflight.json`.

### Accepted

| Source | Version / acquisition | License decision |
|---|---|---|
| `CSN-PY` | Zenodo v1 record 7908468; local Phase 2B.2 filtered seed; 2026-11-15 | `ACCEPT`; per-repository `CLEAR_PERMISSIVE_CANDIDATE` only, not blanket public-domain permission |
| `CSN-JS` | Zenodo v1 record 7908468; local Phase 2B.2 filtered seed; 2026-11-15 | `ACCEPT`; per-repository `CLEAR_PERMISSIVE_CANDIDATE` only |
| `PYDOCS` | CPython v3.11.9; 500 eligibility-audited English documentation files; 2026-11-14 | `ACCEPT_TRAIN_ONLY`; PSF-2.0 prose, code samples dual PSF-2.0/0BSD; `Doc/license.rst` excluded |
| `NODE-DOCS` | Node.js git tag v24.12.0 tarball; 2026-11-18 | `ACCEPT_TRAIN_ONLY`; repository LICENSE is MIT-style permission; 149 Markdown files |
| `CODEPARROT` | Hugging Face revision `b5661e6b17396364b2bcf8e68977b0d28e1ebd19`; dataset snapshot 2022-03-16; acquired 2026-11-18 | `ACCEPT_WITH_PER_FILE_LICENSE_GATE`; dataset-level tag is `other` and was **not** treated as permission |

CodeParrot was accepted only through explicit per-file SPDX values:

`MIT`, `Apache-2.0`, `BSD-2-Clause`, `BSD-3-Clause`, `ISC`, `CC0-1.0`, `Unlicense`.

Missing, unknown, other, copyleft, and unlisted values were rejected. Accepted CodeParrot records retain repository, path, and license provenance.

Accepted CodeParrot license counts before deduplication:

| License | Accepted records |
|---|---:|
| MIT | 557,202 |
| Apache-2.0 | 206,399 |
| BSD-3-Clause | 73,214 |
| BSD-2-Clause | 18,812 |
| Unlicense | 7,216 |
| ISC | 6,845 |
| CC0-1.0 | 5,610 |

### Excluded

| Source | Decision | Reason |
|---|---|---|
| The Stack v1/v2 | `EXCLUDE` | Gated access; permission not established |
| StarCoderData | `EXCLUDE` | Gated access; permission not established |
| MDN prose | `EXCLUDE` | Mixed prose/code-sample terms and per-snippet date ambiguity |
| Stack Overflow / Stack Exchange code | `EXCLUDE` | CC-BY-SA share-alike terms incompatible with intended use |
| APPS / MBPP / CodeContests | `EXCLUDE` | Benchmark-oriented or non-permissive terms; contamination risk |
| BugsJS / BugsInPy | `EXCLUDE` | Consumed benchmark material; bulk-training contamination risk |

No source was treated as usable merely because it was publicly downloadable. A missing license file was not interpreted as permission.

## 4. Pipeline and gates

The v2 package is under `src/polycodegpt/data_v2/`:

- `sources.py` — source adapters and per-file CodeParrot license gate
- `quality.py` — static suitability checks, fingerprints, MinHash
- `language.py` — Python AST and JavaScript tree-sitter validation, classification, docstring alignment, repository normalization
- `schema.py` — task-first v2 serialization, special tokens, record validation
- `dedup.py` — global exact, normalized, and near deduplication
- `split.py` — deterministic repository hash split; documentation repositories forced to train
- `pipeline.py` — extraction, deduplication, audit
- `tokenizer.py` — v2 tokenizer training and tokenization

Implemented gates:

1. Explicit source and license decisions per record.
2. CodeParrot per-file SPDX allowlist.
3. Python AST and JavaScript tree-sitter syntax validation.
4. File-size, line-count, generated-marker, minified, vendored-path, and decoding-replacement filters.
5. Documentation accepted only when it contains a code-fenced example; documentation without code examples was rejected.
6. Task-first schema with explicit language and task special tokens.
7. Docstring alignment where usable; alignment failure becomes a module record rather than a fabricated prompt.
8. Global exact, normalized, and near deduplication before splitting.
9. Deterministic repository-level train/validation assignment.
10. Documentation is train-only; any repository contributing documentation is forced to train so its code cannot leak into validation.
11. Final record validation, unique record IDs, partition checks, repository-overlap checks, category accounting, and file hashes.

Near deduplication used 16 MinHash permutations, 4 bands × 4 rows, a Jaccard threshold of 0.8, and 5-token shingles.

## 5. Counts

### CodeParrot source coverage

- Shards examined: **80 of 1,126**
- Rows examined: **8,176,720**
- In-scope JavaScript/Python/Markdown rows: **2,018,196**
- Accepted before deduplication: **875,298**
- Rejected: **1,142,898**

Rows in other languages were not in scope and were not counted as quality rejections. The 500M cap was reached with 80 shards; the remaining shards were not needed and were not downloaded.

### Raw, accepted, rejected, quarantined

| Source | Raw in-scope | Accepted | Extraction rejected |
|---|---:|---:|---:|
| CodeParrot | 2,018,196 | 875,298 | 1,142,898 |
| CSN-PY | 305,134 | 297,327 | 7,807 |
| CSN-JS | 108,463 | 92,259 | 16,204 |
| Node docs | 923 | 922 | 1 |
| Python docs | 3,405 | 3,405 | 0 |
| **Total** | **2,436,121** | **1,269,211** | **1,166,910** |

Deduplication then removed **33,189** records, leaving **1,236,022** eligible records. The deterministic Scenario B cap selected **763,321** train records. Validation is **46,761** records.

Quarantined records: **0** in a separate quarantine store. Every rejected record remains traceable in `data/interim/v2/rejected.jsonl` with source, repository, path, content kind, license field, reason, and detail. Rejected material was not deleted or relabeled as accepted.

### Extraction rejection reasons

| Reason | Count |
|---|---:|
| License not allowlisted | 451,216 |
| Documentation without code example | 334,779 |
| Vendored or generated path | 98,848 |
| Too long | 81,817 |
| Too few lines | 81,335 |
| Syntax invalid | 47,205 |
| Too short | 38,912 |
| Generated marker | 24,658 |
| Generated or minified suffix | 2,532 |
| Minified | 1,494 |
| Decoding replacement | 212 |

The largest license rejection groups were GPL-3.0 (190,517), GPL-2.0 (132,750), AGPL-3.0 (63,326), LGPL-3.0 (22,027), MPL-2.0 (20,973), LGPL-2.1 (14,170), EPL-1.0 (5,670), and Artistic-2.0 (1,783).

## 6. Deduplication

Global deduplication ran on all **1,269,211** extracted candidates before the split:

| Class | Removed |
|---|---:|
| Exact duplicate | 9,039 |
| Normalized duplicate | 6,787 |
| Near duplicate | 17,363 |
| **Total removed** | **33,189** |

Retained after deduplication: **1,236,022** records. Deduplication removed **2.61%** of extracted candidates. Rejections are recorded with the duplicate-of identifier.

## 7. Scenario B composition

### Train by source

| Source / language | Records | Tokens |
|---|---:|---:|
| CodeParrot JavaScript | 277,844 | 208,152,041 |
| CodeParrot Python | 168,256 | 168,010,185 |
| CodeParrot documentation | 37,465 | 24,423,811 |
| CSN-JS | 56,743 | 16,847,945 |
| CSN-PY | 222,062 | 81,989,796 |
| Node docs | 208 | 170,658 |
| Python docs | 743 | 405,476 |
| **Total** | **763,321** | **499,999,912** |

### Train by category

| Language / category | Records | Tokens |
|---|---:|---:|
| Python / function | 256,918 | 116,674,091 |
| Python / class | 72,006 | 82,221,865 |
| Python / tests | 33,030 | 38,153,040 |
| Python / module | 24,937 | 11,482,695 |
| Python / async | 3,427 | 1,468,290 |
| JavaScript / module | 143,754 | 78,002,064 |
| JavaScript / stateful | 47,934 | 46,984,229 |
| JavaScript / function | 57,855 | 32,452,068 |
| JavaScript / tests | 45,468 | 33,926,201 |
| JavaScript / async | 19,965 | 18,576,015 |
| JavaScript / class | 19,611 | 15,059,409 |
| English / documentation | 38,416 | 24,999,945 |

All code categories are present. Documentation is represented at the target 25M tokens. The category mix is not claimed to be benchmark-balanced; it is the measured composition of the accepted, gated pool.

## 8. Split separation and validation

- Train repositories: **239,194**
- Validation repositories: **13,919**
- Repository overlap: **0**
- Validation policy: deterministic repository hash, 5% of code repositories
- Documentation in validation: **0 records**
- Validation records: **46,761**
- Validation tokens: **30,894,000**

### Validation by language

| Language | Records | Tokens |
|---|---:|---:|
| JavaScript | 24,968 | 16,647,922 |
| Python | 21,793 | 14,246,078 |
| English documentation | 0 | 0 |
| **Total** | **46,761** | **30,894,000** |

### Validation by source

| Source | Records | Tokens |
|---|---:|---:|
| CodeParrot JavaScript | 20,132 | 15,106,900 |
| CodeParrot Python | 10,190 | 10,127,173 |
| CSN-JS | 4,836 | 1,541,022 |
| CSN-PY | 11,603 | 4,118,905 |

Validation is code-only by design. Its 30.9M tokens are additional to the 500M training target: total selected corpus size, including validation, is **530,893,912 tokens**.

The earlier pilot failure, `retrohacker/getos` appearing in both train and validation, was caused by CodeParrot code and Markdown documentation from one repository being assigned independently. The final split pre-scans documentation repositories and forces all of their records to train. The final audit found zero overlap.

## 9. Syntax and prompt/docstring alignment

### Extraction

- Syntax-invalid rejections: **47,205**
- Main Python failures: legacy `print` syntax, Python 2 exception syntax, invalid syntax, tabs/spaces, invalid numeric literals
- Main JavaScript failure: tree-sitter parse error

Syntax rejection is a hard gate; no record with invalid syntax entered the final corpus.

### Alignment

Full eligible pool:

| Alignment result | Records |
|---|---:|
| Pass | 371,785 |
| Failed, converted to module record | 15,026 |
| Not applicable | 849,211 |

Scenario B train:

| Alignment result | Records |
|---|---:|
| Pass | 268,417 |
| Failed, converted to module record | 10,388 |
| Not applicable | 484,516 |

Scenario B validation:

| Alignment result | Records |
|---|---:|
| Pass | 15,755 |
| Failed, converted to module record | 684 |
| Not applicable | 30,322 |

A failed docstring alignment was **not** turned into a fabricated prompt. It was retained as a module record without a docstring. “Not applicable” covers records where no docstring alignment was expected, including documentation records.

## 10. Tokenizer

The v2 tokenizer is a 24,000-vocabulary ByteLevel BPE model trained only on the v2 training split:

- Artifact: `tokenizer/artifacts/v2/polycodegpt_tokenizer_v2.json`
- Metadata: `tokenizer/artifacts/v2/tokenizer_metadata_v2.json`
- Training records: 1,189,261
- Vocabulary: 24,000
- Merges: 23,729
- Added special-token count: 27
- Fit time: 557.39 seconds
- Token IDs are within the uint16 range; `.bin` files use little-endian uint16 tokens as written by Python `array("H")`.

Special tokens include language tokens (`<|lang:python|>`, `<|lang:javascript|>`, `<|lang:en|>`), task tokens, and structure tokens such as `<|file|>`, `<|context|>`, `<|docstring|>`, `<|signature|>`, `<|prefix|>`, `<|suffix|>`, `<|buggy|>`, `<|tests|>`, `<|code|>`, `<|explanation|>`, and `<|text|>`.

Documentation is serialized with `<|lang:en|>` and its text task marker. Code records are serialized task-first. The v1 tokenizer was not overwritten.

## 11. Hashes

### Scenario B corpus

| File | Bytes (approx.) | SHA-256 |
|---|---:|---|
| `train.jsonl` | 2.7 GB | `140bf946d1c3637aa0399a01b3f004e98f9de5d0c1b2854df09ff4dfb1bb5ee5` |
| `train.bin` | 954 MB | `2706cff336b42bee8bfaf610cce9a143f025a4a468939aa647ff7e44e69efa03` |
| `valid.jsonl` | 168 MB | `5b98ce443f7f495fc28c2c99bbeca6054efb20f02473005bdcc30e8f4a361ca6` |
| `valid.bin` | 59 MB | `63e7e4218657734dc2f64e6ade530e6fac9156694b78caa052d60cbd04de4c80` |

### Full eligible pool

| File | SHA-256 |
|---|---|
| `train.jsonl` | `1a5e57b9178a6ccb3e30febf68a84e2a17e5065878b97889ead121f3dd2a792c` |
| `valid.jsonl` | `5b98ce443f7f495fc28c2c99bbeca6054efb20f02473005bdcc30e8f4a361ca6` |

### Supporting artifacts

| Artifact | SHA-256 |
|---|---|
| v2 tokenizer JSON | `0e7f466ee516fe34becc638a3451eba29bced5efa5317647e486ab735cc7f120` |
| `source_registry_v2.csv` | `3721a16da8be09a7d716c28bc6fe804052382b991631460537408147b7c2703a` |
| `source_preflight.json` | `7d3b77922c195dd34a2ca8dbbb71f4abfe85b6810ee799798a790add51d506b6` |
| `preservation_manifest.csv` | `7af510951cfb8fb3e0b8e8420ac36412114aadb23cf2ebdcd8c5603d18244fc2` |
| `verification_report.json` | `cacfd988d1b7e56c468cb0201b3cc77d56daa85a160ca07de2a09163346bb752` |

The scenario manifest is duplicated at `data/processed/polycodegpt_v2/scenario_b/manifest.json` and `data/manifests/v2/scenario_b_manifest.json`. The independent verification record is `data/manifests/v2/verification_report.json`.

## 12. Reproducibility and observed costs

Principal commands:

```text
python scripts/phase18_preflight.py
python scripts/phase18_build.py extract --shard-limit 20
python scripts/phase18_build.py extract --shard-offset 20 --shard-limit 20 --append --no-base
python scripts/phase18_build.py extract --shard-offset 40 --shard-limit 20 --append --no-base
python scripts/phase18_build.py extract --shard-offset 60 --shard-limit 20 --append --no-base
python scripts/phase18_build.py dedup
python scripts/phase18_build.py audit
python scripts/phase18_tokenize.py train --vocab-size 24000 --min-frequency 2
python scripts/phase18_tokenize.py tokenize
python scripts/phase18_scenario_b.py
python scripts/phase18_verify.py
```

The Scenario B selection is deterministic: within each language, records are sorted by ascending `sha256("polycodegpt-v2-scenario-b-500m" + "|" + record_id)` and whole records are added until the language cap would be exceeded. The selection can be reproduced from `data/processed/polycodegpt_v2/train.jsonl` and its `token_count` fields.

Observed costs:

| Stage | Observation |
|---|---|
| CodeParrot download | 80 shards, 23,118,451,109 bytes (23.1 GB) |
| Extraction | 5,050.71 seconds across four batches (about 84 minutes) |
| Deduplication and split | completed, but wall-clock was not separately instrumented |
| Tokenizer fit | 557.39 seconds |
| Tokenization and per-record counts | completed; not separately instrumented |
| Scenario B selection and binning | completed; not separately instrumented |

No speedup or resource-efficiency claim is made for stages that were not separately instrumented.

## 13. Deviations and limitations

1. **80 of 1,126 CodeParrot shards were examined.** The 500M target was reached without the full source. The full eligible pool is 780,997,244 train tokens; the deterministic cap discarded 281M eligible train tokens. This is a deliberate target cap, not a quality downgrade.
2. **CodeParrot’s dataset-level license is `other`.** Only per-file SPDX allowlisted values were accepted. The dataset-level tag was not treated as permission.
3. **Documentation is train-only.** Validation contains no documentation. This preserves the historical documentation policy and avoids documentation leakage.
4. **The 25M documentation cap is below the available documentation volume.** The full eligible pool contains 115,644,755 documentation tokens; Scenario B selects 24,999,945.
5. **No separate quarantine file exists.** Rejected records are retained in `data/interim/v2/rejected.jsonl` with reasons and provenance. The quarantined count is therefore zero, not unreported.
6. **No model training, fine-tuning, benchmark execution, or test-partition access occurred.** A complete dataset does not authorize training.
7. **The 500M cap is a training-target cap.** Validation adds 30,894,000 tokens, so the total selected corpus is 530,893,912 tokens. The 500M figure refers to the training split.
8. **No claim is made that the category mix is benchmark-optimal.** It is the measured composition after license, syntax, suitability, alignment, and deduplication gates.

## 14. Scenario B readiness

Scenario B is ready:

- Python training volume: 249,999,981 tokens (target 250M)
- JavaScript training volume: 224,999,986 tokens (target 225M)
- Documentation training volume: 24,999,945 tokens (target 25M)
- Repository-level train/validation overlap: 0
- Validation is non-empty and code-only: 30,894,000 tokens across 46,761 records
- All code categories are represented
- Syntax-invalid records were excluded
- Licensing was resolved per file for CodeParrot and per source for CSN, Node, and CPython docs
- Deduplication and rejected-record traceability are complete
- Tokenizer, tokenized bins, manifests, and hashes are final

Scenario A is not selected. The A fallback remains available if a future audit rejects the B corpus, but no such failure was found here.

A complete dataset does not authorize training.

PHASE 18 COMPLETE — SCENARIO B DATASET VERIFIED
