# PolyCodeGPT Combined Python + JavaScript Corpus — Phase 2E Report

**Date:** 2026-11-15
**Scope:** combine the two verified CodeSearchNet filtered datasets into one corpus, resolve
cross-language repository leakage globally, audit exact duplicates, analyse quality, inventory
the Python documentation corpus, and specify the tokenizer corpus.
**Not done in this phase:** tokenizer training, model training, any modification of raw or
filtered source datasets.

---

## 1. Inputs (unmodified)

| Input | Path | Records | SHA-256 verified |
|---|---|---|---|
| CodeSearchNet Python | `data/processed/codesearchnet_python_filtered/` | 320,515 | ✅ unchanged |
| CodeSearchNet JavaScript | `data/processed/codesearchnet_javascript_filtered/` | 114,162 | ✅ unchanged |

Both inputs were opened read-only. Their partition SHA-256 values were re-computed after the
build and match the Phase 2C/2D values (`2c23d07f…`, `3bcb20af…`, `94eaa05e…`; JS hashes as
recorded in `codesearchnet_javascript_filtered_manifest.csv`). The raw archives are also
unchanged: `python.zip` 940,909,997 B / `7223c646…`, `javascript.zip` 1,664,713,350 B /
`fdc743f5…`.

## 2. Global repository leakage audit

Repository identifier for the rule: `lowercase(owner/repository)` — the `repo` field
(equivalently `provenance.source_repository`) lower-cased and stripped. Original casing is
preserved in every output record.

Partitions each repository occupies, across **both** languages:

| Metric | Value |
|---|---|
| Distinct normalized repositories | **27,221** |
| Repositories present in more than one original partition | **1** |
| Repositories containing both Python and JavaScript | **7** |
| Repository-pair overlaps before resolution | train/valid 0, train/test 1, valid/test 0 |

Precedence applied: **TEST > VALID > TRAIN**. The single multi-partition repository:

| Repository | Original partitions | Assigned | Records moved |
|---|---|---|---|
| `maartenbreddels/ipyvolume` | train (44 Python records), test (JavaScript records) | **test** | 44 Python records train → test |

This is exactly the cross-language leak the audit was designed to catch: the repository's
Python half was in the training partition while its JavaScript half was in test.

After resolution: **0 repositories span any two partitions**, and train∩valid = train∩test =
valid∩test = 0.

## 3. Reassignment accounting

| Original → final | Records |
|---|---|
| train → train | 392,632 |
| train → **test** | **44** |
| valid → valid | 20,965 |
| test → test | 21,036 |

* Repositories reassigned: **1**
* Records reassigned: **44** (0.0101% of the corpus)
* Records discarded: **0**

## 4. Original versus final counts

| Partition | Original (PY + JS) | Final | Δ |
|---|---|---|---|
| train | 290,507 + 102,169 = 392,676 | 392,632 | −44 |
| valid | 14,671 + 6,294 = 20,965 | 20,965 | 0 |
| test | 15,337 + 5,699 = 21,036 | 21,080 | +44 |
| **total** | **434,677** | **434,677** | **0** |

## 5. Exact duplicate audit

`code_sha256` = SHA-256 over the UTF-8 bytes of the `code` field, computed for all 434,677
records before any other processing. The CodeSearchNet `sha` field (a repository commit hash)
is **not** used as a content hash — it is retained unchanged as metadata.

| Check | Result |
|---|---|
| Distinct code hashes | **434,677** (equal to record count) |
| Records sharing code with another record | **0** |
| Identical code across Python ↔ JavaScript | **0** |
| Identical code train ↔ valid | **0** |
| Identical code train ↔ test | **0** |
| Identical code valid ↔ test | **0** |
| Duplicate pairs identified | **none** |

No deduplication action was applied or is recommended: there is nothing to deduplicate.
(Note for future phases: this audit covers *exact* content only. Near-duplicates — whitespace,
renamed identifiers, reformatted code — are not detected and were not removed.)

## 6. Final partition statistics

| Partition | Records | Python | JavaScript | Share | Repositories | Unique (repo, path) |
|---|---|---|---|---|---|---|
| train | 392,632 | 290,463 | 102,169 | 90.33% | 24,517 | 95,421 |
| valid | 20,965 | 14,671 | 6,294 | 4.82% | 1,352 | 5,069 |
| test | 21,080 | 15,381 | 5,699 | 4.85% | 1,352 | 5,481 |
| **total** | **434,677** | **320,515** | **114,162** | 100% | **27,221** | **105,971** |

Language split overall: Python 73.74%, JavaScript 26.26%.

## 7. Data quality statistics

### Python

| Partition | Records | Repos | (repo,path) | Code chars | Doc chars | Code avg | Code med | Doc avg | Doc med |
|---|---|---|---|---|---|---|---|---|---|
| train | 290,463 | 9,785 | 60,110 | 305,030,950 | 87,211,889 | 1,050 | 667 | 300 | 149 |
| valid | 14,671 | 540 | 3,070 | 16,822,762 | 5,447,369 | 1,147 | 681 | 371 | 174 |
| test | 15,381 | 537 | 3,589 | 17,431,829 | 5,418,634 | 1,133 | 712 | 352 | 172 |

### JavaScript

| Partition | Records | Repos | (repo,path) | Code chars | Doc chars | Code avg | Code med | Doc avg | Doc med |
|---|---|---|---|---|---|---|---|---|---|
| train | 102,169 | 14,738 | 35,311 | 94,899,648 | 20,302,918 | 929 | 458 | 199 | 115 |
| valid | 6,294 | 812 | 1,999 | 5,852,183 | 1,187,223 | 930 | 470 | 189 | 104 |
| test | 5,699 | 816 | 1,892 | 5,272,211 | 1,169,144 | 925 | 454 | 205 | 117 |

### Combined

| Partition | Records | Repos | (repo,path) | Code chars | Doc chars | Code avg | Code med | Doc avg | Doc med |
|---|---|---|---|---|---|---|---|---|---|
| train | 392,632 | 24,517 | 95,421 | 399,930,598 | 107,514,807 | 1,018.6 | 612 | 273.8 | 137 |
| valid | 20,965 | 1,352 | 5,069 | 22,674,945 | 6,634,592 | 1,081.6 | 615 | 316.5 | 143 |
| test | 21,080 | 1,352 | 5,481 | 22,704,040 | 6,587,778 | 1,077.0 | 636 | 312.5 | 151 |
| **overall** | **434,677** | **27,221** | **105,971** | **445,309,583** | **120,737,177** | **1,024.5** | **614** | **277.8** | **138** |

Integrity counters (all partitions, both languages): empty code **0**, empty docstring **0**,
invalid JSON **0**, records missing provenance **0**, non-permissive licence category **0**.

Corpus size: 445.3 M code characters (425 MB) + 120.7 M docstring characters (115 MB).
No token counts are estimated anywhere — the vocabulary does not exist yet.

## 8. Output files and hashes

`data/processed/polycodegpt_corpus/` — 2,301,573,062 B total.

| File | Records | Size (B) | SHA-256 |
|---|---|---|---|
| `train.jsonl` | 392,632 | 2,071,736,482 | `2a9c71607a1893711cd7f2963ec172784415f838ec1c633c988949b45e345c74` |
| `valid.jsonl` | 20,965 | 115,075,352 | `2eb02406d22101716214a6509314e9d7e1596e1cc6640d0eb4a886c00fb715af` |
| `test.jsonl` | 21,080 | 114,761,228 | `93f624f6e1f24d8da8d3855fbf407b205671e1f5f1f29ab88914fff16b4f3c2e` |
| `_stats.json` | — | build + quality statistics | — |

Written as `*.jsonl.tmp`, fully re-parsed and re-validated (partition field, licence category,
provenance, counts), then atomically `os.replace`d. Manifest:
`data/manifests/polycodegpt_corpus_manifest.csv` (hash + size re-checked against disk).
Validation result: `data/manifests/polycodegpt_corpus_validation.json` (`checks_passed: true`).

### Record schema

All original CodeSearchNet fields are preserved untouched — `repo`, `path`, `func_name`,
`original_string`, `language`, `code`, `code_tokens`, `docstring`, `docstring_tokens`, `sha`,
`url`, `partition`, plus the nested `provenance` block (9 fields) from the filtering phase.

Added fields:

| Field | Meaning |
|---|---|
| `record_id` | SHA-256 of `source_dataset|repo|path|func_name|code_sha256` |
| `code_sha256` | SHA-256 of the exact `code` content |
| `original_partition` | partition in the source dataset (`train`/`valid`/`test`) |
| `global_partition` | partition after repository-level resolution |
| `source_dataset` | `CodeSearchNet Python` / `CodeSearchNet JavaScript` |
| `source_doi` | `10.5281/zenodo.7908468` |
| `source_archive_version` | `v1 (Zenodo record 7908468)` |
| `license_category` | `CLEAR_PERMISSIVE_CANDIDATE` (all 434,677 records) |
| `license_filter_basis` | the Phase 2B.2 classification basis |

`provenance` additionally carries `source_repository`, `source_path`, `source_url`,
`source_commit`, `source_partition`, `license_category`, `license_filter_basis`,
`source_dataset`, `source_doi`. No original information was dropped, no content generated,
and no examples concatenated.

## 9. Python documentation inventory

* File: `data/manifests/python_docs_training_corpus_inventory.csv` — **500 rows**, 276,379 B.
* Selection: every retrieved `Doc/**/*.rst` (501 files) **except `Doc/license.rst`**
  (`PENDING_MANUAL_REVIEW`) → 500 `ELIGIBLE_CANDIDATE` files.
* Contents: `doc_id`, `source_id` (PYDOCS), version (3.11.9), original URL, local path,
  licence (`PSF-2.0` prose; samples dual PSF-2.0/0BSD), `file_sha256`, size, character count,
  language (en), `corpus_role = train_only_documentation`,
  `partition_assignment = train_only`.
* Total: **11,096,475 characters** of English documentation.
* Raw documentation untouched: still 569 files under `data/raw/python_docs_en/cpython-v3.11.9/`.
* These documents are **never** placed in validation or test.

## 10. Tokenizer corpus specification

`docs/TOKENIZER_CORPUS_SPEC.md` (11 sections):

* **Fields tokenized:** `code` and `docstring` only. Metadata (`repo`, `path`, `url`, `sha`,
  `code_tokens`, `docstring_tokens`) is never serialized.
* **Stream format:** `<|bos|><|lang:python|><|code|>{code}<|doc|>{docstring}<|eos|>`;
  the `<|doc|>` segment is dropped when the docstring is already verbatim inside `code`
  (verified: 100% of a 20,000-record sample).
* **Language markers:** `<|lang:python|>`, `<|lang:javascript|>`, `<|lang:en|>`, with
  `<|lang:fr|>` and `<|lang:ar|>` reserved for the future French/Arabic data.
* **Special tokens (11):** `<|pad|>`, `<|unk|>`, `<|bos|>`, `<|eos|>`, `<|code|>`, `<|doc|>`,
  five language markers.
* **Vocabulary:** 16,000 tokens (11 special + 15,989 learned).
* **Algorithm:** byte-level BPE over UTF-8 bytes → guaranteed byte fallback for any valid
  UTF-8; no Unicode normalization (code must round-trip byte-exactly); no lowercasing;
  deterministic (seed 1234).
* **Tokenizer training data:** the final **training partition only** (392,632 records) plus the
  500 train-only documentation files. `valid.jsonl` and `test.jsonl` are never read during
  fitting; the next phase must ship a manifest of input file hashes proving this.
* No tokenizer was trained in this phase.

## 11. Validation performed

| Check | Result |
|---|---|
| Combined records = sum of retained language records (320,515 + 114,162) | ✅ 434,677 |
| Every record parses as JSON, all required fields present | ✅ 0 missing |
| Every record has complete provenance | ✅ 0 incomplete |
| Every repository assigned exactly one global partition | ✅ 27,221 / 27,221 |
| No repository overlaps across final partitions | ✅ 0 / 0 / 0 |
| `original_partition` preserved and consistent | ✅ all records |
| All licences `CLEAR_PERMISSIVE_CANDIDATE` | ✅ 434,677 / 434,677 |
| Raw archives unchanged (size + SHA-256) | ✅ both |
| Filtered source datasets unchanged (hash + counts) | ✅ all 6 files |
| Output manifest hashes and sizes match disk | ✅ 3 / 3 |
| Python docs untouched (569 files) | ✅ |
| Processed output directories Git-ignored | ✅ `data/processed/**`, `data/raw/**` ignored |
| PyTorch / CUDA operational | ✅ torch 2.15.0+cu132, CUDA 13.2, RTX 5060 Laptop, GPU matmul verified |

Performance: build 53.1 s, peak RSS 391.1 MB of 16 GB, verification 17.0 s, 0 MB GPU.

## 12. Remaining warnings

1. **Zenodo v1 provenance only** — the archive is the Zenodo deposit (DOI
   `10.5281/zenodo.7908468`); S3 `v2` parity remains unverified (S3 returns `403`). This
   applies unchanged to both languages.
2. **Licence filtering is a heuristic metadata classification**, not a legal review; vendored
   subdirectories may carry third-party terms under a permissive repository root.
3. **One reassignment changed the test set**: 44 Python records of
   `maartenbreddels/ipyvolume` moved train → test. Any earlier Python-only metric computed on
   the pre-combination test set (15,337 records) is superseded by 15,381.
4. **Exact-duplicate audit only** — 0 exact duplicates found, but near-duplicates
   (reformatted, renamed, whitespace variants) were not audited and may exist.
5. **Repository normalization for the partition rule only** — lower-cased `owner/repo`;
   original casing preserved in records. Two repos differing only in case would be merged by
   the rule (none observed).
6. **No cleaning beyond licence filtering**: no length filtering, no truncation, no
   near-dedup, no comment stripping, no docstring paraphrase.
7. **Documentation corpus is raw RST** with markup/directives; not normalized, and licence
   notices must be retained in any derived use.
8. **No token counts** are reported or estimated anywhere.
9. **Storage**: 2.30 GB corpus + 2.61 GB archives + 2.04 GB filtered subsets stay out of Git
   (verified ignored); only manifests, scripts, docs and reports are trackable.

---

**Verdict: `COMBINED CORPUS VERIFIED WITH WARNINGS`**

The corpus is complete (434,677 records, nothing discarded), licence-clean, provenance-complete,
repository-disjoint across partitions, and duplicate-free at exact-content level. Warnings are
provenance (Zenodo v1, S3 v2 unverified), heuristic licence classification, and the single
documented recurrence of a cross-language repository that forced 44 records into test.
