# CodeSearchNet JavaScript — Collection & License Filtering Report (Phase 2D)

> **Scope.** Download and verification of `javascript.zip`, streaming extraction of its
> seven JSONL.GZ shards, strict repository-level license filtering, provenance annotation.
> No tokenizer, no model, no deduplication, no extraction of the dedup pickle.
>
> **Verdict:** `JAVASCRIPT DATASET VERIFIED WITH WARNINGS` — every count and hash checked,
> independent recount matched; the single warning is the unverified S3 `v2` parity.

---

## 1. Provenance

| Item | Value |
|---|---|
| Dataset | CodeSearchNet JavaScript |
| Provider | **Zenodo** (not S3) |
| DOI | `10.5281/zenodo.7908468` |
| Record | https://zenodo.org/records/7908468 |
| Record version | v1 (the deposit exposes no separate `version` field in its API; it is the sole published version) |
| Archive filename | `javascript.zip` |
| Source URL | `https://zenodo.org/records/7908468/files/javascript.zip?download=1` |
| Local path | `data/raw/codesearchnet_zenodo_javascript/javascript.zip` |
| Retrieval date | 2026-11-15 |

⚠️ **Provenance limitation.** The original CodeSearchNet S3 URLs return `403 AccessDenied`;
this is the Zenodo archival deposit. **Byte-level parity with the S3 `v2` archive has NOT
been established** and must not be claimed.

## 2. Download and checksums

Pre-download checks: 497 GB free disk; record metadata retrieved from
`https://zenodo.org/api/records/7908468` (filename `javascript.zip`, size 1,664,713,350,
published `md5:7649178a02f7b5c8fbccb64abe7946b6`); embedded licence metadata compared
against the Phase 2B.2 audit copy (see §5). Download used HTTPS with `curl -L --fail -C -`
(resumable).

| Check | Value | Expected / published |
|---|---|---|
| Size | 1,664,713,350 B | 1,664,713,350 B ✅ |
| MD5 | `7649178a02f7b5c8fbccb64abe7946b6` | `7649178a02f7b5c8fbccb64abe7946b6` ✅ **trusted source checksum** |
| SHA-256 | `fdc743f5af27f90c77584a2d29e2b7f8cecdd00c37b433c385b888ee062936dd` | computed (Zenodo publishes MD5 only) |
| ZIP CRC check (`testzip()`) | `None` — all member CRCs valid | ✅ |
| Members | 15 (7 directories, 7 shards, 2 pickles) | ✅ |

Unlike the Python archive, a **trusted source checksum was available** and matched.

## 3. Archive structure

| Member | Compressed (B) | Uncompressed (B) | Type |
|---|---|---|---|
| `javascript/final/jsonl/train/*.jsonl.gz` (5 shards) | 72,360,553 | 72,682,474 | code |
| `javascript/final/jsonl/valid/javascript_valid_0.jsonl.gz` | 4,551,704 | 4,570,551 | code |
| `javascript/final/jsonl/test/javascript_test_0.jsonl.gz` | 3,659,283 | 3,676,434 | code |
| **code total (7 shards)** | **80,571,540** | **80,929,459** | code |
| `javascript_licenses.pkl` | 32,315,567 | 196,114,558 | licence metadata |
| `javascript_dedupe_definitions_v2.pkl` | 1,551,823,021 | 6,597,724,320 | dedup metadata (untouched) |

Train shards are numbered 0–4 (5 shards). The JavaScript code payload is smaller than the
Python one (80.9 MB vs 219.7 MB uncompressed) even though the archive itself is larger
(1.66 GB vs 0.94 GB): most of the JavaScript archive is its 1.55 GB dedup pickle.

## 4. License filtering rules

Input: `data/manifests/codesearchnet_license_repo_summary.csv`
(`source_id, programming_language, repository, license_files, repository_category, families`;
67,233 rows, 50,174 of them JavaScript).

Accept a record **iff** its repository's conservative worst-case
`repository_category == "CLEAR_PERMISSIVE_CANDIDATE"`. Everything else is excluded:
`COPYLEFT_REVIEW_REQUIRED`, `PROPRIETARY_OR_RESTRICTIVE`, `AMBIGUOUS`,
`NO_LICENSE_FOUND`, unknown categories, and repositories absent from the summary. No new
classifications were invented.

## 5. License metadata consistency

| Property | ZIP member | Phase 2B.2 copy | Match |
|---|---|---|---|
| Uncompressed size | 196,114,558 | 196,114,558 | ✅ |
| CRC-32 | `0x145bda7a` | `0x145bda7a` | ✅ |
| SHA-256 | — | `7ae57111…d6181f1` | identical to audit copy |

The filter therefore used exactly the metadata that was audited in Phase 2B.2.

## 6. Summary per partition

| Partition | Raw | Accepted | Accepted % | Excluded | Unique repos (accepted) | Output size |
|---|---|---|---|---|---|---|
| train | 123,889 | 102,169 | 82.47% | 21,720 | 14,738 | 479,175,162 B |
| valid | 8,253 | 6,294 | 76.26% | 1,959 | 812 | 29,371,909 B |
| test | 6,483 | 5,699 | 87.91% | 784 | 816 | 26,377,624 B |
| **total** | **138,625** | **114,162** | **82.35%** | **24,463** | **16,366** | **534,924,695 B** |

## 7. Rejection breakdown

| Reason | Records | % of raw | Repositories |
|---|---|---|---|
| `COPYLEFT_REVIEW_REQUIRED` | 14,185 | 10.23% | — |
| `AMBIGUOUS` | 9,090 | 6.56% | — |
| `METADATA_MISSING` (absent from summary) | 1,074 | 0.77% | — |
| `PROPRIETARY_OR_RESTRICTIVE` | 114 | 0.08% | — |
| `NO_LICENSE_FOUND` | 0 | 0.00% | 0 |
| Unknown category | 0 | 0.00% | 0 |
| **Total** | **24,463** | **17.65%** | — |

`NO_LICENSE_FOUND` is zero for JavaScript because the single such repository contributes no
records. Per-partition detail lives in `_stats.json`.

## 8. Data quality statistics

| Check | Value |
|---|---|
| Invalid JSON rows | **0** |
| Empty `code` records | 0 |
| Empty `docstring` records | 0 |
| Missing `repo` / `path` / `url` / `sha` | 0 / 0 / 0 / 0 |
| Partition field / path mismatch | 0 |
| Languages encountered | `javascript` only (138,625 raw; 114,162 accepted) |
| Unique accepted repositories | 16,366 (of 47,848 permissive repos in the summary) |
| Unique accepted source paths | 30,288 |

| Partition | Code avg | Code median | Docstring avg | Docstring median |
|---|---|---|---|---|
| train | 928.8 | 458 | 198.7 | 115 |
| valid | 929.8 | 470 | 188.6 | 104 |
| test | 925.1 | 454 | 205.1 | 117 |
| **overall** | **928.7** | **458** | **198.5** | **115** |

No token counts are reported — the tokenizer does not exist yet.

## 9. Duplicate statistics

| Metric | Value |
|---|---|
| Distinct `code` SHA-256 values | 114,162 |
| Duplicated code values | **0** (0 extra occurrences) |
| `sha` field distinct values | 16,368 |
| `sha` repeat occurrences | 97,794 |

The dataset's `sha` field is the **repository commit hash**, not a per-function hash, so
repeated values are expected and are not duplicate code. Exact-duplicate code is zero; no
deduplication was performed or claimed.

## 10. Split integrity

train ∩ valid = 0 · train ∩ test = 0 · valid ∩ test = **0 repositories**. Original
partitions preserved; the test split is untouched and reserved for evaluation.

## 11. Output files

Directory: `data/processed/codesearchnet_javascript_filtered/`

| File | Records | Size | SHA-256 |
|---|---|---|---|
| `train.jsonl` | 102,169 | 479,175,162 B | `48f1b7ce7056c2c5…` |
| `valid.jsonl` | 6,294 | 29,371,909 B | `fc89a92f0ade1975…` |
| `test.jsonl` | 5,699 | 26,377,624 B | `0430974094e52171…` |
| `_stats.json` | — | — | run + verification stats |

Full hashes: `data/manifests/codesearchnet_javascript_filtered_manifest.csv`.

Outputs were written to `*.jsonl.tmp`, fully re-parsed and validated, and only then
`os.replace`d to their final names — no partially written final file can exist.

Every record keeps all original CodeSearchNet fields
(`repo, path, func_name, original_string, language, code, code_tokens, docstring,
docstring_tokens, sha, url, partition`) plus a `provenance` object with
`source_dataset, source_doi, source_archive_version, source_repository, source_path,
source_url, source_partition, license_category, license_filter_basis`.

## 12. Validation performed

Independent verifier: `scripts/verify_csn_filtered.py --language javascript`. It reads
every output line and re-derives the accepted set directly from the raw ZIP through a
separate code path (permissive membership + identity hash of `repo|path|func_name|url|code`).

| Partition | Output records | Raw-accepted recount | Matched | Foreign records |
|---|---|---|---|---|
| train | 102,169 | 102,169 | 102,169 | 0 |
| valid | 6,294 | 6,294 | 6,294 | 0 |
| test | 5,699 | 5,699 | 5,699 | 0 |

`checks_passed: true` — `data/manifests/codesearchnet_javascript_filtered_verification.json`.

Additional checks: 0 invalid JSON, 0 missing provenance fields, 0 missing record fields,
0 wrong licence categories, 0 repositories failing the permissive re-join, 0 partition
mismatches, 0 duplicate identities.

### Cross-dataset integrity

| Check | Result |
|---|---|
| `javascript.zip` MD5 / SHA-256 | ✅ `7649178a…46b6` / `fdc743f5…36dd` (unchanged) |
| `python.zip` MD5 / SHA-256 | ✅ unchanged (`07b49dd0…62e4f` / `7223c646…b6b40e`) |
| Python filtered outputs | ✅ all three SHA-256 unchanged (320,515 records intact) |
| Python docs collection | ✅ 569 files / 569 manifest rows |
| PyTorch / CUDA | ✅ 2.15.0+cu132, CUDA 13.2, RTX 5060 Laptop (12.0), GPU matmul verified |
| Git ignore | ✅ both raw archives and both processed dirs ignored |

## 13. Performance

| Metric | Value |
|---|---|
| Filtering wall time | 11.2 s for 138,625 records |
| Peak RSS | 68.4 MB of 16 GB |
| GPU memory used | 0 MB (CPU-only, as required) |
| Verification time | 7.5 s |

## 14. Limitations and warnings

1. **S3 v2 parity unverified.** Zenodo v1 only; MD5 matched the Zenodo-published value,
   which says nothing about the S3 object.
2. **Licence filtering remains a heuristic classification** (Phase 2B.2), not a legal
   review; "permissive" means a recognised permissive family with no copyleft or
   restrictive language detected.
3. **Vendored subdirectories** can still carry third-party terms even when the repo root is
   permissive; only repo-root metadata exists in the pickle.
4. **1,074 records from unclassified repositories were dropped**, not resolved.
5. **No cleaning, length filtering or deduplication** beyond the licence filter;
   `original_string` retains docstrings and indentation.
6. JavaScript's docstrings are shorter (median 115 chars vs 151 for Python) and its code
   median is 458 vs 667, so the two corpora differ in profile — relevant when weighting
   them for tokenizer/model training.
7. Output is 535 MB of JSONL, plus 1.66 GB raw archive; both must stay out of Git.

## 15. Next step (requires approval)

1. Review the JavaScript filter result and the 114,162-record corpus.
2. Candidate next phase: tokenizer corpus preparation over the combined 434,677 accepted
   records (320,515 Python + 114,162 JavaScript), with validation/test text held out.
3. Tokenizer and model training remain unstarted.
