# CodeSearchNet Python — Extraction & License Filtering Report (Phase 2C)

> **Scope.** Streaming extraction of the 16 Python JSONL.GZ shards directly from the raw
> ZIP, strict repository-level license filtering, provenance annotation. No tokenizer, no
> model, no shuffling, no deduplication performed.
>
> **Verdict:** `DATASET VERIFIED` — 320,515 of 457,461 records accepted, all output
> records valid, no partition overlap, and an independent recount from the raw archive
> reproduces every accepted record. No regeneration was required (Phase 2C verification
> pass).

---

## 1. Source provenance

| Item | Value |
|---|---|
| Dataset | CodeSearchNet Python |
| Provider | **Zenodo** (not S3) |
| DOI | `10.5281/zenodo.7908468` |
| Record version | v1 |
| Archive | `data/raw/codesearchnet_zenodo_python/python.zip` |
| Archive size | 940,909,997 bytes |
| Archive MD5 | `07b49dd01fbac894fbdae22da6462e4f` (re-verified, unchanged) |
| Archive SHA-256 | `7223c6460bebfa85697b586da91e47bc5d64790a4d60bba5917106458ab6b40e` (re-verified, unchanged) |

⚠️ **Provenance limitation (applies to everything below).** The original CodeSearchNet S3
URLs return `403 AccessDenied`. This archive is the Zenodo archival deposit, record
version v1. **Byte-level parity with the S3 `v2` archive has NOT been established** and
cannot be established while the bucket is private. Do not describe this corpus as
"verified S3 v2".

License metadata used: `python_licenses.pkl` inside this same archive (byte-identical to
the ranged copy from Phase 2B.2 — size 134,172,919 B, CRC-32 `0x6980bb9f`), aggregated by
Phase 2B.2 into `data/manifests/codesearchnet_license_repo_summary.csv`.

## 2. License filtering rules

Input schema (verified before writing the filter):
`source_id, programming_language, repository, license_files, repository_category, families`
— 67,233 rows (17,059 Python + 50,174 JavaScript).

Rule: a record is accepted **iff** `repository_category == "CLEAR_PERMISSIVE_CANDIDATE"`
for that repository, using the existing conservative **repository-level worst-case**
classification. No ad hoc family guessing, no per-file override, no assumption that a
public repository is permissive.

| Category | Action |
|---|---|
| `CLEAR_PERMISSIVE_CANDIDATE` | **INCLUDE** |
| `COPYLEFT_REVIEW_REQUIRED` | exclude |
| `PROPRIETARY_OR_RESTRICTIVE` | exclude |
| `AMBIGUOUS` | exclude |
| `NO_LICENSE_FOUND` | exclude |
| Repository absent from summary (`METADATA_MISSING`) | exclude, counted separately |
| Unknown / empty category | exclude, counted separately |

## 3. Processing method

- `zipfile` + `gzip` + `json` streaming straight from the archive; the ZIP was **not**
  extracted, no temp files, no full-corpus load into RAM.
- One record in flight at a time; only aggregate counters and length accumulators kept.
- **Processing time:** 71.1 s for all 457,461 records.
- **Peak RSS:** 103 MB (of 16 GB).
- No GPU memory allocated; CPU-only operation.
- No packages installed.

## 4. Summary per partition

| Partition | Raw records | Accepted | Accepted % | Excluded | Unique repos (accepted) | Output size |
|---|---|---|---|---|---|---|
| train | 412,178 | 290,507 | 70.49% | 121,671 | 9,786 | 1,358,821,036 B |
| valid | 23,107 | 14,671 | 63.49% | 8,436 | 540 | 73,204,092 B |
| test | 22,176 | 15,337 | 69.16% | 6,839 | 536 | 75,575,892 B |
| **total** | **457,461** | **320,515** | **70.06%** | **137,946** | **10,862** | **1,507,601,020 B** |

## 5. Rejection breakdown

| Exclusion reason | Records | % of raw | Repositories |
|---|---|---|---|
| `COPYLEFT_REVIEW_REQUIRED` | 102,854 | 22.49% | 2,296 |
| `AMBIGUOUS` | 31,552 | 6.90% | 356 |
| `METADATA_MISSING` (repo not in summary) | 1,958 | 0.43% | 69 |
| `PROPRIETARY_OR_RESTRICTIVE` | 312 | 0.07% | 6 |
| `NO_LICENSE_FOUND` | 270 | 0.06% | 1 |
| Unknown / empty category | 0 | 0.00% | 0 |
| **Total excluded** | **137,946** | **30.15%** | — |

Largest missing-metadata contributors: `chrisspen/burlap` (271), `Hackerfleet/hfos` (234),
`treycucco/bidon` (191), `taskcluster/taskcluster-client.py` (171),
`biosignalsnotebooks/biosignalsnotebooks` (114). Excluded rather than guessed, as required.

## 6. License metadata coverage

- 17,059 Python repositories carry a license classification; 13,885 of them are
  `CLEAR_PERMISSIVE_CANDIDATE`.
- **10,862** permissive repositories actually contribute accepted examples.
- 3,023 permissive repositories have **no** examples in the Python shards (allowed: the
  pickle covers every repo where a license file was found, not every repo in the corpus).
- 69 corpus repositories have no entry in the pickle → 1,958 records excluded.
- Strict `METADATA_MISSING` (repositories present in the corpus but absent from the
  licence summary) = **69 repositories / 1,958 records**, all excluded. Zero unknown
  repositories reached the output, so no record was included without a positive
  `CLEAR_PERMISSIVE_CANDIDATE` classification.

## 7. Data quality statistics

| Check | Value |
|---|---|
| Invalid JSON rows | **0** |
| Empty `code` records | 0 |
| Empty `docstring` records | 0 |
| Missing `repo` | 0 |
| Missing `path` | 0 |
| Missing `url` | 0 |
| Missing `sha` | 0 |
| `partition` field vs archive path mismatch | 0 |
| Languages encountered | `python` only (457,461 raw / 320,515 accepted) |
| Unique accepted repositories | 10,862 |
| Unique accepted source paths | 66,144 |

`source_path` is the repository-relative file path, so identical strings such as
`setup.py` are shared by many repositories: per-partition path counts (59,568 / 3,053 /
3,560) sum to more than the global distinct count. Repository-level split overlap remains
zero.

Code / docstring character length (accepted records):

| Partition | Code avg | Code median | Code max | Docstring avg | Docstring median |
|---|---|---|---|---|---|
| train | 1,050.2 | 667 | 103,665 | 300.3 | 149 |
| valid | 1,146.7 | 681 | 70,934 | 371.3 | 174 |
| test | 1,132.2 | 711 | 31,499 | 352.1 | 171 |
| overall | 1,058.6 | 670 | 103,665 | 306.0 | 151 |

No token counts are reported — the tokenizer does not exist yet.

## 8. Duplicate statistics

| Metric | Value |
|---|---|
| Distinct `code` SHA-256 values | 320,515 |
| Duplicated code values | **0** (0 extra occurrences) |
| `sha` field distinct values | 10,862 (= one commit per repository) |
| `sha` repeat occurrences | 309,653 |

Two important distinctions:

1. **The dataset's `sha` field is the repository commit hash**, not a per-function hash —
   the sample record's `sha` is exactly the commit embedded in its `url`. Repeated `sha`
   values are therefore expected and are **not** duplicate code.
2. Measured on SHA-256 of the `code` text, there are **zero exact duplicates**, and that
   was independently confirmed on two raw shards (30,000/30,000 distinct). **No
   deduplication was performed or claimed** — the inputs were already exact-duplicate-free.

## 9. Split overlap verification

Repository sets per partition, intersected pairwise:

| Pair | Overlapping repositories |
|---|---|
| train ∩ valid | **0** |
| train ∩ test | **0** |
| valid ∩ test | **0** |

Partitions were preserved exactly as published; nothing was reshuffled. The test split
remains untouched for evaluation. Integrity holds — no stop condition triggered.

## 10. Output files

Directory: `data/processed/codesearchnet_python_filtered/`

| File | Records | Size | SHA-256 |
|---|---|---|---|
| `train.jsonl` | 290,507 | 1,358,821,036 B | `2c23d07f9fb4e2ab…` (full hash in manifest) |
| `valid.jsonl` | 14,671 | 73,204,092 B | `3bcb20af27eb5773…` |
| `test.jsonl` | 15,337 | 75,575,892 B | `94eaa05e672ae5a6…` |
| `_stats.json` | — | — | run statistics |

- UTF-8, one JSON object per line, streamed writes.
- All original CodeSearchNet fields preserved
  (`repo, path, func_name, original_string, language, code, code_tokens, docstring,
  docstring_tokens, sha, url, partition`) plus a `provenance` object.
- Manifest: `data/manifests/codesearchnet_python_filtered_manifest.csv`
  (now carries `verification_status`, `verified_at_utc`, `unique_source_paths`).
- Verification report: `data/manifests/codesearchnet_python_filtered_verification.json`.

Provenance object written on every record:

```json
"provenance": {
  "source_dataset": "CodeSearchNet Python",
  "source_doi": "10.5281/zenodo.7908468",
  "source_archive_version": "v1 (Zenodo record 7908468)",
  "source_repository": "ageitgey/face_recognition",
  "source_path": "examples/face_recognition_knn.py",
  "source_url": "https://github.com/.../examples/face_recognition_knn.py#L46-L108",
  "source_partition": "train",
  "license_category": "CLEAR_PERMISSIVE_CANDIDATE",
  "license_filter_basis": "repository-level worst-case license classification from Phase 2B.2 audit ..."
}
```

## 11. Validation performed (Phase 2C verification pass)

Independent verifier: `scripts/verify_csn_filtered.py --language python` (shared with the
JavaScript dataset). It re-reads every output
line and **re-derives the accepted set from the raw ZIP through a separate code path**
(repository → `CLEAR_PERMISSIVE_CANDIDATE` membership test, per-record identity hash of
`repo|path|func_name|url|code`), then compares the two sets both ways.

| Partition | Output records | Raw-accepted recount | Matched | Output records not derivable from raw |
|---|---|---|---|---|
| train | 290,507 | 290,507 | 290,507 | 0 |
| valid | 14,671 | 14,671 | 14,671 | 0 |
| test | 15,337 | 15,337 | 15,337 | 0 |

Counts match the previous report exactly (290,507 / 14,671 / 15,337 = 320,515). No file
needed regeneration; all four outputs were present and internally consistent.

Machine-readable result: `data/manifests/codesearchnet_python_filtered_verification.json`
(`checks_passed: true`).

### Integrity re-check

| Check | Result |
|---|---|
| Raw ZIP MD5 | ✅ `07b49dd01fbac894fbdae22da6462e4f` (unchanged) |
| Raw ZIP SHA-256 | ✅ `7223c646…b6b40e` (unchanged) |
| Python docs collection | ✅ 569 files / 569 manifest rows |
| PyTorch | ✅ 2.15.0+cu132, CUDA 13.2 available |
| GPU | ✅ RTX 5060 Laptop, capability (12, 0), matmul ran |
| `data/processed/**` ignored by Git | ✅ all four outputs |

## 12. Validation checks on the output files

| Check | Result |
|---|---|
| Raw ZIP SHA-256 unchanged | ✅ `7223c646…b6b40e` |
| Raw ZIP MD5 unchanged | ✅ `07b49dd0…62e4f` |
| Raw Python docs unchanged | ✅ 569 files, 569 manifest rows |
| All output JSONL parses | ✅ 320,515/320,515 records |
| Every record has all 9 provenance keys | ✅ |
| Every `license_category` == `CLEAR_PERMISSIVE_CANDIDATE` | ✅ |
| Every repository re-joins to `CLEAR_PERMISSIVE_CANDIDATE` in the summary | ✅ 0 violations |
| No ambiguous/unknown/copyleft repository included | ✅ |
| Partition integrity (no repo overlap) | ✅ |
| Manifest hashes match files on disk | ✅ |
| Three partitions reopen and stream-parse | ✅ |

## 13. Limitations and warnings

1. **Provenance is Zenodo v1, not S3 v2.** Parity with the original S3 archive is
   unverified; cite the DOI, not "CodeSearchNet v2".
2. **License filtering is repository-level heuristics** from Phase 2B.2, not a legal
   review. "Permissive" = recognised permissive family with no copyleft/restrictive
   language detected (81.4% of Python repos).
3. **Vendored subdirectories can still carry third-party terms** even when the repository
   root is permissive; per-file licence text was not re-checked (only repo-root metadata
   exists in the pickle).
4. **1,958 records from 69 unknown repositories were dropped**, not resolved.
5. **No deduplication, filtering by length, tokenization, or cleaning was applied** beyond
   the license filter. `original_string` still contains docstrings and indentation.
6. **Docstring quality is unassessed**; CodeSearchNet docstrings are user-written and may
   be junk or templated.
7. Output is ~1.40 GiB of JSONL; keep it out of Git (already covered by
   `data/processed/*`) and out of RAM when streaming later.

## 14. Next step (requires approval)

1. Review the filter criterion and the 320,515-record corpus.
2. Next phase candidate: tokenizer corpus preparation (train/valid/test **text-only**
   streams, explicit `BOS/EOS`, no validation/test leakage into tokenizer training).
3. JavaScript remains un-downloaded, and tokenizer/model training has not started.
