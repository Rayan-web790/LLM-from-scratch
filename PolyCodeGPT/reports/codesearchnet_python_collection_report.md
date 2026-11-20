# CodeSearchNet Python — Controlled Collection Report (Phase 2B.3)

> **Scope.** Collection of the raw Python archive only. No extraction of code shards, no
> filtering, no preprocessing, no tokenizer or model work. Archive treated as immutable
> raw data.

**Verdict: `DOWNLOAD SUCCESSFUL`** — archive downloaded, size and MD5 verified, ZIP
readable, license metadata byte-identical to the ranged copy from Phase 2B.2.

---

## 1. Source details

| Item | Value |
|---|---|
| Dataset | CodeSearchNet Python |
| Provider | Zenodo |
| DOI | `10.5281/zenodo.7908468` |
| Record | https://zenodo.org/records/7908468 |
| Record version | v1 |
| Archive filename | `python.zip` |
| Expected size | 940,909,997 bytes |
| Expected MD5 | `07b49dd01fbac894fbdae22da6462e4f` |
| Local path | `data/raw/codesearchnet_zenodo_python/python.zip` |
| Retrieval date | 2026-11-15 |

Original project: `github/CodeSearchNet` (archived, last push 2022-01-31). The Zenodo
deposit is the only publicly readable copy of the archive.

## 2. Provenance limitation

The original CodeSearchNet S3 URLs (`https://s3.amazonaws.com/code-search-net/…`)
return `403 AccessDenied` (verified in Phase 2B.2). The Zenodo deposit is a
**separately identified archival source** (record version v1). Byte-level parity with
the S3 `v2` archive has **NOT** been established and cannot be established while the
bucket is private. This archive must **not** be described as "verified S3 v2". Its
provenance is the Zenodo DOI and record version only.

## 3. Download method

- Transport: HTTPS (`https://zenodo.org/records/7908468/files/python.zip?download=1`).
- Tool: `curl -L --fail -C -` (follows redirects, resumable, fails on HTTP errors).
- Pre-download checks: archive absent; remote `Content-Length` 940,909,997; remote
  `Content-Disposition: attachment; filename=python.zip` (filename matches).
- Disk space available: ~497 GB (archive is 0.88 GB).
- `javascript.zip` and all other language archives were **not** downloaded.

## 4. Archive verification

| Check | Result |
|---|---|
| File exists | yes |
| Size | 940,909,997 bytes = expected ✅ |
| MD5 | `07b49dd01fbac894fbdae22da6462e4f` = expected ✅ |
| SHA-256 | `7223c6460bebfa85697b586da91e47bc5d64790a4d60bba5917106458ab6b40e` (recorded) |
| Readable ZIP | `zipfile.testzip()` → `None` ✅ |
| Central directory | readable, 24 members ✅ |

No discrepancy — preprocessing was **not** started.

## 5. ZIP member inventory

24 members: 6 directories, 14 train + 1 valid + 1 test JSONL.GZ shard, 1 licence
metadata pickle, 1 deduplication pickle.

| Partition | Shards | Compressed (bytes) | Uncompressed (bytes) |
|---|---|---|---|
| train | 14 | 196,430,258 | 197,450,754 |
| valid | 1 | 11,431,138 | 11,498,917 |
| test | 1 | 10,723,341 | 10,780,631 |
| **code total** | **16** | **218,584,737** | **219,730,302** |

| Member | Compressed (bytes) | Uncompressed (bytes) | Type |
|---|---|---|---|
| `python_licenses.pkl` | 32,666,588 | 134,172,919 | licence metadata |
| `python_dedupe_definitions_v2.pkl` | 689,653,710 | 2,903,767,061 | deduplication metadata (not extracted) |

The structure matches the expected CodeSearchNet layout:
`python/final/jsonl/{train,valid,test}/*.jsonl.gz`.

## 6. License metadata consistency

The `python_licenses.pkl` member of the downloaded archive was compared with the
ranged copy retrieved in Phase 2B.2 (`data/interim/codesearchnet_license_metadata/python_licenses.pkl`):

| Property | ZIP member | Local ranged copy | Match |
|---|---|---|---|
| Uncompressed size | 134,172,919 | 134,172,919 | ✅ |
| CRC-32 | `0x6980bb9f` | `0x6980bb9f` | ✅ |

The previously built inventory (`data/manifests/codesearchnet_license_inventory.csv`,
`data/manifests/codesearchnet_license_repo_summary.csv`) is therefore consistent with
the downloaded archive. No discrepancy — no investigation required.

## 7. Git ignore verification

`PolyCodeGPT/.gitignore` updated to exclude:

- `data/interim/codesearchnet_license_metadata/`
- `data/manifests/codesearchnet_license_inventory.csv`
- `data/manifests/codesearchnet_license_repo_summary.csv`
- `data/raw/codesearchnet_zenodo_python/`

`git ls-files` found none of these already tracked, so no index removal was needed.
The root workspace `.gitignore` already ignores `.venv/` and `PolyCodeGPT/data/raw/*`,
so `.venv` and the raw datasets remain ignored. `git check-ignore` confirmed all five
paths resolve to an ignore rule. No commit or push was performed.

## 8. Files created or modified

- `data/raw/codesearchnet_zenodo_python/python.zip` (940,909,997 B) — raw, unmodified.
- `data/manifests/codesearchnet_python_source_metadata.json` — source metadata entry.
- `data/manifests/source_registry.csv` — appended `CSN-PY-ZENODO` row.
- `reports/codesearchnet_python_collection_report.md` — this report.
- `PolyCodeGPT/.gitignore` — four new exclusion lines.

## 9. Warnings

1. **Provenance is Zenodo, not S3 v2.** S3 `v2` parity unverified; do not describe as
   verified S3 v2.
2. **Dedup pickle untouched.** `python_dedupe_definitions_v2.pkl` is 2.90 GB
   uncompressed and was not extracted or inspected.
3. **No extraction performed.** Code shards remain inside the archive; nothing has been
   filtered, preprocessed, tokenized, or trained.
4. **Archive is raw data.** It must not be modified, deleted, or replaced.

## 10. Remaining issues

- `METADATA_MISSING` (repos with no licence file) still unknown; requires extraction
  and matching against `_licenses.pkl` keys.
- 383 Python + 443 JavaScript `AMBIGUOUS` repositories and 2,781 + 1,853 copyleft
  repositories await review before any example is admitted.
- Awaiting your review before proceeding to extraction or filtering.
