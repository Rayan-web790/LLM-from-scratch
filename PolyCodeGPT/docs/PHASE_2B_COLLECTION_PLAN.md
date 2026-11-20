# PolyCodeGPT — Phase 2B Collection Plan

> **Status:** Planning document only. No data has been downloaded or collected yet.
> **Prerequisite:** Phase 2A.1 approval gate completed.

## 1. Collection Priority Order

| Priority | Source | Status | Notes |
|----------|--------|--------|-------|
| 1 | Python Official Documentation (English) | APPROVED_CANDIDATE | Only approved source. Start here. |
| 2 | CodeSearchNet Python | PENDING_REVIEW | Requires license filtering first |
| 3 | CodeSearchNet JavaScript | PENDING_REVIEW | Requires license filtering first |
| 4 | Other sources | PENDING_REVIEW / UNVERIFIED | Only when exact reuse terms verified |

French and Arabic sources remain separate work items until their licenses are verified.

---

## 2. Source: Python Official Documentation (English)

### Retrieval Method

- Download reStructuredText source from the Python documentation repository (https://github.com/python/cpython, `Doc/` directory).
- Alternatively, scrape rendered HTML from https://docs.python.org/3/ using a respectful crawler.
- Record the exact version (e.g., Python 3.11.x) and retrieval date.

### Expected Format

- Input: reStructuredText (.rst) files or HTML pages.
- Output: JSONL with one JSON object per document (section or page).

### Required Metadata

```json
{
  "doc_id": "pydocs-en-<section-id>",
  "source_id": "PYDOCS",
  "source_url": "https://docs.python.org/3/<path>",
  "license": "PSF-2.0",
  "code_license": "PSF-2.0 + 0BSD (dual)",
  "programming_language": "python",
  "natural_language": "en",
  "content_type": "documentation",
  "version": "3.11.x",
  "retrieval_date": "YYYY-MM-DD",
  "attribution": "Copyright (c) 2001 Python Software Foundation; All Rights Reserved"
}
```

### License Requirements

- Preserve PSF copyright notice in all derived documents.
- Retain full license text reference.
- Code examples may be used under 0BSD (no attribution needed) or PSF-2.0 (attribution needed). Use 0BSD terms where possible.
- Exclude content identified as third-party material with different license terms unless separately reviewed.

### Provenance Requirements

- Source URL for every document.
- Version number.
- Retrieval date.
- License identifier.

### Estimated Storage

- ~56 MB (source reStructuredText).
- Processed JSONL: ~100–200 MB (with metadata).

### Validation Steps

1. Verify all documents are valid UTF-8.
2. Verify PSF copyright notice is present in metadata.
3. Verify no third-party licensed content is included without separate review.
4. Verify document count and token count match estimates.
5. Verify language tag is "en" for all documents.

---

## 3. Source: CodeSearchNet (Python + JavaScript)

### Retrieval Method

- Download from Zenodo mirror (https://zenodo.org/records/7857872 or https://zenodo.org/records/7908468).
- Extract only Python and JavaScript zip files.
- Inspect `_licenses.pkl` files to build a per-repository license map.
- Filter examples to include only repositories with explicitly permissive licenses.

### Expected Format

- Input: Compressed zip files containing JSONL and `.pkl` files.
- Output: JSONL with one JSON object per function example.

### Required Metadata

```json
{
  "doc_id": "csn-py-<repo>-<func_name>-<hash>",
  "source_id": "CSN-PY or CSN-JS",
  "source_url": "https://github.com/<repo>/blob/<sha>/<path>#L<start>-L<end>",
  "license": "<per-repository license>",
  "programming_language": "python or javascript",
  "natural_language": "en",
  "content_type": "code",
  "repo_name": "<owner/repo>",
  "file_path": "<path>",
  "func_name": "<function name>",
  "retrieval_date": "YYYY-MM-DD",
  "attribution": "<license-specific attribution if required>"
}
```

### License Requirements

- Inspect `_licenses.pkl` to determine license per repository.
- Include only repositories with permissive licenses (MIT, Apache-2.0, BSD-2/3-Clause, 0BSD, CC0, PSF-2.0, etc.).
- Exclude repositories with unknown, copyleft (GPL, AGPL, LGPL, MPL, EPL), or restrictive licenses.
- Do not assume the GitHub repository's MIT license applies to all code examples.
- Record the specific license identifier per example.

### Provenance Requirements

- Repository name (`owner/repo`).
- File path within repository.
- Function name.
- GitHub URL with line numbers.
- License identifier (from `_licenses.pkl`).

### Estimated Storage

- Python: ~940 MB compressed, ~2–3 GB uncompressed.
- JavaScript: ~1.7 GB compressed, ~4–5 GB uncompressed.
- Processed JSONL (filtered): ~1–2 GB per language.

### Validation Steps

1. Verify `_licenses.pkl` files are present and readable.
2. Verify license distribution across repositories.
3. Verify filtered dataset contains only permissive licenses.
4. Verify no repository appears in multiple splits.
5. Verify deduplication removes near-duplicate functions.
6. Verify document count and token count match estimates.

---

## 4. Source: The Stack (Python + JavaScript Subset)

### Retrieval Method

- Accept Terms of Use on Hugging Face.
- Use `datasets` library with `streaming=True` to stream metadata.
- Filter by programming language (Python, JavaScript).
- Download file contents from Software Heritage S3 bucket (requires AWS credentials and Software Heritage/INRIA agreement).
- Apply per-repository license filter.

### Expected Format

- Input: Hugging Face dataset (metadata) + S3 file contents.
- Output: JSONL with one JSON object per file.

### Required Metadata

```json
{
  "doc_id": "stack-<language>-<repo>-<file_hash>",
  "source_id": "STACK-V1 or STACK-V2",
  "source_url": "https://github.com/<repo>/blob/<commit>/<path>",
  "license": "<per-repository license>",
  "programming_language": "python or javascript",
  "natural_language": "en",
  "content_type": "code",
  "repo_name": "<owner/repo>",
  "file_path": "<path>",
  "swh_id": "<Software Heritage ID>",
  "retrieval_date": "YYYY-MM-DD",
  "attribution": "<license-specific attribution if required>"
}
```

### License Requirements

- Apply per-repository license filter (only permissive licenses).
- Retain provenance for license attribution.
- Exclude repositories with unknown or restrictive licenses.

### Provenance Requirements

- Repository name.
- File path.
- Software Heritage ID.
- License identifier.
- Source URL.

### Estimated Storage

- Metadata: ~100 MB (streaming, no full download).
- File contents (Python + JS subset): ~5–10 GB (estimated, depends on subset size).
- Processed JSONL: ~2–5 GB.

### Validation Steps

1. Verify Terms of Use accepted on Hugging Face.
2. Verify streaming works for Python and JavaScript subsets.
3. Verify license filtering is applied correctly.
4. Verify file content download works (requires AWS credentials).
5. Verify provenance metadata is complete.
6. Verify document count and token count match estimates.

---

## 5. French and Arabic Sources (Separate Work Items)

### Python French Documentation

- **Status:** PENDING_REVIEW
- **Action:** Verify exact repository license and attribution requirements in `python/python-docs-fr`.
- **Do not collect until license is confirmed.**

### Python Arabic Documentation

- **Status:** UNVERIFIED
- **Action:** Verify license and completeness in `Abdur-rahmaanJ/python-docs-ar`.
- **Do not collect until license is confirmed.**

### MDN French/Arabic Content

- **Status:** UNVERIFIED
- **Action:** Verify license terms for localized MDN content.
- **Do not collect until license is confirmed.**

---

## 6. General Collection Principles

1. **No download without approval.** Only collect sources marked APPROVED_CANDIDATE or explicitly approved after review.
2. **Raw data is immutable.** Once collected, raw data is never modified.
3. **Provenance is mandatory.** Every document must carry full metadata.
4. **License filtering first.** Filter by license before any processing.
5. **Streaming preferred.** Use streaming or chunked processing to respect memory constraints.
6. **Checkpointing.** Save progress periodically for resumption.
7. **Separation of stages.** Raw → Interim → Processed. Each stage is independent.
8. **No preprocessing in this phase.** Collection only. Preprocessing is Phase 2C.

---

## 7. Estimated Total Pilot Storage

| Component | Raw | Processed | Total |
|-----------|-----|-----------|-------|
| Python Docs (EN) | ~56 MB | ~200 MB | ~256 MB |
| CodeSearchNet PY+JS | ~2.6 GB | ~3 GB | ~5.6 GB |
| The Stack PY+JS subset | ~5–10 GB | ~5 GB | ~10–15 GB |
| **Total** | ~8–13 GB | ~8 GB | ~16–21 GB |

Well within the 522 GB available disk space.

---

**Awaiting your review and approval before beginning collection.**
