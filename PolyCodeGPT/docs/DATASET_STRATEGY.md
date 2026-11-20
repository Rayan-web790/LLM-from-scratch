# PolyCodeGPT — Dataset Strategy

## 1. Dataset Objectives

Build a small, high-quality, programming-oriented multilingual text corpus for training a decoder-only Transformer language model from random initialization. The corpus must support:

- **Programming content:** Python (primary), JavaScript (primary), Java, C++, SQL (secondary).
- **Natural language content:** English (primary), French (secondary), Arabic (secondary).
- **Content types:** Source code, code comments, docstrings, documentation, and technical explanations.
- **Pilot target:** 10–20 million tokens, subject to data availability and hardware constraints.

The dataset must be fully traceable: every document must carry provenance metadata including source, license, and original URL.

## 2. Programming-Language Priorities

| Priority | Languages | Rationale |
|----------|-----------|-----------|
| Primary | Python, JavaScript | Highest availability of licensed code and documentation; aligns with project objectives |
| Secondary | Java, C++, SQL | Available in CodeSearchNet and The Stack; add if suitable licensed material exists |
| Excluded (for now) | Ruby, Go, PHP | Not in project scope; may be added later |

## 3. Natural-Language Priorities

| Priority | Languages | Rationale |
|----------|-----------|-----------|
| Primary | English | Largest volume of programming documentation and code comments |
| Secondary | French | Python documentation has an official French translation; MDN has some French content |
| Secondary | Arabic | Python documentation has an official Arabic translation (status needs verification) |

## 4. Candidate Dataset Sources

### 4.1 CodeSearchNet

- **Source:** https://github.com/github/CodeSearchNet (archived; Zenodo mirror available)
- **Languages:** Python, JavaScript, Java, Go, PHP, Ruby
- **Content:** ~6 million functions with ~2 million having associated documentation (docstrings/comments)
- **Format:** JSONL, one example per function
- **Size:** ~5.1 GB compressed (all languages combined)
- **License:** Per-repository. License metadata stored in `{language}_licenses.pkl` files. The key is `owner/repo` and the value is a tuple of `(path, license_content)`.
- **Provenance:** Repository name, file path, function name, URL with line numbers
- **Deduplication:** Repository-level split (same repo only in one partition: train/valid/test)
- **Status:** PENDING_REVIEW — Must filter by explicit license metadata. Do not use examples with unknown licenses.

### 4.2 The Stack (BigCode)

- **Source:** https://huggingface.co/datasets/bigcode/the-stack
- **Languages:** 30+ (v1.0) to 358 (v1.1) to 658 (v2)
- **Content:** Full source code files from permissively licensed GitHub repositories
- **Format:** Parquet (metadata) + S3 (file contents)
- **Size:** 3–6 TB (near-deduplicated)
- **License:** Per-repository. 193 permissive licenses in v1.1+. Copyleft (MPL/EPL/LGPL) excluded in v1.1+.
- **Provenance:** Repository name, file path, Software Heritage ID
- **Access:** Gated — requires acceptance of Terms of Use. Bulk download requires agreement with Software Heritage and INRIA.
- **Streaming:** Available for metadata; file content download requires AWS credentials.
- **Status:** PENDING_REVIEW — Large size makes full download infeasible. Streaming a small subset of Python and JavaScript files is feasible.

### 4.3 Python Official Documentation

- **Source:** https://docs.python.org
- **Language:** Python
- **Natural language:** English (primary), with official French and Arabic translations
- **Content:** Full Python documentation (reStructuredText)
- **License:** PSF License 2.0 (documentation prose); code examples dual-licensed under PSF-2.0 and 0BSD (from Python 3.8.6+)
- **Provenance:** docs.python.org URLs
- **Attribution:** Retain PSF copyright notice and license text
- **Status:** APPROVED_CANDIDATE (English). French and Arabic translations are PENDING_REVIEW / UNVERIFIED.

### 4.4 MDN Web Docs

- **Source:** https://developer.mozilla.org
- **Language:** JavaScript (and web technologies)
- **Natural language:** English (primary)
- **Content:** Web development documentation and code examples
- **License:** CC-BY-SA 2.5 (prose); CC0 (code examples added after 2010-08-20); MIT (code examples before 2010-08-20)
- **Provenance:** Page URLs
- **Attribution:** "Mozilla Contributors" + link to specific page
- **Status:** PENDING_REVIEW (English). Prose deferred until ShareAlike implications are reviewed. Code examples require individual provenance and date verification. French/Arabic MDN content is UNVERIFIED.

### 4.5 Stack Overflow / Stack Exchange

- **Source:** https://stackoverflow.com
- **Content:** Q&A with code snippets and explanations
- **License:** CC-BY-SA 2.5 (pre-2011-04-08), CC-BY-SA 3.0 (2011-04-08 to 2018-05-02), CC-BY-SA 4.0 (post-2018-05-02)
- **Attribution:** Author name + link to original post
- **ShareAlike:** Derivative works must be under same license
- **Status:** PENDING_REVIEW — ShareAlike implications for model training require careful legal review. May be suitable for academic pilot but requires explicit approval.

## 5. License and Provenance Strategy

### 5.1 Core Principles

1. **No source without verified license.** Every document must have a known, explicit license. Unknown-license content is excluded.
2. **Provenance is mandatory.** Every document must record: source name, URL, license identifier, retrieval date, and programming/natural language.
3. **Repository license ≠ content license.** A repository's software license does not automatically apply to all files within it. Per-file license detection is required where possible.
4. **License filtering before training.** Content must be filtered by license before any training data is assembled.
5. **Attribution tracking.** Sources requiring attribution must have their attribution metadata preserved in the manifest.
6. **Approved candidate ≠ downloaded.** A source marked APPROVED_CANDIDATE has passed license review but has not yet been collected. Approval is a prerequisite, not an action.
7. **Publicly accessible ≠ unrestricted.** A source being publicly available does not mean its content is free to reuse for model training.
8. **Each selected training example must retain provenance.** No document enters the training corpus without full metadata (source, URL, license, retrieval date).

### 5.2 License Categories

| Category | Licenses | Training Suitability |
|----------|----------|---------------------|
| Permissive | MIT, Apache-2.0, BSD-2/3-Clause, 0BSD, CC0, PSF-2.0 | ✅ Approved |
| Weak copyleft | MPL-2.0, EPL-2.0, LGPL-2.1/3.0 | ⚠️ Pending review |
| Strong copyleft | GPL-2.0/3.0, AGPL-3.0 | ❌ Rejected (for now) |
| ShareAlike | CC-BY-SA 2.5/3.0/4.0 | ⚠️ Pending review (ShareAlike implications) |
| Unknown | (none detected) | ❌ Excluded |

### 5.3 Provenance Metadata Schema

Every document in the processed corpus must include:

```json
{
  "doc_id": "unique identifier",
  "source_id": "source registry ID",
  "source_url": "original URL",
  "license": "license identifier",
  "language_code": "programming language",
  "natural_language": "en/fr/ar",
  "content_type": "code/documentation/explanation",
  "repo_name": "repository name (if applicable)",
  "file_path": "file path (if applicable)",
  "retrieval_date": "ISO date",
  "attribution": "attribution string (if required)"
}
```

## 6. Data Quality Requirements

1. **Encoding:** All text must be valid UTF-8. Arabic text must be properly handled (RTL not relevant for training, but character encoding must be correct).
2. **Minimum content length:** Documents shorter than 50 characters (after stripping) should be excluded.
3. **Maximum content length:** Documents longer than 100,000 characters should be split or excluded.
4. **Language detection:** Programming language must be explicitly tagged, not inferred from content.
5. **No binary or non-text files.**
6. **No auto-generated code** (e.g., minified JavaScript, generated documentation) unless explicitly included.
7. **No duplicate content** within or across sources.

## 7. Deduplication Strategy

1. **Exact deduplication:** Remove byte-identical documents.
2. **Near-deduplication:** Use MinHash/LSH or similar techniques to detect near-duplicate documents (threshold: Jaccard similarity > 0.9).
3. **Repository-level deduplication:** Ensure documents from the same repository do not appear in multiple splits.
4. **Cross-source deduplication:** Check for content overlap between CodeSearchNet and The Stack (both derive from GitHub).

## 8. Train/Validation/Test Splitting Strategy

1. **Repository-level splitting:** All documents from the same repository must be in the same split. This prevents data leakage.
2. **Split ratios:** 90% train, 5% validation, 5% test.
3. **Stratification:** Ensure proportional representation of each programming language and natural language across splits.
4. **No overlap:** Verify that no repository appears in more than one split.

## 9. Data Leakage Prevention

1. **Repository-level splits** prevent code from the same project appearing in both train and test.
2. **Deduplication** prevents near-identical content from leaking across splits.
3. **Temporal awareness:** If using Stack Overflow, consider temporal splits (train on older content, test on newer).
4. **Evaluation isolation:** Test set must be held out and never used during training or hyperparameter tuning.

## 10. Expected Pilot Corpus Size

| Component | Estimated Tokens | Source |
|-----------|-----------------|--------|
| Python code + docstrings | 5–8M | CodeSearchNet Python (filtered) + The Stack (Python subset) |
| JavaScript code + docstrings | 3–5M | CodeSearchNet JavaScript (filtered) + The Stack (JS subset) |
| Python documentation (English) | 1–2M | Python official docs |
| JavaScript documentation (English) | 1–2M | MDN Web Docs |
| French technical content | 0.5–1M | Python docs French translation (if approved) |
| Arabic technical content | 0.5–1M | Python docs Arabic translation (if approved) |
| **Total** | **10–20M** | |

**Note:** These are estimates subject to actual data availability and license filtering. The pilot target of 10–20 million tokens is a planning figure, not a guarantee of model performance.

## 11. Hardware-Aware Data Processing Strategy

- **RAM:** 16 GB — process data in streaming/chunked mode. Never load entire datasets into memory.
- **GPU VRAM:** 8 GB — not relevant for data processing, but limits batch sizes during training.
- **Disk:** 522 GB available — sufficient for raw + interim + processed data.
- **Format:** JSONL for sequential processing. CSV for manifests. Parquet for structured metadata if needed.
- **Batch size:** Process in chunks of 1,000–10,000 documents at a time.
- **Separation:** Raw data is immutable after collection. Interim data holds partially processed files. Processed data holds final training-ready files.
- **Checkpointing:** Save progress periodically to allow resumption after interruption.

## 12. Known Limitations and Risks

1. **CodeSearchNet is archived.** The official GitHub repository is archived; data must be obtained from Zenodo mirror. License metadata must be manually matched.
2. **The Stack requires gated access.** Bulk download requires Software Heritage/INRIA agreement. Streaming a small subset is feasible but file content download requires AWS credentials.
3. **Arabic technical content is scarce.** Official Python documentation has an Arabic translation, but its license status and completeness need verification.
4. **French technical content is limited.** Python docs French translation exists; MDN French content exists but license verification needed.
5. **ShareAlike implications.** CC-BY-SA content (MDN, Stack Overflow) may impose ShareAlike requirements on derivative models. This needs legal review.
6. **License filtering reduces available data.** Filtering by permissive licenses will significantly reduce the volume of usable content from CodeSearchNet and The Stack.
7. **No evaluation of data quality yet.** Data quality assessment requires actual data inspection, which is deferred to Phase 2B.
8. **Estimated sizes are approximate.** Actual token counts will only be known after tokenization.
