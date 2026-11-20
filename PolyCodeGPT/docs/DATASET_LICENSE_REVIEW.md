# PolyCodeGPT — Dataset License Review

> **Disclaimer:** This document is a technical assessment of publicly available license information. It is **not legal advice**. Consult a qualified legal professional before making final licensing decisions for model training or distribution.

## 1. CodeSearchNet

### What is known

- **Official repository:** https://github.com/github/CodeSearchNet (archived)
- **Zenodo mirror:** https://zenodo.org/records/7857872 (unofficial backup) and https://zenodo.org/records/7908468 (official)
- **Languages:** Go, Java, JavaScript, PHP, Python, Ruby
- **Content:** ~6 million functions, ~2 million with associated documentation
- **Format:** JSONL (one example per function) with fields: `repo`, `path`, `func_name`, `original_string`, `language`, `code`, `code_tokens`, `docstring`, `docstring_tokens`, `sha`, `partition`, `url`
- **Train/valid/test splits:** Repository-level (same repo only in one partition)
- **Size:** ~5.1 GB compressed (all languages)
- **License metadata:** Per-repository licenses stored in `{language}_licenses.pkl` files. Key = `owner/repo`, value = tuple of `(path, license_content)`.
- **License filtering:** The CodeSearchNet paper states that projects without a license or whose license does not explicitly permit redistribution of parts were removed. However, the actual license per repository is stored in the `_licenses.pkl` files and must be checked individually.
- **Repository-level provenance:** Yes — each example includes `repo` (owner/repo), `path` (file path), and `url` (GitHub URL with line numbers).

### What is uncertain

- The exact license distribution across all repositories is not documented in a single summary. The `_licenses.pkl` files must be inspected programmatically.
- The Zenodo mirror may not include the `_licenses.pkl` files. This needs verification.
- Whether "permissive" filtering was applied consistently across all languages.
- Whether near-duplicate functions exist across repositories.

### Assessment

- **Approval status:** PENDING_REVIEW
- **Action required:** Before using any CodeSearchNet data, programmatically inspect the `_licenses.pkl` files to identify the license per repository. Only include repositories with explicitly permissive licenses (MIT, Apache-2.0, BSD, etc.). Exclude repositories with unknown, copyleft, or restrictive licenses.
- **Risk:** Medium. License metadata exists but requires programmatic filtering. The repository-level license may not reflect the license of individual files.

## 2. The Stack (BigCode)

### What is known

- **Official page:** https://www.bigcode-project.org/docs/about/the-stack/
- **Hugging Face:** https://huggingface.co/datasets/bigcode/the-stack (v1) and https://huggingface.co/datasets/bigcode/the-stack-v2 (v2)
- **v1.0:** 30 languages, 18 permissive licenses, ~3 TB near-deduplicated
- **v1.1:** 358 languages, 193 permissive licenses, ~6 TB, excludes copyleft (MPL/EPL/LGPL), applies opt-out requests
- **v1.2:** Additional opt-out requests and malicious file removal
- **v1.3:** Removed opted-out repositories
- **v2:** 658 languages, larger than v1
- **License detection:** GHArchive license info for ~12% of repos; go-license-detector for remaining. ~81% of repos had no detectable license and were excluded.
- **A file was included if at least one repository containing it had a permissive license.**
- **Provenance:** Repository name, file path, Software Heritage ID
- **Access:** Gated — requires acceptance of Terms of Use. Bulk download requires agreement with Software Heritage and INRIA.
- **Streaming:** Available for metadata (file IDs). File content download requires AWS credentials and Software Heritage S3 access.
- **Terms of Use:** Requires (1) abiding by original source code licenses including attribution, (2) updating to latest usable version, (3) including Terms of Use if hosting/sharing.

### What is uncertain

- Whether The Stack's permissive license list aligns with our acceptable license categories.
- Whether streaming a small subset (Python + JavaScript only) is feasible without bulk download agreement.
- Whether file-level license attribution is sufficient (vs. repository-level).
- The exact size of a Python + JavaScript subset.

### Assessment

- **Approval status:** PENDING_REVIEW
- **Action required:** Accept Terms of Use on Hugging Face. Verify that streaming a small subset of Python and JavaScript files is feasible. Confirm license attribution per file.
- **Risk:** Medium-High. Large dataset requiring gated access. Streaming is possible but file content download requires additional agreements.

## 3. Python Official Documentation

### What is known

- **Source:** https://docs.python.org
- **License:** PSF License 2.0 for documentation prose. Code examples dual-licensed under PSF-2.0 and 0BSD (from Python 3.8.6+).
- **Attribution:** Retain PSF copyright notice ("Copyright © 2001 Python Software Foundation; All Rights Reserved") and license text.
- **Restrictions:** Trademark usage (Python logo) requires separate permission. Must summarize changes in derivative works.
- **French translation:** https://github.com/python/python-docs-fr (mirror of git.afpy.org). Translator contributions under CC0, but translation is derivative work of PSF-licensed docs, so PSF license applies.
- **Arabic translation:** https://github.com/Abdur-rahmaanJ/python-docs-ar. Coordinator: Abdur-Rahmaan Janhangeer. License status needs verification.

### What is uncertain

- Whether the French translation repository has an explicit license file.
- Whether the Arabic translation is complete and actively maintained.
- Whether the Arabic translation repository has an explicit license.

### Assessment

- **English docs:** APPROVED_CANDIDATE — License is clear (PSF-2.0 + 0BSD for code examples). Attribution required.
- **French docs:** PENDING_REVIEW — Translation exists and is official, but license verification needed.
- **Arabic docs:** UNVERIFIED — Translation exists per Python devguide, but license and completeness need verification.

## 4. MDN Web Docs

### What is known

- **Source:** https://developer.mozilla.org
- **Prose license:** CC-BY-SA 2.5 (or later version)
- **Code examples (post-2010-08-20):** CC0 (public domain, no attribution needed)
- **Code examples (pre-2010-08-20):** MIT License
- **Attribution:** "Mozilla Contributors" + link to specific page
- **Content repo:** https://github.com/mdn/content (LICENSE.md documents the licensing)
- **French/Arabic content:** MDN has localized content, but license verification for non-English content is needed.

### What is uncertain

- Whether MDN's French and Arabic content is available under the same license terms.
- Whether the MDN content repository includes all localized content or only English.

### Assessment

- **English docs + code examples:** PENDING_REVIEW — Prose under CC-BY-SA 2.5 (ShareAlike implications for model training under review). Code examples under CC0 (post-2010-08-20) or MIT (pre-2010-08-20). Individual provenance and date verification required before approving code examples. Do not collect prose until ShareAlike implications are reviewed.
- **French/Arabic MDN content:** UNVERIFIED — Needs license verification.

## 5. Stack Overflow / Stack Exchange

### What is known

- **Official licensing page:** https://stackoverflow.com/help/licensing
- **License by date:**
  - Before 2011-04-08 (UTC): CC-BY-SA 2.5
  - 2011-04-08 to 2018-05-02 (UTC): CC-BY-SA 3.0
  - On or after 2018-05-02 (UTC): CC-BY-SA 4.0
- **Attribution:** Author name (display name) + link to original post (use "share" link for short URL)
- **ShareAlike:** Derivative works must be licensed under the same or compatible license
- **Commercial use:** Allowed under CC-BY-SA
- **Data dump:** Available at archive.org/details/stackexchange

### What is uncertain

- Whether training a language model on CC-BY-SA content constitutes a "derivative work" requiring ShareAlike.
- Whether the CC-BY-SA ShareAlike requirement applies to model weights or only to direct redistribution of content.
- The legal status of using CC-BY-SA content for ML training is an active area of debate and may vary by jurisdiction.

### Assessment

- **Approval status:** PENDING_REVIEW
- **Action required:** Legal review needed before using Stack Overflow content for model training. The ShareAlike implications are significant and unresolved.
- **Risk:** High. CC-BY-SA ShareAlike may require the trained model to be released under CC-BY-SA, which conflicts with typical model distribution. Use only after legal approval.

## 6. Summary of License Assessment

| Source | License | Attribution | ShareAlike | Training Suitability | Status |
|--------|---------|-------------|------------|---------------------|--------|
| CodeSearchNet | Per-repo (varies) | Yes | Depends on repo license | Filter by permissive licenses | PENDING_REVIEW |
| The Stack | Per-repo (193 permissive) | Yes | Depends on repo license | Filter by permissive licenses | PENDING_REVIEW |
| Python Docs (EN) | PSF-2.0 + 0BSD (code) | Yes (copyright notice) | No | Suitable | APPROVED_CANDIDATE |
| Python Docs (FR) | PSF-2.0 (derivative) + CC0 (translators) | Yes | No | Likely suitable | PENDING_REVIEW |
| Python Docs (AR) | PSF-2.0 (derivative) | Yes | No | Needs verification | UNVERIFIED |
| MDN (EN) | CC-BY-SA 2.5 (prose) / CC0 / MIT (code) | Yes (Mozilla Contributors) | Yes (prose only) | Deferred (ShareAlike review) | PENDING_REVIEW |
| MDN (FR/AR) | Likely CC-BY-SA 2.5 | Yes | Yes | Needs verification | UNVERIFIED |
| Stack Overflow | CC-BY-SA 2.5/3.0/4.0 | Yes | Yes | Requires legal review | PENDING_REVIEW |

## 7. Recommendations

1. **Start with the only approved source:** Python documentation (English) is the only source currently approved for collection. MDN Web Docs (English) is now PENDING_REVIEW due to ShareAlike implications.
2. **Filter CodeSearchNet by license:** Programmatically inspect `_licenses.pkl` files and include only permissively licensed repositories. Do not assume the GitHub repository's MIT license applies to all code examples.
3. **Stream The Stack subset:** Accept Terms of Use and stream a small Python + JavaScript subset. Verify file-level license attribution. Bulk download requires Software Heritage/INRIA agreement.
4. **Defer MDN prose:** Do not collect MDN prose until CC-BY-SA ShareAlike implications for model training are reviewed.
5. **Verify MDN code examples individually:** Require reliable provenance and date evidence before approving individual code examples (CC0 post-2010-08-20, MIT pre-2010-08-20).
6. **Defer Stack Overflow:** Do not use until CC-BY-SA ShareAlike implications are resolved.
7. **Verify French/Arabic sources:** Check license files in python-docs-fr and python-docs-ar repositories before use. MDN French/Arabic content needs individual verification.
8. **Track all provenance:** Every document must carry source, URL, license, and retrieval date.
9. **Do not use any source marked UNVERIFIED or PENDING_REVIEW without additional verification.**

## 8. Unresolved Questions

1. Does training a language model on CC-BY-SA content constitute a "derivative work"?
2. What is the exact license of the Python documentation Arabic translation repository?
3. Does the Zenodo mirror of CodeSearchNet include the `_licenses.pkl` files?
4. Can The Stack be streamed in small subsets (Python + JavaScript only) without bulk download agreement?
5. What is the license of MDN's French and Arabic localized content?
6. Are there additional French or Arabic programming education sources with explicit permissive licenses?
7. How should attribution be handled in model training data (e.g., including attribution strings in training text)?
