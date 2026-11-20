# Python Documentation — Training Eligibility Classification (Phase 2B.1.1)

> Source: `PYDOCS` — Official Python Documentation (English), CPython tag `v3.11.9`
> Raw path: `data/raw/python_docs_en/cpython-v3.11.9/`
> Manifest: `data/manifests/python_docs_en_files.csv` (569 rows)
>
> **Disclaimer.** This document records a *dataset-governance* decision about what to
> feed into the initial text corpus. It is **not legal advice**. Licence terms (PSF-2.0,
> the 0BSD dual-licence for code examples, and the third-party notices reproduced in
> `Doc/license.rst`) must be re-verified by qualified counsel before any commercial
> redistribution of derived material. Raw data is never deleted by this classification.

## 1. Classification categories

| Category | Meaning |
|---|---|
| `ELIGIBLE_CANDIDATE` | May be included in the initial text corpus as-is, subject to the global retention obligations (PSF copyright notice + licence text). |
| `PENDING_MANUAL_REVIEW` | Contains mixed or third-party material; must be reviewed (and possibly split) before any use. Excluded from the initial pipeline. |
| `EXCLUDE_FROM_TEXT_CORPUS` | Intentionally excluded from the text corpus (legal text, build config, generated templates). Retained in raw data for compliance/provenance. |
| `ASSET_NOT_TEXT` | Not natural-language documentation text (binary assets, data files, images, diagrams, tooling word lists). Retained for hierarchy completeness. |

Only files classified `ELIGIBLE_CANDIDATE` are considered by the initial training
pipeline. No file is deleted; classification only controls corpus inclusion.

## 2. Source and licence basis

- 569 files total: 501 `.rst` documentation, 1 root `LICENSE`, 67 supporting assets.
- Documentation prose: PSF-2.0.
- Code examples in documentation: dual-licensed PSF-2.0 / 0BSD (Python 3.8.6+).
- Retention obligation: keep the PSF copyright notice
  ("Copyright © 2001 Python Software Foundation; All Rights Reserved") and the
  licence text with any derived material.
- `Doc/license.rst` additionally reproduces third-party licence notices for software
  incorporated into Python (Mersenne Twister, OpenSSL, zlib, expat, libffi, libmpdec,
  asyncio/uvloop, …). Those passages are **not** PSF-licensed.

## 3. Eligibility by file category

| # | File category | Count | Classification | Notes |
|---|---|---|---|---|
| 1 | RST documentation (all `Doc/**/*.rst` except `Doc/license.rst`) | 500 | `ELIGIBLE_CANDIDATE` | PSF-2.0 prose; embedded examples dual PSF-2.0/0BSD. Retain copyright notice. |
| 2 | `Doc/license.rst` | 1 | `PENDING_MANUAL_REVIEW` | Mixes PSF licence text with a third-party licence appendix (see §4). Must be split or excluded before use. |
| 3 | `Doc/copyright.rst` | 1 | `ELIGIBLE_CANDIDATE` | PSF/BeOpen/CNRI/CWI copyright lines only; no third-party material. Retain notice. |
| 4 | `Doc/library/pprint.rst` | 1 | `ELIGIBLE_CANDIDATE` | The only licence token is incidental sample PyPI metadata (`'License :: OSI Approved :: MIT License'`, `'license': 'MIT'`) inside an example output; not a licence grant or third-party code. |
| 5 | Root `LICENSE` | 1 | `EXCLUDE_FROM_TEXT_CORPUS` | Legal text. Retained for compliance/provenance; not training content. |
| 6 | Images (`.png`, `.svg`, `.pdf`, `.ps`, `.dia`) | 12 | `ASSET_NOT_TEXT` | Binary/non-text visual assets. |
| 7 | Data files (`.dat`, `.abi`, `.xml`, `.csv` word list) | 4 | `ASSET_NOT_TEXT` | `susp-ignored.csv` is a Sphinx spelling-suppression word list, not prose. |
| 8 | C/C++ headers and samples (`.c`, `.h`, `.inc`) | 9 | `ASSET_NOT_TEXT` | Documentation code snippets/build samples, not standalone training corpus text. |
| 9 | Python build helpers (`.py`, incl. `Doc/conf.py`, `Doc/tools/extensions/*`, `Doc/includes/*.py`) | 26 | `ASSET_NOT_TEXT` | Sphinx configuration/extensions and doc code samples; not general-purpose Python training code. |
| 10 | HTML templates (`.html`) | 7 | `ASSET_NOT_TEXT` | Sphinx templates. |
| 11 | JavaScript (`changelog_search.js`) | 1 | `ASSET_NOT_TEXT` | Doc search helper. |
| 12 | Build/config files (`Makefile`, `make.bat`, `constraints.txt`, `requirements*.txt`, `.nitignore`) | 6 | `ASSET_NOT_TEXT` | Build tooling. (`make.bat` corrected to exact upstream blob bytes on 2026-11-15.) |

Totals: `ELIGIBLE_CANDIDATE` = 501 (500 RST + `copyright.rst` + `pprint.rst` is already in the 500? see note) — see §5 for exact counts.

> Note: `Doc/library/pprint.rst` and `Doc/copyright.rst` are part of the 501 `.rst` count.
> Excluding only `Doc/license.rst`, the eligible RST count is **500**.
> Adding the two explicitly reviewed RST files does not change that (they are already
> within the 500). So: `ELIGIBLE_CANDIDATE` = 500 RST files.

## 4. Keyword-flagged file review (detailed)

Four paths matched a licence-keyword scan. That is four *matches*, not four problems:
only `Doc/license.rst` remains unresolved.

### 4.1 `Doc/license.rst` — `PENDING_MANUAL_REVIEW`

- **Exact location of flagged content:** lines 296–1098, section
  `Licenses and Acknowledgements for Incorporated Software` (anchor `.. _OtherLicenses:`).
  The PSF licence and copyright history occupy lines 1–295.
- **Nature of the third-party material:** verbatim licence notices and acknowledgements
  for software incorporated into the Python distribution (Mersenne Twister, OpenSSL,
  zlib, expat, libffi, libmpdec, asyncio/uvloop, …).
- **Relevant licence notice:** each block carries its own notice (MIT, 3-clause BSD,
  Apache-2.0, OpenSSL licence, …). These are **not** PSF-2.0.
- **Suitable for training?** Not as a single file. The first part (PSF licence + history)
  is eligible; the appendix reproduces third-party licence texts that must not be
  attributed to PSF or silently mixed into a PSF-licensed corpus.
- **Action:** exclude from the initial pipeline **or** split into two documents
  (`license_psf.rst` eligible, `third_party_licenses.rst` excluded/pending) during
  preprocessing in a later phase. Do not alter raw data now.

### 4.2 `Doc/copyright.rst` — `ELIGIBLE_CANDIDATE`

- **Exact location of flagged content:** the whole file (19 lines) is a copyright notice;
  no separate flagged passage. The earlier flag was a keyword hit on "All rights reserved".
- **Nature of the third-party material:** none. Only PSF / BeOpen / CNRI / CWI copyright lines.
- **Relevant licence notice:** references `:ref:`history-and-license`` (i.e. `Doc/license.rst`).
- **Suitable for training?** Yes. Must retain the copyright notice in any derived use.
- **Action:** none.

### 4.3 `Doc/library/pprint.rst` — `ELIGIBLE_CANDIDATE`

- **Exact location of flagged content:** lines 281, 314, 361, 410 — inside the
  `.. _pprint-example:` example block, which prints a sample PyPI project's metadata.
  Tokens: `'License :: OSI Approved :: MIT License'` (line 281) and `'license': 'MIT'`
  (lines 314, 361, 410).
- **Nature of the third-party material:** none. These strings are *data* produced by
  pretty-printing a real PyPI response; they are not a licence grant, code, or notice
  attributable to this project.
- **Relevant licence notice:** none applicable to the documentation itself.
- **Suitable for training?** Yes — the file is PSF-2.0 documentation with dual-licensed
  code examples; the MIT string is incidental sample output.
- **Action:** none; keep the incidental mention as-is (it is accurate data).

### 4.4 `Doc/tools/susp-ignored.csv` — `ASSET_NOT_TEXT`

- **Exact location of flagged content:** lines 198, 263–269 (and similar), where the
  spelling-suppression list includes snippets containing licence tokens such as
  `'License :: OSI Approved :: MIT License'` and fragments of the third-party notices
  from `Doc/license.rst`.
- **Nature of the third-party material:** none new. The file is a Sphinx spelling-checker
  suppression list (word/phrase allow-list), not documentation prose. It merely quotes
  substrings that occur elsewhere in the docs.
- **Relevant licence notice:** N/A — tooling configuration.
- **Suitable for training?** No. It is not natural-language documentation; including it
  would inject fragmented licence text and build noise into the corpus.
- **Action:** exclude (already `ASSET_NOT_TEXT`).

## 5. Counts summary

| Classification | Count |
|---|---|
| `ELIGIBLE_CANDIDATE` | 500 RST files (all 501 `.rst` minus `Doc/license.rst`) |
| `PENDING_MANUAL_REVIEW` | 1 (`Doc/license.rst`) |
| `EXCLUDE_FROM_TEXT_CORPUS` | 1 (root `LICENSE`) |
| `ASSET_NOT_TEXT` | 67 (all `documentation_asset` rows) |
| **Total** | **569** |

`Doc/copyright.rst` and `Doc/library/pprint.rst` are inside the 500 `ELIGIBLE_CANDIDATE`
RST files — they were reviewed and cleared, not flagged. `Doc/license.rst` is the only
file that still needs manual review.

Initial pipeline input: **500 files** (`ELIGIBLE_CANDIDATE`). `Doc/license.rst` stays
excluded until its third-party appendix is split off or cleared.

## 6. Global obligations for any derived use

1. Retain the PSF copyright notice and the PSF-2.0 licence text.
2. Preserve the 0BSD dual-licence note for code examples.
3. Do not strip or alter the third-party notices in `Doc/license.rst` if that file (or
   its appendix) is ever included.
4. Do not convert `ASSET_NOT_TEXT` files into training text without a separate decision.
5. Re-verify licences before commercial use (not legal advice).

## 7. Verification (Phase 2B.1.1)

- Raw files present: 569 (after `Doc/make.bat` correction to exact upstream blob).
- Manifest rows: 569.
- SHA-256 + size match: 569/569, 0 mismatches.
- Byte verification vs git blobs (`git cat-file blob`): 569/569 exact matches,
  0 normalized, 0 mismatches, 0 missing, 0 extra.
- No additional dataset introduced; only `python_docs_en` under `data/raw/`.
- Raw directory structure preserved.
- CUDA environment untouched.