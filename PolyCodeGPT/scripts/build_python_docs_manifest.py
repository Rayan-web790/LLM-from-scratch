#!/usr/bin/env python
"""PolyCodeGPT — build the provenance manifest for the collected Python docs (EN).

Reads the immutable raw copy at data/raw/python_docs_en/cpython-v3.11.9,
computes SHA-256 for every collected file, and writes a CSV manifest.

Read-only with respect to the raw data. No dependency beyond the stdlib.
"""

from __future__ import annotations

import csv
import hashlib
import re
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "python_docs_en" / "cpython-v3.11.9"
MANIFEST = PROJECT_ROOT / "data" / "manifests" / "python_docs_en_files.csv"

SOURCE_ID = "PYDOCS"
DATASET_NAME = "Python Official Documentation (English)"
VERSION = "3.11.9"
REPO_URL = "https://github.com/python/cpython"
REPO_URL_AT_TAG = "https://github.com/python/cpython/tree/v3.11.9/Doc"
RETRIEVAL_DATE = date.today().isoformat()

# Indicators that a file may carry third-party or non-PSF licensing material.
THIRD_PARTY_PATTERNS = [
    (re.compile(r"All rights reserved", re.IGNORECASE), "contains 'All rights reserved' notice"),
    (re.compile(r"\bMIT (?:License|licen[cs]e)\b"), "mentions MIT license"),
    (re.compile(r"\bGNU General Public License\b"), "mentions GNU GPL"),
    (re.compile(r"\bApache License\b"), "mentions Apache License"),
    (re.compile(r"\bBSD[- ]?(?:2|3|Zero)?[- ]?Clause\b"), "mentions BSD license"),
    (re.compile(r"\bMozilla Public License\b"), "mentions MPL"),
    (re.compile(r"\bCC[- ]BY\b"), "mentions Creative Commons BY"),
]

FIELDNAMES = [
    "source_id",
    "dataset_name",
    "version",
    "original_repository_url",
    "original_file_path",
    "local_file_path",
    "source_license",
    "file_sha256",
    "file_size_bytes",
    "retrieval_date",
    "natural_language",
    "content_type",
    "license_notes",
]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def license_notes_for(path: Path, relative: str) -> tuple[str, str]:
    """Return (source_license, license_notes) for a collected file."""
    if relative == "LICENSE":
        return (
            "PSF-2.0",
            "Root CPython LICENSE file (History of the Software + PSF LICENSE "
            "AGREEMENT VERSION 2 + BeOpen/CNRI/CWI historical licenses). "
            "Collected verbatim as authoritative license text.",
        )
    if relative in ("Doc/license.rst", "Doc/copyright.rst"):
        return (
            "PSF-2.0",
            "Documentation license and copyright page. Includes the PSF License "
            "and an appendix of licenses for incorporated third-party software; "
            "that appendix describes software, not the documentation prose. "
            "Requires manual review before any downstream use of the appendix.",
        )

    # Content-based flagging for files that mention other licenses.
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ("PSF-2.0", "unreadable during scan; license not content-verified")

    hits = [note for pattern, note in THIRD_PARTY_PATTERNS if pattern.search(text)]
    if hits:
        return (
            "PSF-2.0",
            "FLAGGED for manual review: " + "; ".join(sorted(set(hits)))
            + ". Reference to a third-party license in prose does not change the "
              "documentation license, but the passage must be reviewed before use.",
        )

    return (
        "PSF-2.0",
        "Documentation prose under PSF-2.0; embedded code examples dual-licensed "
        "PSF-2.0 / 0BSD (Python 3.8.6+). No third-party license notice detected.",
    )


def main() -> int:
    if not RAW_ROOT.is_dir():
        raise SystemExit(f"Raw directory missing: {RAW_ROOT}")

    files = sorted(p for p in RAW_ROOT.rglob("*") if p.is_file())
    rst_files = [p for p in files if p.suffix == ".rst"]
    supporting = [p for p in files if p.suffix != ".rst"]

    rows = []
    for path in rst_files + supporting:
        relative = path.relative_to(RAW_ROOT).as_posix()
        source_license, notes = license_notes_for(path, relative)
        if path.suffix == ".rst":
            content_type = "documentation"
        elif relative == "LICENSE":
            content_type = "license_text"
        else:
            content_type = "documentation_asset"
            notes = (
                "Non-.rst asset shipped with the Doc/ tree (image, data, script "
                "or config). Collected for completeness of the source hierarchy; "
                "not intended as training text. "
                + notes
            )
        rows.append(
            {
                "source_id": SOURCE_ID,
                "dataset_name": DATASET_NAME,
                "version": VERSION,
                "original_repository_url": f"{REPO_URL}/blob/v{VERSION}/{relative}",
                "original_file_path": relative,
                "local_file_path": path.relative_to(PROJECT_ROOT).as_posix(),
                "source_license": source_license,
                "file_sha256": sha256_of(path),
                "file_size_bytes": path.stat().st_size,
                "retrieval_date": RETRIEVAL_DATE,
                "natural_language": "en",
                "content_type": content_type,
                "license_notes": notes,
            }
        )

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"raw files on disk : {len(files)}")
    print(f".rst files        : {len(rst_files)}")
    print(f"supporting files  : {len(supporting)}")
    print(f"manifest rows     : {len(rows)}")
    print(f"manifest path     : {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
