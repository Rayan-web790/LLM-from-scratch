#!/usr/bin/env python
"""Build the CodeSearchNet license inventory and distribution report (Phase 2B.2).

Input : data/interim/codesearchnet_license_metadata/{python,javascript}_licenses.pkl
Output: data/manifests/codesearchnet_license_inventory.csv
        data/manifests/codesearchnet_license_repo_summary.csv

Everything here is a classification exercise, NOT legal advice. Criteria are
documented in reports/codesearchnet_license_audit.md and mirrored in the
REVIEW_NOTES emitted per row.

Standard library only. Read-only on data/raw.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import sys
from collections import Counter, defaultdict

SOURCE_ID = "CSN"
RETRIEVAL_DATE = "2026-11-15"

MAX_FILE_BYTES = 512 * 1024 * 1024
FORBIDDEN_OPCODES = {
    "GLOBAL", "STACK_GLOBAL", "REDUCE", "BUILD", "OBJ", "INST",
    "NEWOBJ", "NEWOBJ_EX", "EXT1", "EXT2", "EXT4", "PERSID", "BINPERSID",
}

# --------------------------------------------------------------- families ---
# Detection is hierarchical, not additive: an LGPL document also contains the
# words "GNU General Public License", and a BSD-3 document also contains the
# BSD-2 "redistribution and use" clause. Naive additive matching produces
# spurious "multiple licence" verdicts, so each class resolves to one family.
GNU_GRANT_MARKERS = (
    "you can redistribute it and/or modify it under the terms of the gnu",
    "everyone is permitted to copy and distribute verbatim copies",
    "terms of the gnu general public license as published by the free software",
    "terms of the gnu lesser general public license as published by the free software",
    "terms of the gnu affero general public license as published by the free software",
)

# Non-copyleft families: (family, all-of phrases).
PERMISSIVE_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("CC-BY-SA", ("creative commons", "sharealike")),
    ("CC0-1.0", ("cc0 1.0",)),
    ("Unlicense", ("free and unencumbered software released into the public domain",)),
    ("WTFPL", ("do what the fuck you want",)),
    ("Apache-2.0", ("apache license", "version 2.0")),
    ("BSL-1.0", ("boost software license",)),
    ("PostgreSQL", ("postgresql license",)),
    ("Zlib", ("origin of this software must not be misrepresented",)),
    ("ISC", ("permission to use, copy, modify", "with or without fee",
             "provided that the above copyright notice")),
    ("0BSD", ("permission to use, copy, modify", "with or without fee")),
    ("MIT", ("permission is hereby granted, free of charge",)),
    ("BSD-3-Clause", ("redistribution and use in source and binary forms",
                      "neither the name")),
    ("BSD-2-Clause", ("redistribution and use in source and binary forms",)),
    ("Artistic-2.0", ("artistic license",)),
]

PERMISSIVE = {
    "MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC", "0BSD", "Zlib",
    "Unlicense", "CC0-1.0", "BSL-1.0", "PostgreSQL", "WTFPL", "PSF-2.0",
    "Artistic-2.0",
}
COPYLEFT = {"GPL", "GPL-2.0", "GPL-3.0", "LGPL", "AGPL", "MPL-2.0", "EPL", "CC-BY-SA"}
# Attribution-required content licences: usable but impose obligations, so they
# stay out of the automatic permissive bucket pending review.
ATTRIBUTION = {"CC-BY-4.0"}

GRANT_MARKERS = (
    "permission is hereby granted", "redistribution and use", "permission to use",
    "licensed under", "is free software", "public domain", "grants licensee",
    "terms and conditions",
)
RESTRICTIVE_MARKERS = ("all rights reserved", "proprietary", "confidential",
                       "no license", "not licensed")

CATEGORY_RANK = {
    "CLEAR_PERMISSIVE_CANDIDATE": 0,
    "COPYLEFT_REVIEW_REQUIRED": 1,
    "PROPRIETARY_OR_RESTRICTIVE": 2,
    "AMBIGUOUS": 3,
    "NO_LICENSE_FOUND": 4,
    "METADATA_MISSING": 5,
}


def norm(text: str) -> str:
    return " ".join(text.lower().split())


def detect_families(text: str) -> list[str]:
    hay = norm(text)
    hits: list[str] = []

    # 1. Copyleft family, resolved to exactly one value. Prefers a real GNU grant
    #    clause; a short stand-alone notice ("licensed under the GNU GPL v3") is
    #    also accepted, but a long MPL/EPL document that merely cites the GPL in
    #    its exhibit is not, which is what produced false dual-licences earlier.
    gnu_name = ("gnu affero general public license" in hay
                or "gnu lesser general public license" in hay
                or "gnu general public license" in hay)
    other_copyleft_doc = ("mozilla public license" in hay
                          or "eclipse public license" in hay)
    short_notice = gnu_name and not other_copyleft_doc and len(hay) < 4000
    if any(m in hay for m in GNU_GRANT_MARKERS) or short_notice:
        if "gnu affero general public license" in hay:
            hits.append("AGPL")
        elif "gnu lesser general public license" in hay:
            hits.append("LGPL")
        elif other_copyleft_doc:
            pass  # an MPL/EPL document that only cites the GPL by name
        elif "version 3" in hay or "v3" in hay:
            hits.append("GPL-3.0")
        elif "version 2" in hay or "v2" in hay:
            hits.append("GPL-2.0")
        else:
            hits.append("GPL")

    # 2. Other source-available copyleft families.
    if "mozilla public license" in hay:
        hits.append("MPL-2.0")
    if "eclipse public license" in hay:
        hits.append("EPL")

    # 3. Permissive families, with subsumption (BSD-3 before BSD-2, ISC before 0BSD).
    perm = [f for f, phrases in PERMISSIVE_RULES if all(p in hay for p in phrases)]
    if "BSD-3-Clause" in perm:
        perm.remove("BSD-2-Clause")
    if "ISC" in perm:
        perm.remove("0BSD")

    # 4. Underscore/escape-damaged and short-notice variants seen in the data.
    extra_any = (
        ("Unlicense", "unencumbered software released into the public"),
        ("MIT", "under the mit license"),
        ("PSF-2.0", "python software foundation license"),
        ("CC-BY-4.0", "attribution 4.0 international"),
        ("CC-BY-4.0", "creative commons attribution 4.0"),
    )
    for family, phrase in extra_any:
        if phrase in hay and family not in perm:
            perm.append(family)
    hits.extend(perm)
    return hits


def classify(text: str) -> tuple[str, str, str]:
    """Return (family, category, review_note)."""
    if not text.strip():
        return "NONE-IN-METADATA", "NO_LICENSE_FOUND", "empty license content"

    families = detect_families(text)
    hay = norm(text)

    if ".. include::" in hay and not families:
        return "INDIRECT", "AMBIGUOUS", \
            "license file only includes another file; no licence text present"

    if len(families) == 1:
        fam = families[0]
        if fam in PERMISSIVE:
            return fam, "CLEAR_PERMISSIVE_CANDIDATE", "single recognised permissive licence"
        if fam in COPYLEFT:
            return fam, "COPYLEFT_REVIEW_REQUIRED", "single recognised copyleft licence"
        if fam in ATTRIBUTION:
            return fam, "AMBIGUOUS", "attribution-required content licence; obligations to confirm"
        return fam, "AMBIGUOUS", "recognised family outside the approved list"

    if len(families) > 1:
        fam = "MULTIPLE:" + "+".join(sorted(families))
        if all(f in PERMISSIVE for f in families):
            return fam, "CLEAR_PERMISSIVE_CANDIDATE", \
                "multiple permissive licences (dual/multi-licensed), no copyleft"
        if any(f in COPYLEFT for f in families):
            return fam, "COPYLEFT_REVIEW_REQUIRED", \
                "text mixes copyleft with other families; needs review"
        return fam, "AMBIGUOUS", "multiple families detected; needs review"

    has_grant = any(m in hay for m in GRANT_MARKERS)
    has_restriction = any(m in hay for m in RESTRICTIVE_MARKERS)
    if has_restriction and not has_grant:
        return "OTHER-RESTRICTIVE", "PROPRIETARY_OR_RESTRICTIVE", \
            "restrictive language without a permission grant"
    if len(text) < 200:
        return "UNKNOWN", "AMBIGUOUS", f"unrecognised short text ({len(text)} bytes)"
    if has_grant:
        return "OTHER", "AMBIGUOUS", "grant language but unrecognised licence family"
    return "UNKNOWN", "AMBIGUOUS", "no recognisable licence markers"


# ------------------------------------------------------------ safe loading --
class _Restricted(pickle.Unpickler):
    def find_class(self, module, name):
        raise RuntimeError(f"refused global {module}.{name}")


def _scan(path: str) -> None:
    import pickletools
    with open(path, "rb") as fh:
        for opcode, _arg, _pos in pickletools.genops(fh):
            if opcode.name in FORBIDDEN_OPCODES:
                raise RuntimeError(f"{path}: forbidden opcode {opcode.name}")


def load_licenses(path: str) -> dict:
    if os.path.getsize(path) > MAX_FILE_BYTES:
        raise RuntimeError(f"{path}: exceeds size cap")
    _scan(path)
    with open(path, "rb") as fh:
        return _Restricted(fh, encoding="utf-8").load()


# ------------------------------------------------------------------ build ---
def build(lang: str, pkl_path: str, inv_writer, summary_rows: list) -> dict:
    data = load_licenses(pkl_path)
    if not isinstance(data, dict):
        raise RuntimeError(f"{lang}: expected dict, got {type(data)}")

    stats = {
        "language": lang, "repositories": len(data),
        "records": 0, "empty_content": 0, "non_utf8": 0,
        "multi_entry_repos": 0, "families": Counter(),
        "categories": Counter(), "records_per_repo": Counter(),
    }

    for repo, value in data.items():
        entries = value if isinstance(value, list) else [value]
        if len(entries) > 1:
            stats["multi_entry_repos"] += 1
        stats["records_per_repo"][len(entries)] += 1

        repo_categories = []
        repo_families = []
        for item in entries:
            if not (isinstance(item, tuple) and len(item) == 2):
                continue
            lpath, content = item
            if not isinstance(lpath, str):
                lpath = repr(lpath)
            if not isinstance(content, str):
                content = ""
            raw = content.encode("utf-8", "surrogateescape")
            has_surrogate = any(0xD800 <= ord(ch) <= 0xDFFF for ch in content)

            family, category, note = classify(content)
            stats["records"] += 1
            stats["families"][family] += 1
            stats["categories"][category] += 1
            if category == "NO_LICENSE_FOUND":
                stats["empty_content"] += 1
            if has_surrogate:
                stats["non_utf8"] += 1
                note += "; contains non-UTF-8 bytes (surrogate-escaped)"

            metadata_status = {
                "NO_LICENSE_FOUND": "empty_license_content",
                "AMBIGUOUS": "ambiguous_text",
            }.get(category, "present")
            if len(entries) > 1:
                metadata_status = "present_multiple_license_files"

            inv_writer.writerow([
                SOURCE_ID, lang, repo, lpath, family, category,
                hashlib.sha256(raw).hexdigest(), content, metadata_status, note,
            ])
            repo_categories.append(category)
            repo_families.extend(
                [f for f in family.split(":", 1)[-1].split("+")] if family else [])

        repo_category = max(repo_categories, key=lambda c: CATEGORY_RANK[c]) \
            if repo_categories else "METADATA_MISSING"
        summary_rows.append({
            "source_id": SOURCE_ID, "programming_language": lang,
            "repository": repo, "license_files": len(entries),
            "repository_category": repo_category,
            "families": "+".join(sorted(set(repo_families))),
        })

    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--interim", default="data/interim/codesearchnet_license_metadata")
    ap.add_argument("--manifests", default="data/manifests")
    args = ap.parse_args(argv)

    os.makedirs(args.manifests, exist_ok=True)
    inv_path = os.path.join(args.manifests, "codesearchnet_license_inventory.csv")
    sum_path = os.path.join(args.manifests, "codesearchnet_license_repo_summary.csv")

    summary_rows: list[dict] = []
    all_stats = []
    with open(inv_path, "w", newline="", encoding="utf-8",
              errors="surrogateescape") as inv_fh:
        writer = csv.writer(inv_fh, quoting=csv.QUOTE_MINIMAL)
        writer.writerow([
            "source_id", "programming_language", "repository", "license_file_path",
            "detected_license_family", "license_category", "license_text_sha256",
            "license_text", "metadata_status", "review_notes",
        ])
        for lang in ("python", "javascript"):
            p = os.path.join(args.interim, f"{lang}_licenses.pkl")
            all_stats.append(build(lang, p, writer, summary_rows))

    with open(sum_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    report = {"languages": all_stats,
              "inventory_rows": sum(s["records"] for s in all_stats),
              "summary_rows": len(summary_rows),
              "inventory_bytes": os.path.getsize(inv_path)}
    print(json.dumps(report, indent=2, default=str))
    for s in all_stats:
        print(f"\n== {s['language']}: repos={s['repositories']} records={s['records']}")
        for k, v in s["categories"].most_common():
            print(f"   {k}: {v}")
        for k, v in s["families"].most_common(12):
            print(f"   family {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
