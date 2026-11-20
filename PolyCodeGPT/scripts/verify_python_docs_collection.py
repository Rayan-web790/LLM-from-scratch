#!/usr/bin/env python
"""PolyCodeGPT — verify the raw Python docs copy against the authoritative git tag.

Compares every collected file's git blob hash (computed locally) with the blob
hash recorded in CPython tag v3.11.9. Any mismatch means the raw copy is not
byte-identical to the official source.

Usage:
    python scripts/verify_python_docs_collection.py --git-dir <path-to-sparse-clone>
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "python_docs_en" / "cpython-v3.11.9"
TAG = "v3.11.9"
PATHS = ["Doc", "LICENSE"]


def blob_hash(data: bytes) -> str:
    """Git blob hash of raw bytes (sha1 over 'blob <len>\\0' + data)."""
    header = b"blob " + str(len(data)).encode() + b"\x00"
    return hashlib.sha1(header + data).hexdigest()


def crlf_declared_paths(git_dir: Path) -> set[str]:
    """Paths whose working-tree form CPython's .gitattributes declares as CRLF."""
    out = subprocess.run(
        ["git", "-C", str(git_dir), "ls-files", "--eol", "--", *PATHS],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    declared = set()
    for line in out.splitlines():
        if "eol=crlf" in line:
            declared.add(line.rsplit("\t", 1)[-1])
    return declared


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--git-dir", required=True, help="path to the sparse clone")
    args = parser.parse_args()

    git_dir = Path(args.git_dir)
    if not (git_dir / ".git").exists():
        print(f"ERROR: not a git repository: {git_dir}")
        return 2

    out = subprocess.run(
        ["git", "-C", str(git_dir), "ls-tree", "-r", TAG, "--", *PATHS],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    expected = {}
    for line in out.splitlines():
        meta, path = line.split("\t", 1)
        _mode, _type, sha = meta.split()
        expected[path] = sha

    crlf_paths = crlf_declared_paths(git_dir)

    actual = {}
    mismatches = []
    missing = []
    normalized = []
    for path, sha in expected.items():
        local = RAW_ROOT / path
        if not local.is_file():
            missing.append(path)
            continue
        data = local.read_bytes()
        computed = blob_hash(data)
        if computed != sha and path in crlf_paths:
            # CPython declares this file as CRLF in .gitattributes, so the
            # canonical checkout carries CRLF while the stored blob is LF.
            computed = blob_hash(data.replace(b"\r\n", b"\n"))
            if computed == sha:
                normalized.append(path)
        actual[path] = computed
        if computed != sha:
            mismatches.append((path, sha, computed))

    extra = sorted(
        p.relative_to(RAW_ROOT).as_posix()
        for p in RAW_ROOT.rglob("*")
        if p.is_file() and p.relative_to(RAW_ROOT).as_posix() not in expected
    )

    print(f"tag                    : {TAG}")
    print(f"files in tag           : {len(expected)}")
    print(f"files verified         : {len(actual)}")
    print(f"hash mismatches        : {len(mismatches)}")
    print(f"missing locally        : {len(missing)}")
    print(f"extra local files      : {len(extra)}")
    print(f"declared-CRLF verified : {len(normalized)}")
    for path in normalized:
        print(f"  (CRLF per .gitattributes, blob matches): {path}")

    if mismatches:
        print("\nMISMATCHES:")
        for path, want, got in mismatches[:20]:
            print(f"  {path}\n    expected {want}\n    got      {got}")
    if missing:
        print("\nMISSING:")
        for path in missing[:20]:
            print(f"  {path}")
    if extra:
        print("\nEXTRA:")
        for path in extra[:20]:
            print(f"  {path}")

    ok = not mismatches and not missing
    print("\nRESULT:", "BYTE-IDENTICAL TO TAG" if ok else "INTEGRITY FAILURE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
