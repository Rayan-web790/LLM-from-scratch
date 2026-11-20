"""Hashing and integrity helpers for tokenized data artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path, chunk_size: int = 8 << 20) -> str:
    """Stream a file through SHA-256 without loading it into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def disk_space(path: str | Path) -> dict[str, float]:
    """Report free/total disk space in bytes and GiB for the volume holding ``path``."""
    probe = Path(path)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    return {
        "total_bytes": float(usage.total),
        "free_bytes": float(usage.free),
        "total_gib": round(usage.total / 1024**3, 2),
        "free_gib": round(usage.free / 1024**3, 2),
    }


def require_disk_space(path: str | Path, needed_bytes: int, headroom: float = 1.2) -> dict[str, float]:
    """Raise before writing if the target volume lacks room for the planned output."""
    space = disk_space(path)
    required = int(needed_bytes * headroom)
    if space["free_bytes"] < required:
        raise OSError(
            f"insufficient disk space at {path}: need ~{required / 1024**3:.2f} GiB "
            f"(with {headroom:.0%} headroom), have {space['free_gib']} GiB free"
        )
    space["required_bytes_with_headroom"] = float(required)
    return space


def load_manifest(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def verify_outputs(manifest: dict[str, Any], base_dir: str | Path | None = None) -> dict[str, Any]:
    """Recompute each split's size and SHA-256 and compare against the manifest.

    Returns a report with ``ok`` and per-split details; never repairs anything.
    """
    base = Path(base_dir) if base_dir is not None else Path(manifest["output_dir"])
    checks: list[dict[str, Any]] = []

    for split, info in manifest["splits"].items():
        path = base / info["file"]
        if not path.is_file():
            checks.append({"split": split, "ok": False, "error": "missing file", "file": str(path)})
            continue
        actual_size = os.path.getsize(path)
        actual_sha = sha256_file(path)
        ok = actual_size == info["file_size_bytes"] and actual_sha == info["sha256"]
        checks.append(
            {
                "split": split,
                "file": str(path),
                "ok": ok,
                "expected_size": info["file_size_bytes"],
                "actual_size": actual_size,
                "expected_sha256": info["sha256"],
                "actual_sha256": actual_sha,
            }
        )

    return {"ok": all(c["ok"] for c in checks) and manifest.get("completed_successfully", False), "splits": checks}
