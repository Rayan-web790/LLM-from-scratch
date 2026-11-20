"""Recompute block geometry in an existing manifest without touching the ``.bin`` streams.

Phase 5A.1 changed only the *window offsets*: the stride is now ``context_length``
instead of ``context_length + 1``, so consecutive windows overlap by one token.
The token streams themselves are unaffected, so this script never rewrites them.
It verifies every stream against its recorded SHA-256 and token count, recomputes
block and tail counts from the file sizes, and rewrites ``manifest.json`` and
``sources.csv`` (recording the previous values for audit).

Usage:
    python scripts/update_manifest_geometry.py [--output-dir DIR] [--dry-run]

Refuses to proceed if a stream is missing, truncated, or does not match the hash
in the manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from polycodegpt.data.integrity import sha256_file  # noqa: E402
from polycodegpt.data.prepare import (  # noqa: E402
    DATASET_VERSION,
    MANIFEST_VERSION,
    write_sources_csv,
)
from polycodegpt.data.stream import block_geometry, stream_token_count  # noqa: E402

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / DATASET_VERSION


def update(output_dir: Path, dry_run: bool = False) -> int:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    context_length = int(manifest["context_length"])
    stride = context_length
    previous_blocks = {name: info["blocks"] for name, info in manifest["splits"].items()}
    previous_tails = {
        name: info["dropped_trailing_tokens"] for name, info in manifest["splits"].items()
    }

    # ---------------------------------------------------------------- verify
    print("--- verifying streams against the existing manifest (read-only) ---")
    for name, info in manifest["splits"].items():
        path = output_dir / info["file"]
        if not path.is_file():
            raise SystemExit(f"{name}: stream missing: {path}")
        size = path.stat().st_size
        if size != info["file_size_bytes"]:
            raise SystemExit(
                f"{name}: size {size} != manifest {info['file_size_bytes']} for {path.name}"
            )
        tokens = stream_token_count(path)
        if tokens != info["tokens"]:
            raise SystemExit(f"{name}: {tokens} tokens != manifest {info['tokens']}")
        actual = sha256_file(path)
        if actual != info["sha256"]:
            raise SystemExit(
                f"{name}: SHA-256 {actual} != manifest {info['sha256']}. "
                "Streams must not be modified by a geometry update."
            )
        print(f"{name:6s} OK  {tokens:>12,} tokens  {actual[:24]}...  (file untouched)")

    # -------------------------------------------------------------- recompute
    print(f"\n--- recomputing geometry (window {context_length + 1}, stride {stride}) ---")
    for name, info in manifest["splits"].items():
        geometry = block_geometry(info["tokens"], context_length, stride)
        if name == "train":
            print(
                f"window_length={geometry['window_length']} "
                f"window_stride={geometry['window_stride']} "
                f"window_overlap={geometry['window_overlap']}"
            )
        print(
            f"{name:6s} blocks {previous_blocks[name]:>9,} -> {geometry['blocks']:>9,}   "
            f"tail {previous_tails[name]:>4,} -> {geometry['dropped_trailing_tokens']:>4,}"
        )
        info["blocks"] = geometry["blocks"]
        info["dropped_trailing_tokens"] = geometry["dropped_trailing_tokens"]

    first = block_geometry(manifest["splits"]["train"]["tokens"], context_length, stride)
    manifest["manifest_version"] = MANIFEST_VERSION
    manifest["context_length"] = context_length
    manifest["window_length"] = first["window_length"]
    manifest["window_stride"] = first["window_stride"]
    manifest["window_overlap"] = first["window_overlap"]
    manifest["block_span"] = first["window_length"]
    manifest["block_scheme"] = "overlapping"
    manifest["block_indexing"] = "start = i * window_stride, tokens[start:start + window_length]"
    manifest["totals"]["blocks"] = sum(info["blocks"] for info in manifest["splits"].values())
    manifest["geometry_revision"] = {
        "at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reason": (
            "Phase 5A.1: window stride changed from context_length + 1 "
            "(non-overlapping) to context_length (one-token overlap) so no "
            "next-token target is skipped at a block boundary"
        ),
        "streams_regenerated": False,
        "streams_modified": False,
        "previous": {
            "window_stride": manifest["context_length"] + 1,
            "block_scheme": "non_overlapping",
            "blocks": previous_blocks,
            "dropped_trailing_tokens": previous_tails,
        },
    }

    print(f"\ntotal blocks: {manifest['totals']['blocks']:,}")

    if dry_run:
        print("\ndry run: manifest.json and sources.csv not written")
        return 0

    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=False)
    write_sources_csv(manifest, output_dir / "sources.csv")
    print(f"\nupdated {manifest_path.name} and sources.csv")

    # ---------------------------------------------------------- post-verify
    for name, info in manifest["splits"].items():
        actual = sha256_file(output_dir / info["file"])
        if actual != info["sha256"]:
            raise SystemExit(f"{name}: stream changed during the metadata update ({actual})")
    print("stream hashes unchanged after the metadata update")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return update(args.output_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
