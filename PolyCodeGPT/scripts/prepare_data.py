"""Prepare binary token streams for PolyCodeGPT (Phase 5A).

Usage:
    python scripts/prepare_data.py [--output-dir DIR] [--force]

Reads the approved combined corpus partitions and the 500 approved Python
documentation files, serializes them with serializer 1.1.0, tokenizes with the
v1.1 tokenizer, and writes uint16 token streams plus a manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from polycodegpt.data.integrity import disk_space, require_disk_space, verify_outputs  # noqa: E402
from polycodegpt.data.prepare import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    EXPECTED_SPLIT_SHA256,
    prepare_dataset,
)
from polycodegpt.model.config import ModelConfig  # noqa: E402

#: Conservative upper bound for a uint16 stream derived from input bytes.
ESTIMATE_RATIO = 0.35


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", type=Path, default=None, help="model YAML (default: configs/model.yaml)")
    parser.add_argument("--no-verify-sources", action="store_true",
                        help="skip the expected SHA-256 checks on corpus files")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing prepared dataset directory")
    parser.add_argument("--buffer-tokens", type=int, default=1 << 20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir: Path = args.output_dir

    if (output_dir / "manifest.json").exists() and not args.force:
        print(f"refusing to overwrite existing dataset: {output_dir}/manifest.json exists.")
        print("Re-run with --force to rebuild.")
        return 2

    config = ModelConfig.from_yaml(args.config) if args.config else ModelConfig.from_yaml()
    print(f"context_length: {config.context_length} (from model config)")
    print(f"vocab_size:     {config.vocab_size}")

    # --- disk space, before any large write -------------------------------
    corpus_dir = PROJECT_ROOT / "data" / "processed" / "polycodegpt_corpus"
    input_bytes = sum((corpus_dir / f"{s}.jsonl").stat().st_size for s in ("train", "valid", "test"))
    needed = int(input_bytes * ESTIMATE_RATIO)
    space = require_disk_space(output_dir, needed)
    print(f"disk free:      {space['free_gib']} GiB of {space['total_gib']} GiB")
    print(f"estimated out:  {needed / 1024**2:.0f} MiB (uint16, worst case)")

    print("\npreparing splits ...")
    started = time.perf_counter()
    manifest = prepare_dataset(
        output_dir=output_dir,
        config_path=args.config,
        corpus_dir=corpus_dir,
        buffer_tokens=args.buffer_tokens,
        verify_sources=not args.no_verify_sources,
    )
    elapsed = time.perf_counter() - started

    print("\n--- summary ---")
    print(
        f"window={manifest['window_length']} stride={manifest['window_stride']} "
        f"overlap={manifest['window_overlap']} context={manifest['context_length']}"
    )
    for name, split in manifest["splits"].items():
        print(
            f"{name:6s} tokens={split['tokens']:>12,} blocks={split['blocks']:>10,} "
            f"dropped={split['dropped_trailing_tokens']:>3,} "
            f"size={split['file_size_bytes'] / 1024**2:>8.2f} MiB "
            f"records={split['records']:>8,} docs={split['documents']:>4,}"
        )
    print(
        f"total  tokens={manifest['totals']['tokens']:,} "
        f"blocks={manifest['totals']['blocks']:,} "
        f"failures={manifest['totals']['failures']}"
    )
    print(f"elapsed: {elapsed:.1f}s")

    report = verify_outputs(manifest, output_dir)
    for check in report["splits"]:
        status = "OK" if check["ok"] else "MISMATCH"
        print(f"verify {check['split']:6s} {status} {check.get('actual_sha256', '')[:16]}...")

    ok = report["ok"] and manifest["completed_successfully"]
    print(f"\nmanifest: {output_dir / 'manifest.json'}")
    print("PREPARATION " + ("SUCCEEDED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
