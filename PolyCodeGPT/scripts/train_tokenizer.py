#!/usr/bin/env python
"""Executable entry point: train the PolyCodeGPT byte-level BPE tokenizer.

Usage:
    .venv\\Scripts\\python.exe scripts\\train_tokenizer.py
    .venv\\Scripts\\python.exe scripts\\train_tokenizer.py --vocab-size 16000 --min-frequency 2

Trains on the combined training partition plus the 500 eligible Python
documentation files. Never reads valid.jsonl or test.jsonl. CPU only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from polycodegpt.tokenization.train import TRAIN_JSONL, train  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the PolyCodeGPT byte-level BPE tokenizer.")
    parser.add_argument("--vocab-size", type=int, default=16000, help="target vocabulary size (default 16000)")
    parser.add_argument("--min-frequency", type=int, default=2, help="minimum merge frequency (default 2)")
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")
    args = parser.parse_args(argv)

    if not TRAIN_JSONL.is_file():
        print(f"training corpus not found: {TRAIN_JSONL}", file=sys.stderr)
        return 1

    train(vocab_size=args.vocab_size, min_frequency=args.min_frequency, verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
