"""Command-line entry points for Phase 18."""

from __future__ import annotations

import argparse
import json

from .pipeline import audit_final, dedup_and_split, extract, print_summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PolyCodeGPT Phase 18 v2 corpus builder")
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract", help="apply source/license/quality gates")
    p_extract.add_argument("--shard-limit", type=int, default=None, help="bounded CodeParrot shard count")
    p_extract.add_argument("--shard-offset", type=int, default=0)
    p_extract.add_argument("--max-candidates", type=int, default=None)
    p_extract.add_argument("--append", action="store_true")
    p_extract.add_argument("--no-base", action="store_true", help="skip CSN/docs; CodeParrot only")

    sub.add_parser("dedup", help="global dedup and repository-isolated split")
    sub.add_parser("audit", help="validate final JSONL corpus")
    p_build = sub.add_parser("build", help="run extract + dedup + audit")
    p_build.add_argument("--shard-limit", type=int, default=None)
    p_build.add_argument("--shard-offset", type=int, default=0)
    p_build.add_argument("--max-candidates", type=int, default=None)

    args = parser.parse_args(argv)
    if args.command == "extract":
        result = extract(
            shard_limit=args.shard_limit,
            shard_offset=args.shard_offset,
            max_candidates=args.max_candidates,
            append=args.append,
            include_base=not args.no_base,
        )
    elif args.command == "dedup":
        result = dedup_and_split()
    elif args.command == "audit":
        result = audit_final()
    else:
        result = {
            "extract": extract(shard_limit=args.shard_limit, max_candidates=args.max_candidates),
            "dedup": dedup_and_split(),
            "final": audit_final(),
        }
    print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
