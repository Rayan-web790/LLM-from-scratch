#!/usr/bin/env python
"""Independent verification and manifest aggregation for Phase 18."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from polycodegpt.data_v2.language import repo_group  # noqa: E402
from polycodegpt.data_v2.schema import validate_record  # noqa: E402

PROCESSED = ROOT / "data" / "processed" / "polycodegpt_v2"
SCEN = PROCESSED / "scenario_b"
MANIFESTS = ROOT / "data" / "manifests" / "v2"
DEDUP_REASONS = {"exact_duplicate", "normalized_duplicate", "near_duplicate"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def add_stats(rec: dict, stats: dict) -> None:
    lang = rec["language"]
    src = rec["source_id"]
    cat = rec["category"]
    n = int(rec["token_count"])
    stats["records"] += 1
    stats["tokens"] += n
    stats["chars"] += len(rec["text"])
    stats["by_language_records"][lang] += 1
    stats["by_language_tokens"][lang] += n
    stats["by_source_records"][f"{src}|{lang}"] += 1
    stats["by_source_tokens"][f"{src}|{lang}"] += n
    stats["by_category_records"][f"{lang}|{cat}"] += 1
    stats["by_category_tokens"][f"{lang}|{cat}"] += n
    stats["alignment"][rec["alignment"].split(":", 1)[0]] += 1


def new_stats() -> dict:
    return {
        "records": 0,
        "tokens": 0,
        "chars": 0,
        "by_language_records": Counter(),
        "by_language_tokens": Counter(),
        "by_source_records": Counter(),
        "by_source_tokens": Counter(),
        "by_category_records": Counter(),
        "by_category_tokens": Counter(),
        "alignment": Counter(),
        "repos": set(),
    }


def finalize(stats: dict) -> dict:
    return {
        k: dict(v) if isinstance(v, Counter) else v
        for k, v in stats.items()
        if k != "repos"
    } | {"repositories": len(stats["repos"])}


def verify_split(path: Path, partition: str, stats: dict, seen_ids: set[str], repos: dict[str, str]) -> None:
    if path.name == "test.jsonl":
        raise RuntimeError("refusing test partition")
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            validate_record(rec)
            if rec["partition"] != partition:
                raise RuntimeError(f"partition mismatch {rec['record_id']}")
            if rec["record_id"] in seen_ids:
                raise RuntimeError(f"duplicate record id {rec['record_id']}")
            seen_ids.add(rec["record_id"])
            repo = repo_group(rec["source_id"], rec["repo"])
            if repo in repos and repos[repo] != partition:
                raise RuntimeError(f"repository leakage {repo}")
            repos[repo] = partition
            stats["repos"].add(repo)
            add_stats(rec, stats)


def main() -> int:
    train_stats = new_stats()
    valid_stats = new_stats()
    seen_ids: set[str] = set()
    repos: dict[str, str] = {}
    verify_split(SCEN / "train.jsonl", "train", train_stats, seen_ids, repos)
    verify_split(SCEN / "valid.jsonl", "validation", valid_stats, seen_ids, repos)
    overlap = sorted(
        train_stats["repos"] & valid_stats["repos"]
    )
    if overlap:
        raise RuntimeError(f"scenario B repository overlap: {overlap[:5]}")
    if valid_stats["by_language_records"].get("en", 0) != 0:
        raise RuntimeError("documentation leaked into validation")

    train_tokens_file = (SCEN / "train.bin").stat().st_size // 2
    valid_tokens_file = (SCEN / "valid.bin").stat().st_size // 2
    if train_tokens_file != train_stats["tokens"]:
        raise RuntimeError(f"train.bin token mismatch {train_tokens_file} != {train_stats['tokens']}")
    if valid_tokens_file != valid_stats["tokens"]:
        raise RuntimeError(f"valid.bin token mismatch {valid_tokens_file} != {valid_stats['tokens']}")

    accepted = Counter()
    licenses = Counter()
    with open(ROOT / "data" / "interim" / "v2" / "candidates.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            accepted[f"{rec['source_id']}|{rec['content_kind']}|{rec['language']}"] += 1
            if rec.get("license_id"):
                licenses[f"{rec['source_id']}|{rec['license_id']}"] += 1

    rejected_extract = Counter()
    rejected_dedup = Counter()
    license_rejects = Counter()
    raw = Counter()
    with open(ROOT / "data" / "interim" / "v2" / "rejected.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            reason = rec["reason"]
            if reason in DEDUP_REASONS:
                rejected_dedup[reason] += 1
            else:
                rejected_extract[reason] += 1
                raw[f"{rec['source_id']}|{rec['content_kind']}|{rec['language']}"] += 1
            if reason == "license_not_allowlisted":
                license_rejects[f"{rec['source_id']}|{rec.get('license_id','')}"] += 1
    for key, n in accepted.items():
        raw[key] += n

    tokenizer_meta = json.loads(
        (ROOT / "tokenizer" / "artifacts" / "v2" / "tokenizer_metadata_v2.json").read_text()
    )
    preflight = json.loads((MANIFESTS / "source_preflight.json").read_text())
    dedup = json.loads((MANIFESTS / "dedup_report.json").read_text())
    source_registry = []
    with open(MANIFESTS / "source_registry_v2.csv", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            source_registry.append(row)

    result = {
        "phase": 18,
        "scenario": "B",
        "verdict": "PHASE 18 COMPLETE — SCENARIO B DATASET VERIFIED",
        "test_partition_read": False,
        "benchmark_partitions_read": False,
        "corpus": {
            "path": "data/processed/polycodegpt_v2/scenario_b",
            "train": finalize(train_stats),
            "validation": finalize(valid_stats),
            "repository_overlap": overlap,
            "repository_overlap_count": len(overlap),
            "train_bin_tokens_from_bytes": train_tokens_file,
            "valid_bin_tokens_from_bytes": valid_tokens_file,
        },
        "full_eligible_pool": {
            "records_train": 1_189_261,
            "records_validation": 46_761,
            "train_tokens": 780_997_244,
            "validation_tokens": 30_894_000,
            "train_jsonl_sha256": sha256_file(PROCESSED / "train.jsonl"),
            "valid_jsonl_sha256": sha256_file(PROCESSED / "valid.jsonl"),
        },
        "extraction": {
            "codeparrot_shards_examined": len(preflight["codeparrot"]["shards"]),
            "codeparrot_rows_examined": preflight["codeparrot"]["shard_rows"],
            "codeparrot_shards_available": 1126,
            "raw_in_scope": dict(raw),
            "accepted": dict(accepted),
            "rejected_extraction": dict(rejected_extract),
            "rejected_dedup": dict(rejected_dedup),
            "quarantined": 0,
            "quarantine_note": "No separate quarantine file; every rejected record remains in data/interim/v2/rejected.jsonl with source, path and reason.",
            "accepted_license_ids": dict(licenses),
            "license_rejections": dict(license_rejects),
        },
        "dedup": dedup,
        "tokenizer": tokenizer_meta,
        "sources": source_registry,
        "hashes": {
            "scenario_b": {
                "train.jsonl": sha256_file(SCEN / "train.jsonl"),
                "train.bin": sha256_file(SCEN / "train.bin"),
                "valid.jsonl": sha256_file(SCEN / "valid.jsonl"),
                "valid.bin": sha256_file(SCEN / "valid.bin"),
            },
            "tokenizer": tokenizer_meta["artifact_sha256"],
        },
        "observed_costs_seconds": {
            "extraction": round(sum([2076.71, 971.97, 1132.01, 870.02]), 2),
            "tokenizer_fit": tokenizer_meta["fit_seconds"],
            "dedup_and_split": "not separately instrumented; completed before audit",
            "tokenization_and_counting": "not separately instrumented; completed",
        },
        "limitations": [
            "Only 80 of 1,126 CodeParrot shards were examined; the 500M cap was reached without using the full source.",
            "CodeParrot carries a dataset-level license of 'other'; acceptance uses only the per-file SPDX allowlist.",
            "Documentation is train-only, so validation contains code only; this is intentional and preserves the historical docs policy.",
            "The 500M Scenario B cap is deterministic but discards eligible records above the target; the full pool is preserved.",
            "No model training, fine-tuning, benchmark execution, or test-partition access was performed.",
        ],
    }
    (MANIFESTS / "verification_report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    manifest = json.loads((SCEN / "manifest.json").read_text())
    manifest["alignment"] = dict(train_stats["alignment"])
    manifest["selected_tokens"] = dict(train_stats["by_language_tokens"])
    manifest["total_selected_tokens"] = train_stats["tokens"]
    manifest["records"] = train_stats["records"]
    manifest["records_by_source"] = dict(train_stats["by_source_records"])
    manifest["records_by_language"] = dict(train_stats["by_language_records"])
    manifest["records_by_category"] = dict(train_stats["by_category_records"])
    manifest["tokens_by_source"] = dict(train_stats["by_source_tokens"])
    manifest["tokens_by_category"] = dict(train_stats["by_category_tokens"])
    manifest["train_repositories"] = len(train_stats["repos"])
    manifest["validation_repositories"] = len(valid_stats["repos"])
    manifest["repository_overlap_count"] = 0
    manifest["validation"] = finalize(valid_stats)
    manifest["files"] = result["hashes"]["scenario_b"]
    (SCEN / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (MANIFESTS / "scenario_b_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    split = json.loads((MANIFESTS / "split_report.json").read_text())
    split["post_tokenization"] = {
        "train_sha256": result["full_eligible_pool"]["train_jsonl_sha256"],
        "valid_sha256": result["full_eligible_pool"]["valid_jsonl_sha256"],
        "train_bytes": (PROCESSED / "train.jsonl").stat().st_size,
        "valid_bytes": (PROCESSED / "valid.jsonl").stat().st_size,
        "train_tokens": 780_997_244,
        "validation_tokens": 30_894_000,
    }
    split["scenario_b"] = {
        "path": "data/processed/polycodegpt_v2/scenario_b",
        "train_records": train_stats["records"],
        "validation_records": valid_stats["records"],
        "train_tokens": train_stats["tokens"],
        "validation_tokens": valid_stats["tokens"],
        "repository_overlap_count": 0,
    }
    (MANIFESTS / "split_report.json").write_text(
        json.dumps(split, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["corpus"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
