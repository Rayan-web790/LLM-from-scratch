"""Phase 18 staged corpus builder: extract, deduplicate, split, report."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator

from .constants import LICENSE_ALLOWLIST, SCHEMA_VERSION
from .dedup import DedupStats, deduplicate
from .language import alignment_check, normalize_repo, repo_group
from .quality import (
    canonicalize,
    code_fingerprints,
    code_rejection,
    documentation_fingerprints,
    documentation_rejection,
)
from .schema import (
    build_code_record,
    build_documentation_record,
    assert_no_forbidden_input,
    validate_record,
)
from .sources import iter_all_sources
from .split import partition_for

PROJECT_ROOT = Path(__file__).resolve().parents[3]
V2_ROOT = PROJECT_ROOT / "data" / "interim" / "v2"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "polycodegpt_v2"
MANIFEST_ROOT = PROJECT_ROOT / "data" / "manifests" / "v2"
CANDIDATES = V2_ROOT / "candidates.jsonl"
REJECTED = V2_ROOT / "rejected.jsonl"
TRAIN = PROCESSED_ROOT / "train.jsonl"
VALID = PROCESSED_ROOT / "valid.jsonl"
STATS = MANIFEST_ROOT / "stage_stats.json"
DEDUP_REPORT = MANIFEST_ROOT / "dedup_report.json"
SPLIT_REPORT = MANIFEST_ROOT / "split_report.json"
CATEGORY_REPORT = MANIFEST_ROOT / "category_report.csv"


def _open_writer(path: Path, mode="w"):
    path.parent.mkdir(parents=True, exist_ok=True)
    return open(path, mode, encoding="utf-8", newline="\n")


def _write_jsonl(fh, rec: dict) -> None:
    fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")


def _reject(fh, item: dict, reason: str, detail: str = "") -> None:
    rec = {
        "source_id": item.get("source_id", ""),
        "source_snapshot": item.get("source_snapshot", ""),
        "repo": item.get("repo", ""),
        "path": item.get("path", ""),
        "language": item.get("language", ""),
        "content_kind": item.get("content_kind", ""),
        "license_id": item.get("license_id", ""),
        "reason": reason,
        "detail": detail[:300],
        "raw_sha256": item.get("raw_sha256", ""),
    }
    _write_jsonl(fh, rec)


def extract(
    *,
    shard_limit: int | None = None,
    shard_offset: int = 0,
    max_candidates: int | None = None,
    append: bool = False,
    include_base: bool = True,
) -> dict:
    """Apply preflight/license/syntax/quality gates and emit candidates."""
    V2_ROOT.mkdir(parents=True, exist_ok=True)
    raw_counts: Counter = Counter()
    accepted: Counter = Counter()
    rejected: Counter = Counter()
    chars: Counter = Counter()
    license_counts: Counter = Counter()
    alignment: Counter = Counter()
    started = time.perf_counter()
    n_candidates = 0
    mode = "a" if append else "w"
    CANDIDATES.parent.mkdir(parents=True, exist_ok=True)
    REJECTED.parent.mkdir(parents=True, exist_ok=True)
    with open(CANDIDATES, mode, encoding="utf-8", newline="\n") as out, open(
        REJECTED, mode, encoding="utf-8", newline="\n"
    ) as rej:
        if include_base:
            source_iter = iter_all_sources(shard_limit=shard_limit, shard_offset=shard_offset)
        else:
            from .sources import iter_codeparrot

            source_iter = iter_codeparrot(shard_limit=shard_limit, shard_offset=shard_offset)
        for item in source_iter:
            sid = item.get("source_id", "")
            lang = item.get("language", "")
            kind = item.get("content_kind", "")
            raw_counts[f"{sid}|{kind}|{lang}"] += 1
            raw = canonicalize(item.get("raw_content") or "")
            item["raw_sha256"] = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            license_id = (item.get("license_id") or "").strip()
            license_counts[f"{sid}|{license_id}"] += 1

            # License gate: CodeParrot is the only source with a per-file field.
            if sid == "CODEPARROT" and license_id not in LICENSE_ALLOWLIST:
                rejected[f"{sid}|license_not_allowlisted"] += 1
                _reject(rej, item, "license_not_allowlisted", license_id)
                continue

            if kind == "documentation":
                why = documentation_rejection(raw, item.get("path", ""))
                if why:
                    rejected[f"{sid}|{why}"] += 1
                    _reject(rej, item, why)
                    continue
                if sid == "CODEPARROT" and "```" not in raw and "~~~" not in raw:
                    rejected[f"{sid}|docs_without_code_example"] += 1
                    _reject(rej, item, "docs_without_code_example")
                    continue
                exact, norm, sig = documentation_fingerprints(raw)
                rec = build_documentation_record(
                    source_id=sid,
                    source_snapshot=item.get("source_snapshot", ""),
                    repo=item.get("repo", ""),
                    path=item.get("path", ""),
                    license_id=license_id,
                    license_evidence=item.get("license_evidence", ""),
                    source_url=item.get("source_url", ""),
                    text=raw,
                    exact_sha256=exact,
                    normalized_sha256=norm,
                )
                rec["near_signature"] = list(sig)
                rec["alignment"] = "not_applicable"
                validate_record(rec)
                _write_jsonl(out, rec)
                accepted[f"{sid}|documentation|en"] += 1
                chars[f"{sid}|documentation|en"] += len(raw)
                n_candidates += 1
                if max_candidates and n_candidates >= max_candidates:
                    break
                continue

            # Code gate.
            why = code_rejection(lang, raw, item.get("path", ""))
            if why:
                rejected[f"{sid}|{why}"] += 1
                _reject(rej, item, why)
                continue
            docstring = (item.get("docstring") or "").strip()
            task = "module"
            align_status = "not_applicable"
            if docstring:
                ok, align_reason = alignment_check(docstring, raw, lang)
                alignment["checked"] += 1
                if ok:
                    task = "function"
                    align_status = "pass"
                    alignment["pass"] += 1
                else:
                    # Do not fabricate a prompt. Keep valid code as a module
                    # record without the unusable docstring.
                    align_status = f"failed_to_module:{align_reason}"
                    alignment["failed"] += 1
            exact, norm, sig = code_fingerprints(lang, raw)
            rec = build_code_record(
                source_id=sid,
                source_snapshot=item.get("source_snapshot", ""),
                repo=normalize_repo(item.get("repo", "")),
                path=item.get("path", ""),
                license_id=license_id,
                license_evidence=item.get("license_evidence", ""),
                source_url=item.get("source_url", ""),
                origin_partition=item.get("origin_partition", ""),
                language=lang,
                task=task,
                category=item.get("category", ""),
                code=raw,
                imports=item.get("imports", ""),
                docstring=docstring if task == "function" else "",
                signature=item.get("signature", ""),
                alignment=align_status,
                exact_sha256=exact,
                normalized_sha256=norm,
            )
            rec["near_signature"] = list(sig)
            validate_record(rec)
            _write_jsonl(out, rec)
            accepted[f"{sid}|{kind}|{lang}"] += 1
            chars[f"{sid}|{kind}|{lang}"] += len(raw)
            n_candidates += 1
            if max_candidates and n_candidates >= max_candidates:
                break
    stats = {
        "stage": "extract",
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "raw": dict(sorted(raw_counts.items())),
        "accepted": dict(sorted(accepted.items())),
        "rejected": dict(sorted(rejected.items())),
        "accepted_chars": dict(sorted(chars.items())),
        "license_observations": dict(sorted(license_counts.items())),
        "alignment": dict(sorted(alignment.items())),
        "candidates": n_candidates,
        "shard_limit": shard_limit,
        "shard_offset": shard_offset,
        "append": append,
        "include_base": include_base,
        "max_candidates": max_candidates,
    }
    STATS.parent.mkdir(parents=True, exist_ok=True)
    STATS.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return stats


def _read_candidates() -> Iterator[dict]:
    with open(CANDIDATES, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def dedup_and_split() -> dict:
    """Deduplicate globally, then assign repository-isolated partitions."""
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    stats = DedupStats()
    rejected_near = Counter()
    # Documentation is train-only. Any repository contributing documentation
    # must therefore contribute all of its code to train as well; otherwise a
    # docs/code pair from the same repository could straddle the split.
    forced_train_repos: set[str] = set()
    for rec in _read_candidates():
        if rec.get("content_kind") == "documentation":
            forced_train_repos.add(repo_group(rec.get("source_id", ""), rec.get("repo", "")))
    with _open_writer(TRAIN) as train_fh, _open_writer(VALID) as valid_fh, open(REJECTED, "a", encoding="utf-8", newline="\n") as rej:
        def reject(rec, reason, other_id):
            rejected_near[reason] += 1
            _write_jsonl(
                rej,
                {
                    "source_id": rec.get("source_id", ""),
                    "source_snapshot": rec.get("source_snapshot", ""),
                    "repo": rec.get("repo", ""),
                    "path": rec.get("path", ""),
                    "language": rec.get("language", ""),
                    "content_kind": rec.get("content_kind", ""),
                    "license_id": rec.get("license_id", ""),
                    "reason": reason,
                    "detail": f"duplicate_of={other_id}",
                    "raw_sha256": rec.get("code_sha256") or rec.get("text_sha256", ""),
                },
            )

        for rec in deduplicate(_read_candidates(), stats=stats, reject=reject):
            rec.pop("near_signature", None)
            rec["partition"] = partition_for(rec["repo"], rec["content_kind"], forced_train_repos)
            rec["schema_version"] = SCHEMA_VERSION
            fh = valid_fh if rec["partition"] == "validation" else train_fh
            _write_jsonl(fh, rec)

    report = stats.as_dict()
    report["duplicate_rejections"] = dict(sorted(rejected_near.items()))
    DEDUP_REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_final() -> dict:
    """Validate final records, split isolation, category coverage, and hashes."""
    seen_ids = set()
    category: Counter = Counter()
    source: Counter = Counter()
    language: Counter = Counter()
    char_counts: Counter = Counter()
    alignment: Counter = Counter()
    split_records: Counter = Counter()
    split_chars: Counter = Counter()
    repo_partition: dict[str, str] = {}
    repo_sets = {"train": set(), "validation": set()}
    for path, partition in ((TRAIN, "train"), (VALID, "validation")):
        assert_no_forbidden_input(str(path))
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                validate_record(rec)
                if rec["record_id"] in seen_ids:
                    raise RuntimeError(f"duplicate record_id in final corpus: {rec['record_id']}")
                seen_ids.add(rec["record_id"])
                if rec["partition"] != partition:
                    raise RuntimeError(f"record partition mismatch: {rec['record_id']}")
                repo = repo_group(rec["source_id"], rec["repo"])
                previous = repo_partition.get(repo)
                if previous is not None and previous != partition:
                    raise RuntimeError(f"repository leakage: {repo} in {previous} and {partition}")
                repo_partition[repo] = partition
                repo_sets[partition].add(repo)
                split_records[partition] += 1
                split_chars[partition] += len(rec["text"])
                category[f"{rec['language']}|{rec['category']}"] += 1
                source[f"{rec['source_id']}|{rec['language']}"] += 1
                language[rec["language"]] += 1
                char_counts[rec["language"]] += len(rec["text"])
                alignment[rec["alignment"].split(":", 1)[0]] += 1
    overlap = sorted(repo_sets["train"] & repo_sets["validation"])
    split = {
        "validation_percent_policy": 5,
        "repository_overlap": overlap,
        "repository_overlap_count": len(overlap),
        "train_repositories": len(repo_sets["train"]),
        "validation_repositories": len(repo_sets["validation"]),
        "records": dict(split_records),
        "serialized_chars": dict(split_chars),
        "tokens": {},
    }
    if overlap:
        raise RuntimeError(f"repository leakage detected: {overlap[:5]}")
    if split_records.get("validation", 0) == 0:
        raise RuntimeError("validation split is empty")
    with open(CATEGORY_REPORT, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["language", "category", "records"])
        for key, n in sorted(category.items()):
            lang, cat = key.split("|", 1)
            writer.writerow([lang, cat, n])
    result = {
        "records": sum(split_records.values()),
        "records_by_language": dict(sorted(language.items())),
        "records_by_source": dict(sorted(source.items())),
        "categories": dict(sorted(category.items())),
        "alignment": dict(sorted(alignment.items())),
        "serialized_chars": dict(sorted(char_counts.items())),
        "split": split,
        "train_sha256": _sha256_file(TRAIN),
        "valid_sha256": _sha256_file(VALID),
        "train_bytes": TRAIN.stat().st_size,
        "valid_bytes": VALID.stat().st_size,
        "test_partition_read": False,
    }
    SPLIT_REPORT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def run_extract(**kwargs) -> dict:
    return extract(**kwargs)


def run_build(**kwargs) -> dict:
    extract_result = extract(**kwargs)
    dedup_result = dedup_and_split()
    final = audit_final()
    return {"extract": extract_result, "dedup": dedup_result, "final": final}


def print_summary(result: dict) -> None:
    print(json.dumps(result, indent=2, sort_keys=True)[:12000])
