"""Phase 12A Parts B/C — JavaScript training data audit and curation.

Read-only with respect to the original corpus and tokenizer. Streams the JS
training split, validates syntax with the Node front end, applies the rules in
``rules.py``, removes duplicates, and writes a curated candidate dataset under
``data/curated/js_v1/`` plus a filtering manifest.

    python experiments/phase12a/audit_js.py [--dry-run]

``--dry-run`` runs the audit and writes the audit report but does not write the
curated dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "phase12a"))

from polycodegpt.data import load_tokenizer  # noqa: E402
from polycodegpt.data.tokenizer_io import TOKENIZER_PATH  # noqa: E402
from polycodegpt.tokenization.serializer import serialize_javascript_record  # noqa: E402
from polycodegpt.data.integrity import sha256_file  # noqa: E402

import rules  # noqa: E402

JS_TRAIN = ROOT / "data" / "processed" / "polycodegpt_corpus" / "train.jsonl"
ARTIFACTS = ROOT / "experiments" / "phase12a" / "artifacts"
CURATED_DIR = ROOT / "data" / "curated" / "js_v1"
AUDIT_RECORDS = ARTIFACTS / "js_audit_records.jsonl"
PARSE_DUMP = ARTIFACTS / "parse_output.jsonl"
AUDIT_JSON = ROOT / "reports" / "phase12a_js_audit.json"
AUDIT_MD = ROOT / "reports" / "phase12a_js_audit.md"
CURATED_MD = ROOT / "reports" / "phase12a_curated_dataset.md"

NODE = "node"
PARSE_SCRIPT = ROOT / "experiments" / "phase12a" / "parse_batch.mjs"

MINHASH_SEED = 20261117
A = np.array([1000003 * (i + 1) for i in range(rules.PERMUTATIONS)], dtype=np.uint64)
B = np.array([(0x9E3779B97F4A7C15 * (i + 1)) & 0xFFFFFFFFFFFFFFFF for i in range(rules.PERMUTATIONS)], dtype=np.uint64)
MASK = np.uint64(0xFFFFFFFFFFFFFFFF)


def stream_js_records():
    if not JS_TRAIN.is_file():
        raise FileNotFoundError(JS_TRAIN)
    with JS_TRAIN.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("language") != "javascript":
                continue
            if record.get("global_partition") not in (None, "train"):
                continue
            yield record


def minhash(code_tokens: list[str]) -> np.ndarray | None:
    """15 minhash values over 5-token shingles. None if too short."""
    if len(code_tokens) < rules.SHINGLE_SIZE:
        return None
    ids = np.array([hash(tok) & 0xFFFFFFFF for tok in code_tokens], dtype=np.uint64)
    n = len(ids) - rules.SHINGLE_SIZE + 1
    shingles = np.empty(n, dtype=np.uint64)
    for k in range(rules.SHINGLE_SIZE):
        shingles = shingles * np.uint64(1000003) + ids[k : k + n]
    # permuted minhash
    matrix = (shingles[:, None] * A[None, :] + B[None, :]) & MASK
    return matrix.min(axis=0)


def run_parser(input_path: Path, output_path: Path) -> None:
    with input_path.open("rb") as stdin, output_path.open("wb") as stdout:
        subprocess.run(
            [NODE, "--experimental-vm-modules", str(PARSE_SCRIPT)],
            stdin=stdin,
            stdout=stdout,
            check=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    CURATED_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(TOKENIZER_PATH)

    # ---------------------------------------------------------------- pass 1
    # Collect metadata + write the parser input. Stream the corpus so peak memory
    # stays bounded by the metadata dictionary.
    print("pass 1: reading JS training split")
    metadata: dict[str, dict[str, Any]] = {}
    input_lines: list[str] = []
    input_path = ARTIFACTS / "_parser_input.jsonl"
    input_path.write_text("", encoding="utf-8")
    count = 0
    with input_path.open("w", encoding="utf-8") as fh:
        for record in stream_js_records():
            rid = str(record["record_id"])
            metadata[rid] = {
                "record": record,
                "repo": record.get("repo", ""),
                "func_name": record.get("func_name", ""),
                "docstring": record.get("docstring", ""),
            }
            fh.write(json.dumps({"id": rid, "code": record.get("code", "")}) + "\n")
            count += 1
    print(f"  records: {count}")

    if count != 102169:
        print(f"WARNING: expected 102169 JS training records, found {count}")

    # Run the parser (compiles only, never executes).
    print("running V8 parser front end")
    run_parser(input_path, PARSE_DUMP)
    input_path.unlink(missing_ok=True)

    parse_by_id: dict[str, dict[str, Any]] = {}
    with PARSE_DUMP.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            parse_by_id[obj["id"]] = obj

    # ---------------------------------------------------------------- features
    print("computing features")
    all_records: list[dict[str, Any]] = []
    for rid, meta in metadata.items():
        record = meta["record"]
        parsed = parse_by_id.get(rid, {})
        doc_category, doc_quality = rules.classify_description(meta["docstring"])
        text = serialize_javascript_record(record)
        token_ids = tokenizer.encode(text, add_special_tokens=False).ids
        code_tokens = int(record.get("code_tokens") and len(record["code_tokens"]) or 0)
        if not code_tokens:
            code_tokens = len(record.get("code", "").split())
        features = {
            "record_id": rid,
            "repo": meta["repo"],
            "func_name": meta["func_name"],
            "parse": parsed.get("parse", "invalid"),
            "kind": parsed.get("kind", "other"),
            "declared_name": parsed.get("name"),
            "complete": bool(parsed.get("complete", False)),
            "body_kind": parsed.get("body_kind"),
            "has_require": bool(parsed.get("has_require", False)),
            "has_import_export": bool(parsed.get("has_import_export", False)),
            "uses_this": bool(parsed.get("uses_this", False)),
            "uses_host_global": bool(parsed.get("uses_host_global", False)),
            "console_calls": int(parsed.get("console_calls", 0)),
            "description_category": doc_category,
            "description_quality": doc_quality,
            "code_tokens": code_tokens,
            "serialized_tokens": len(token_ids),
            "description_chars": len(meta["docstring"]),
            "minhash": None,
        }
        reasons, flags = rules.evaluate(features)
        features["reasons"] = reasons
        features["flags"] = flags
        features["exact_hash"] = rules.exact_hash(record.get("code", ""))
        mh = minhash(list(record.get("code_tokens") or []))
        features["minhash"] = mh.tolist() if mh is not None else None
        all_records.append(features)

    # ---------------------------------------------------------------- dedup
    # Order by record_id for deterministic "keep first" behaviour.
    all_records.sort(key=lambda r: r["record_id"])
    kept_exact: set[str] = set()
    kept_exact_hashes: dict[str, str] = {}
    kept_bands: list[dict[str, tuple]] = []
    for band_idx in range(len(rules.BANDS)):
        kept_bands.append({})

    for r in all_records:
        if r["reasons"]:
            continue
        eh = r["exact_hash"]
        if eh in kept_exact_hashes:
            r["reasons"] = ["duplicate"]
            r["duplicate_of"] = kept_exact_hashes[eh]
            continue
        # near-duplicate via LSH bands
        mh = r["minhash"]
        near_of = None
        if mh is not None:
            arr = np.array(mh, dtype=np.uint64)
            for band_idx, cols in enumerate(rules.BANDS):
                key = tuple(int(arr[c]) for c in cols)
                if key in kept_bands[band_idx]:
                    near_of = kept_bands[band_idx][key]
                    break
        if near_of is not None:
            r["reasons"] = ["near_duplicate"]
            r["near_duplicate_of"] = near_of
            continue
        # keep
        kept_exact_hashes[eh] = r["record_id"]
        if mh is not None:
            arr = np.array(mh, dtype=np.uint64)
            for band_idx, cols in enumerate(rules.BANDS):
                key = tuple(int(arr[c]) for c in cols)
                kept_bands[band_idx][key] = r["record_id"]
        r["reasons"] = []

    # ---------------------------------------------------------------- write audit
    print("writing audit records")
    with AUDIT_RECORDS.open("w", encoding="utf-8") as fh:
        for r in all_records:
            slim = {k: v for k, v in r.items() if k not in ("record", "minhash")}
            fh.write(json.dumps(slim) + "\n")

    # ---------------------------------------------------------------- aggregates
    total = len(all_records)
    rejected = [r for r in all_records if r["reasons"]]
    accepted = [r for r in all_records if not r["reasons"]]

    reason_counter: Counter = Counter()
    for r in rejected:
        reason_counter[r["reasons"][0]] += 1

    kind_counter = Counter(r["kind"] for r in all_records)
    parse_counter = Counter(r["parse"] for r in all_records)
    desc_cat = Counter(r["description_category"] for r in all_records)
    desc_quality = Counter(r["description_quality"] for r in all_records)
    repo_counter_accepted = Counter(r["repo"] for r in accepted)
    repo_counter_all = Counter(r["repo"] for r in all_records)

    def quantiles(values: list[int]) -> dict[str, float]:
        if not values:
            return {}
        arr = np.array(values, dtype=float)
        return {
            "min": float(np.min(arr)),
            "p25": float(np.percentile(arr, 25)),
            "median": float(np.median(arr)),
            "p75": float(np.percentile(arr, 75)),
            "p95": float(np.percentile(arr, 95)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
        }

    all_serial = [r["serialized_tokens"] for r in all_records]
    acc_serial = [r["serialized_tokens"] for r in accepted]
    all_code = [r["code_tokens"] for r in all_records]
    acc_code = [r["code_tokens"] for r in accepted]

    # language/repo distribution for accepted
    distinct_repos_accepted = len(repo_counter_accepted)
    duplicate_exact = sum(1 for r in rejected if r["reasons"] and r["reasons"][0] == "duplicate")
    duplicate_near = sum(1 for r in rejected if r["reasons"] and r["reasons"][0] == "near_duplicate")

    audit = {
        "phase": "12A",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": JS_TRAIN.relative_to(ROOT).as_posix(),
        "original_record_count": total,
        "eligible_record_count": len(accepted),
        "rejected_record_count": len(rejected),
        "rejection_reasons": {reason: reason_counter.get(reason, 0) for reason in rules.REASONS},
        "parse_status": dict(parse_counter),
        "structural_kinds": dict(kind_counter),
        "description_categories": dict(desc_cat),
        "description_qualities": dict(desc_quality),
        "duplicate_exact": duplicate_exact,
        "duplicate_near": duplicate_near,
        "distinct_repos_original": len(repo_counter_all),
        "distinct_repos_curated": distinct_repos_accepted,
        "token_lengths": {
            "all_serialized": quantiles(all_serial),
            "curated_serialized": quantiles(acc_serial),
            "all_code_tokens": quantiles(all_code),
            "curated_code_tokens": quantiles(acc_code),
        },
        "top_repos_curated": repo_counter_accepted.most_common(15),
        "rules": {r: rules.REASON_TEXT[r] for r in rules.REASONS},
        "thresholds": {
            "MIN_DESCRIPTION_WORDS": rules.MIN_DESCRIPTION_WORDS,
            "MIN_DESCRIPTION_CHARS": rules.MIN_DESCRIPTION_CHARS,
            "MAX_DESCRIPTION_CHARS": rules.MAX_DESCRIPTION_CHARS,
            "MIN_ALPHA_RATIO": rules.MIN_ALPHA_RATIO,
            "MIN_CODE_TOKENS": rules.MIN_CODE_TOKENS,
            "MAX_CODE_TOKENS": rules.MAX_CODE_TOKENS,
            "MAX_SERIALIZED_TOKENS": rules.MAX_SERIALIZED_TOKENS,
            "ALLOWED_KINDS": list(rules.ALLOWED_KINDS),
            "NEAR_DUP": "15 minhash values over 5-token shingles, 3 bands of 5",
        },
    }
    AUDIT_JSON.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    # ---------------------------------------------------------------- curated
    accepted_ids = {r["record_id"] for r in accepted}
    curated_records = []
    if not args.dry_run:
        print("pass 2: writing curated dataset")
        with JS_TRAIN.open(encoding="utf-8") as fh, (CURATED_DIR / "train.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as out:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if str(record.get("record_id")) not in accepted_ids:
                    continue
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                curated_records.append(record)

    # manifest files
    curated_path = CURATED_DIR / "train.jsonl"
    curated_sha = sha256_file(curated_path) if curated_path.is_file() else None
    curated_count = sum(1 for _ in curated_path.open(encoding="utf-8")) if curated_path.is_file() else 0

    manifest = {
        "phase": "12A",
        "dataset": "js_v1",
        "source": JS_TRAIN.relative_to(ROOT).as_posix(),
        "original_record_count": total,
        "eligible_record_count": len(accepted),
        "rejected_record_count": len(rejected),
        "rejection_reasons": audit["rejection_reasons"],
        "output_file": curated_path.relative_to(ROOT).as_posix() if curated_path.is_file() else None,
        "output_sha256": curated_sha,
        "output_record_count": curated_count,
        "distinct_repos_curated": distinct_repos_accepted,
        "token_lengths_curated_serialized": quantiles if False else audit["token_lengths"]["curated_serialized"],
        "generated_utc": audit["generated_utc"],
    }
    (CURATED_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    # CSV manifest of rejected records (auditable exclusions).
    with (CURATED_DIR / "rejected.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["record_id", "repo", "primary_reason", "all_reasons"])
        for r in rejected:
            writer.writerow([r["record_id"], r["repo"], r["reasons"][0], ";".join(r["reasons"])])

    write_audit_markdown(audit, accepted, rejected)
    if not args.dry_run:
        write_curated_markdown(audit, manifest)

    print(f"original={total} curated={len(accepted)} rejected={len(rejected)}")
    print(f"wrote {AUDIT_JSON.name}, {AUDIT_MD.name}")
    if not args.dry_run:
        print(f"wrote {curated_path.relative_to(ROOT)} ({curated_count} records, sha {curated_sha[:12]}...)")
    return 0


def quantiles(values):
    if not values:
        return {}
    arr = np.array(values, dtype=float)
    return {
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def write_audit_markdown(audit: dict, accepted: list, rejected: list) -> None:
    lines = [
        "# Phase 12A — JavaScript Training Data Audit",
        "",
        f"Source: `{audit['source']}` (training split only; validation and test splits were not read).",
        "",
        f"- Original records: **{audit['original_record_count']:,}**",
        f"- Eligible (curated): **{audit['eligible_record_count']:,}** "
        f"({100 * audit['eligible_record_count'] / audit['original_record_count']:.1f}%)",
        f"- Rejected: **{audit['rejected_record_count']:,}**",
        "",
        "## Rejection reasons",
        "",
        "| reason | count | meaning |",
        "|---|---:|---|",
    ]
    for reason in rules.REASONS:
        lines.append(
            f"| `{reason}` | {audit['rejection_reasons'][reason]:,} | {rules.REASON_TEXT[reason]} |"
        )
    lines += [
        "",
        "## Parse status (V8 compile, no execution)",
        "",
        "| status | count |",
        "|---|---:|",
    ]
    for k, v in audit["parse_status"].items():
        lines.append(f"| {k} | {v:,} |")
    lines += [
        "",
        "## Structural kinds",
        "",
        "| kind | count |",
        "|---|---:|",
    ]
    for k, v in audit["structural_kinds"].items():
        lines.append(f"| {k} | {v:,} |")
    lines += [
        "",
        "## Description categories",
        "",
        "| category | count |",
        "|---|---:|",
    ]
    for k, v in audit["description_categories"].items():
        lines.append(f"| {k} | {v:,} |")
    lines += [
        "",
        "## Description quality",
        "",
        "| quality | count |",
        "|---|---:|",
    ]
    for k, v in audit["description_qualities"].items():
        lines.append(f"| {k} | {v:,} |")
    lines += [
        "",
        "## Duplicates",
        "",
        f"- Exact (normalized whitespace): {audit['duplicate_exact']:,}",
        f"- Near (minhash LSH): {audit['duplicate_near']:,}",
        "",
        "## Token lengths",
        "",
        "| set | min | p25 | median | p75 | p95 | max | mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, key in (
        ("all serialized", "all_serialized"),
        ("curated serialized", "curated_serialized"),
        ("all code tokens", "all_code_tokens"),
        ("curated code tokens", "curated_code_tokens"),
    ):
        q = audit["token_lengths"][key]
        lines.append(
            f"| {label} | {q['min']:.0f} | {q['p25']:.0f} | {q['median']:.0f} | "
            f"{q['p75']:.0f} | {q['p95']:.0f} | {q['max']:.0f} | {q['mean']:.0f} |"
        )
    lines += [
        "",
        "## Repositories",
        "",
        f"- Distinct repos (original): {audit['distinct_repos_original']:,}",
        f"- Distinct repos (curated): {audit['distinct_repos_curated']:,}",
        "",
        "Top curated repos:",
        "",
        "| repo | records |",
        "|---|---:|",
    ]
    for repo, n in audit["top_repos_curated"]:
        lines.append(f"| {repo} | {n} |")
    lines += ["", "Filtering rules and thresholds are in `experiments/phase12a/rules.py`.", ""]
    AUDIT_MD.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def write_curated_markdown(audit: dict, manifest: dict) -> None:
    lines = [
        "# Phase 12A — Curated JavaScript Candidate Dataset",
        "",
        f"Output: `{manifest['output_file']}`",
        f"SHA-256: `{manifest['output_sha256']}`",
        f"Records: {manifest['output_record_count']:,}",
        "",
        f"From {audit['original_record_count']:,} original JS training records, "
        f"{audit['eligible_record_count']:,} survived "
        f"({100 * audit['eligible_record_count'] / audit['original_record_count']:.1f}%).",
        "",
        "Every original field is preserved verbatim; no record was rewritten.",
        "Exclusions are listed in `data/curated/js_v1/rejected.csv` (one row per rejected record).",
        "",
        "## Filtering rules applied (in order)",
        "",
    ]
    for reason in rules.REASONS:
        lines.append(f"1. `{reason}` — {rules.REASON_TEXT[reason]}")
    lines += [
        "",
        "Thresholds: " + ", ".join(
            f"{k}={v}" for k, v in audit["thresholds"].items() if not isinstance(v, list)
        ),
        "",
    ]
    (ROOT / "reports" / "phase12a_curated_dataset.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    sys.exit(main())
