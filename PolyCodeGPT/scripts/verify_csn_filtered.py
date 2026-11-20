"""Verify a filtered CodeSearchNet output directory against the raw ZIP.

For every output line: JSON validity, record fields, provenance completeness,
license category, repository membership in the permissive set, partition
consistency, and an identity hash. Then independently re-derives the accepted
set straight from the raw archive and compares both directions.

Usage:
    python scripts/verify_csn_filtered.py --language javascript
    python scripts/verify_csn_filtered.py --language python
"""

import argparse
import csv
import gzip
import hashlib
import json
import os
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "data" / "manifests" / "codesearchnet_license_repo_summary.csv"

LANGS = {
    "python": {
        "archive": ROOT / "data" / "raw" / "codesearchnet_zenodo_python" / "python.zip",
        "out_dir": ROOT / "data" / "processed" / "codesearchnet_python_filtered",
        "expected": {"train": 290507, "valid": 14671, "test": 15337},
        "report": ROOT / "data" / "manifests" / "codesearchnet_python_filtered_verification.json",
    },
    "javascript": {
        "archive": ROOT / "data" / "raw" / "codesearchnet_zenodo_javascript" / "javascript.zip",
        "out_dir": ROOT / "data" / "processed" / "codesearchnet_javascript_filtered",
        "expected": {"train": 102169, "valid": 6294, "test": 5699},
        "report": ROOT / "data" / "manifests" / "codesearchnet_javascript_filtered_verification.json",
    },
}

REQUIRED_PROV = {
    "source_dataset", "source_doi", "source_archive_version", "source_repository",
    "source_path", "source_url", "source_partition", "license_category",
    "license_filter_basis",
}
RECORD_KEYS = {
    "repo", "path", "func_name", "original_string", "language", "code",
    "code_tokens", "docstring", "docstring_tokens", "sha", "url", "partition",
}
PARTS = ("train", "valid", "test")


def rec_key(repo, path, func_name, url, code):
    h = hashlib.sha256()
    for part in (repo or "", path or "", func_name or "", url or ""):
        h.update(part.encode("utf-8", "surrogatepass"))
        h.update(b"\x00")
    h.update(hashlib.sha256(code.encode("utf-8", "surrogatepass")).digest())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", required=True, choices=sorted(LANGS))
    args = ap.parse_args()
    cfg = LANGS[args.language]
    archive, out_dir, expected = cfg["archive"], cfg["out_dir"], cfg["expected"]
    t0 = time.perf_counter()

    permissive = set()
    with open(SUMMARY, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["programming_language"] == args.language and r["repository_category"] == "CLEAR_PERMISSIVE_CANDIDATE":
                permissive.add(r["repository"])
    print(f"{args.language}: permissive repositories in summary: {len(permissive)}")

    results, key_sets, repos_by_part, all_paths = {}, {}, {}, set()
    for part in PARTS:
        p = out_dir / f"{part}.jsonl"
        if not p.exists():
            print(f"MISSING FILE: {p}")
            sys.exit(2)
        h = hashlib.sha256()
        n = 0
        bad_json = bad_prov = bad_cat = bad_repo = bad_partition = bad_keys = dup_keys = 0
        repos, paths, keys = set(), set(), set()
        with open(p, "rb") as fh:
            for line in fh:
                h.update(line)
                n += 1
                try:
                    rec = json.loads(line)
                except Exception:
                    bad_json += 1
                    continue
                if not RECORD_KEYS.issubset(rec):
                    bad_keys += 1
                prov = rec.get("provenance")
                if not isinstance(prov, dict) or not REQUIRED_PROV.issubset(prov):
                    bad_prov += 1
                    continue
                if prov["license_category"] != "CLEAR_PERMISSIVE_CANDIDATE":
                    bad_cat += 1
                repo = prov["source_repository"]
                if repo not in permissive:
                    bad_repo += 1
                else:
                    repos.add(repo)
                if prov["source_partition"] != part:
                    bad_partition += 1
                paths.add(prov["source_path"])
                k = rec_key(repo, rec.get("path"), rec.get("func_name"), rec.get("url"), rec.get("code") or "")
                if k in keys:
                    dup_keys += 1
                keys.add(k)
        all_paths |= paths
        results[part] = {
            "file": str(p).replace("\\", "/"),
            "records": n,
            "expected": expected[part],
            "count_matches": n == expected[part],
            "unique_repositories": len(repos),
            "unique_source_paths": len(paths),
            "sha256": h.hexdigest(),
            "file_size_bytes": os.path.getsize(p),
            "invalid_json": bad_json,
            "records_missing_record_fields": bad_keys,
            "records_missing_provenance": bad_prov,
            "records_wrong_license_category": bad_cat,
            "records_repo_not_permissive_in_summary": bad_repo,
            "records_partition_mismatch": bad_partition,
            "duplicate_record_identities": dup_keys,
        }
        key_sets[part], repos_by_part[part] = keys, repos
        print(f"{part}: {n} records, {len(repos)} repos, sha256 {h.hexdigest()[:16]}")

    overlaps = {
        f"{a}_vs_{b}": len(repos_by_part[a] & repos_by_part[b])
        for a, b in (("train", "valid"), ("train", "test"), ("valid", "test"))
    }

    z = zipfile.ZipFile(archive)
    raw_accepted = {p: 0 for p in PARTS}
    matched = {p: 0 for p in PARTS}
    raw_keys = {p: set() for p in PARTS}
    for member in z.namelist():
        if not member.endswith(".jsonl.gz"):
            continue
        tokens = member.split("/")
        part = tokens[tokens.index("jsonl") + 1]
        with z.open(member) as raw, gzip.GzipFile(fileobj=raw) as gz:
            for line in gz:
                rec = json.loads(line)
                if rec.get("repo") not in permissive:
                    continue
                raw_accepted[part] += 1
                k = rec_key(rec["repo"], rec.get("path"), rec.get("func_name"), rec.get("url"), rec.get("code") or "")
                raw_keys[part].add(k)
                if k in key_sets[part]:
                    matched[part] += 1
    z.close()

    recount = {}
    for part in PARTS:
        foreign = key_sets[part] - raw_keys[part]
        recount[part] = {
            "raw_records_from_permissive_repos": raw_accepted[part],
            "matched_in_output": matched[part],
            "output_records_not_derivable_from_raw": len(foreign),
            "full_match": (
                raw_accepted[part] == results[part]["records"]
                and matched[part] == results[part]["records"]
                and not foreign
            ),
        }
        print(f"{part} recount: raw-accepted={raw_accepted[part]} matched={matched[part]} foreign={len(foreign)}")

    report = {
        "language": args.language,
        "verified_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "partitions": results,
        "totals": {
            "records": sum(r["records"] for r in results.values()),
            "expected": sum(expected.values()),
            "count_matches": sum(r["records"] for r in results.values()) == sum(expected.values()),
            "unique_repositories": len(set().union(*repos_by_part.values())),
            "unique_source_paths": len(all_paths),
        },
        "partition_overlaps": overlaps,
        "raw_archive_recount": recount,
        "elapsed_s": time.perf_counter() - t0,
    }
    report["checks_passed"] = all(
        [
            all(r["count_matches"] for r in results.values()),
            all(r["invalid_json"] == 0 for r in results.values()),
            all(r["records_missing_provenance"] == 0 for r in results.values()),
            all(r["records_missing_record_fields"] == 0 for r in results.values()),
            all(r["records_wrong_license_category"] == 0 for r in results.values()),
            all(r["records_repo_not_permissive_in_summary"] == 0 for r in results.values()),
            all(r["records_partition_mismatch"] == 0 for r in results.values()),
            all(v == 0 for v in overlaps.values()),
            all(v["full_match"] for v in recount.values()),
        ]
    )
    with open(cfg["report"], "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: report[k] for k in ("totals", "partition_overlaps", "checks_passed")}, indent=2))
    print(f"verification written to {cfg['report']}")
    print(f"elapsed {report['elapsed_s']:.1f}s")
    sys.exit(0 if report["checks_passed"] else 1)


if __name__ == "__main__":
    main()
