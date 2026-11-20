"""Phase 2E validation: independent checks on the combined corpus, plus a
manifest written to data/manifests/polycodegpt_corpus_manifest.csv.

Checks:
  * combined count == sum of the two filtered datasets
  * every record parses, has all provenance fields, is CLEAR_PERMISSIVE_CANDIDATE
  * every repository maps to exactly one global partition
  * no repository overlap between final partitions
  * original_partition preserved and consistent with the source dataset
  * input datasets unchanged (record counts + file SHA-256 vs recorded values)
  * output hashes/sizes match the manifest
"""

import csv
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "processed" / "polycodegpt_corpus"
SOURCES = {
    "python": ROOT / "data" / "processed" / "codesearchnet_python_filtered",
    "javascript": ROOT / "data" / "processed" / "codesearchnet_javascript_filtered",
}
PARTS = ("train", "valid", "test")
EXPECTED_SOURCE_COUNTS = {"python": 320515, "javascript": 114162}
# SHA-256 of the source partitions as recorded in the earlier phases
EXPECTED_SOURCE_HASHES = {
    ("python", "train"): "2c23d07f9fb4e2ab0a19a93804e43e9ab462b9b7ef97bba4d454d073a2e4c881",
    ("python", "valid"): "3bcb20af27eb57735e4b1be175540df14754fe73b1413039f1eb72a8ca402333",
    ("python", "test"): "94eaa05e672ae5a60877001893767496ad4f0d0f29315bc0ed5b4a7597934f1e",
}
REQUIRED_FIELDS = {
    "repo", "path", "func_name", "original_string", "language", "code", "docstring",
    "sha", "url", "partition", "record_id", "code_sha256", "original_partition",
    "global_partition", "source_dataset", "source_doi", "source_archive_version",
    "license_category", "license_filter_basis", "provenance",
}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    t0 = time.perf_counter()
    problems = []

    # ---- source datasets untouched ----
    src_counts = {}
    for lang, d in SOURCES.items():
        total = 0
        for part in PARTS:
            p = d / f"{part}.jsonl"
            total += sum(1 for _ in open(p, encoding="utf-8"))
            if (lang, part) in EXPECTED_SOURCE_HASHES:
                actual = sha256_file(p)
                if actual != EXPECTED_SOURCE_HASHES[(lang, part)]:
                    problems.append(f"{lang}/{part} hash drifted: {actual}")
        src_counts[lang] = total
        if total != EXPECTED_SOURCE_COUNTS[lang]:
            problems.append(f"{lang} record count {total} != {EXPECTED_SOURCE_COUNTS[lang]}")
    print("source counts:", src_counts)

    # ---- corpus checks ----
    repo_partitions = defaultdict(set)
    counts = {}
    lang_counts = defaultdict(int)
    part_lang = defaultdict(lambda: defaultdict(int))
    missing_fields = set()
    bad_license = 0
    bad_provenance = 0
    bad_partition = 0
    bad_orig = 0
    records = 0
    for part in PARTS:
        p = CORPUS / f"{part}.jsonl"
        n = 0
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                n += 1
                rec = json.loads(line)
                miss = REQUIRED_FIELDS - rec.keys()
                if miss:
                    missing_fields |= miss
                if rec["license_category"] != "CLEAR_PERMISSIVE_CANDIDATE":
                    bad_license += 1
                prov = rec.get("provenance") or {}
                if not prov.get("source_repository") or not prov.get("source_doi"):
                    bad_provenance += 1
                if rec["global_partition"] != part:
                    bad_partition += 1
                lang = "python" if "Python" in rec["source_dataset"] else "javascript"
                if lang == "python" and prov.get("source_dataset") != "CodeSearchNet Python":
                    bad_orig += 1
                if rec["original_partition"] not in PARTS:
                    bad_orig += 1
                repo_partitions[rec["repo"].lower()].add(part)
                lang_counts[lang] += 1
                part_lang[part][lang] += 1
        counts[part] = n
        records += n
    print("corpus counts:", counts, "total", records, "languages", dict(lang_counts))

    multi = {r: sorted(v) for r, v in repo_partitions.items() if len(v) > 1}
    overlaps = {
        f"{a}_vs_{b}": sum(1 for v in repo_partitions.values() if a in v and b in v)
        for a, b in (("train", "valid"), ("train", "test"), ("valid", "test"))
    }

    if records != sum(EXPECTED_SOURCE_COUNTS.values()):
        problems.append(f"combined {records} != sum of sources {sum(EXPECTED_SOURCE_COUNTS.values())}")
    if missing_fields:
        problems.append(f"missing fields: {sorted(missing_fields)}")
    if bad_license:
        problems.append(f"{bad_license} records with non-permissive licence category")
    if bad_provenance:
        problems.append(f"{bad_provenance} records with incomplete provenance")
    if bad_partition:
        problems.append(f"{bad_partition} records in wrong partition file")
    if bad_orig:
        problems.append(f"{bad_orig} records with inconsistent original_partition")
    if multi:
        problems.append(f"repositories spanning partitions: {multi}")
    if any(overlaps.values()):
        problems.append(f"partition overlap: {overlaps}")

    # ---- manifest ----
    rows = []
    for part in PARTS:
        p = CORPUS / f"{part}.jsonl"
        rows.append({
            "output_file": f"data/processed/polycodegpt_corpus/{part}.jsonl",
            "partition": part,
            "record_count": counts[part],
            "python_records": part_lang[part]["python"],
            "javascript_records": part_lang[part]["javascript"],
            "unique_repository_count": sum(1 for v in repo_partitions.values() if part in v),
            "sha256": sha256_file(p),
            "file_size_bytes": os.path.getsize(p),
            "source_datasets": "CodeSearchNet Python + CodeSearchNet JavaScript",
            "source_doi": "10.5281/zenodo.7908468",
            "license_filter": "repository_category == CLEAR_PERMISSIVE_CANDIDATE",
            "partition_rule": "TEST > VALID > TRAIN per normalized repository",
            "verified_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
    man = ROOT / "data" / "manifests" / "polycodegpt_corpus_manifest.csv"
    with open(man, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- manifest self-check ----
    for r in csv.DictReader(open(man, newline="", encoding="utf-8")):
        p = ROOT / r["output_file"]
        if sha256_file(p) != r["sha256"]:
            problems.append(f"manifest hash mismatch: {r['output_file']}")
        if os.path.getsize(p) != int(r["file_size_bytes"]):
            problems.append(f"manifest size mismatch: {r['output_file']}")

    result = {
        "checked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_counts": src_counts,
        "partition_counts": counts,
        "total_records": records,
        "language_totals": dict(lang_counts),
        "partition_language_counts": {p: dict(v) for p, v in part_lang.items()},
        "repositories": len(repo_partitions),
        "repositories_spanning_partitions": multi,
        "partition_overlaps": overlaps,
        "missing_fields": sorted(missing_fields),
        "non_permissive_records": bad_license,
        "incomplete_provenance": bad_provenance,
        "wrong_partition_records": bad_partition,
        "inconsistent_original_partition": bad_orig,
        "problems": problems,
        "checks_passed": not problems,
        "elapsed_s": time.perf_counter() - t0,
    }
    out = ROOT / "data" / "manifests" / "polycodegpt_corpus_validation.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: result[k] for k in ("total_records", "partition_counts", "repositories",
                                             "partition_overlaps", "checks_passed", "problems")}, indent=2))
    print(f"elapsed {result['elapsed_s']:.1f}s")
    sys.exit(0 if result["checks_passed"] else 1)


if __name__ == "__main__":
    main()
