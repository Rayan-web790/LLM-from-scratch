"""Phase 2C: stream CodeSearchNet Python JSONL shards from the Zenodo ZIP,
apply repository-level license filtering, write filtered JSONL + provenance."""

import csv
import gzip
import hashlib
import json
import os
import statistics
import time
try:
    import psutil
    _PSUTIL = True
except Exception:
    _PSUTIL = False
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data" / "raw" / "codesearchnet_zenodo_python" / "python.zip"
SUMMARY = ROOT / "data" / "manifests" / "codesearchnet_license_repo_summary.csv"
OUT_DIR = ROOT / "data" / "processed" / "codesearchnet_python_filtered"

SOURCE_DATASET = "CodeSearchNet Python"
SOURCE_DOI = "10.5281/zenodo.7908468"
SOURCE_ARCHIVE_VERSION = "v1 (Zenodo record 7908468)"
LICENSE_FILTER_BASIS = (
    "repository-level worst-case license classification from Phase 2B.2 audit "
    "(codesearchnet_license_repo_summary.csv); only CLEAR_PERMISSIVE_CANDIDATE accepted"
)
ACCEPTED_CATEGORY = "CLEAR_PERMISSIVE_CANDIDATE"

_peak_samples = []

def _current_rss_mb():
    if not _PSUTIL:
        return 0.0
    return psutil.Process().memory_info().rss / (1024 * 1024)

def _peak_rss_mb():
    cur = _current_rss_mb()
    _peak_samples.append(cur)
    return max(_peak_samples) if _peak_samples else cur


def load_repo_categories(path):
    cats = {}
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row["programming_language"] != "python":
                continue
            cats[row["repository"]] = row["repository_category"]
    return cats


def main():
    t0 = time.perf_counter()
    repo_cats = load_repo_categories(SUMMARY)
    print(f"loaded {len(repo_cats)} python repo categories")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_files = {p: open(OUT_DIR / f"{p}.jsonl", "w", encoding="utf-8") for p in ("train", "valid", "test")}

    # stats
    raw_total = 0
    accepted_total = 0
    invalid_json = 0
    empty_code = 0
    empty_docstring = 0
    missing_repo = 0
    missing_path = 0
    missing_url = 0
    missing_sha = 0
    excluded = Counter()  # category -> count (incl. MISSING/UNKNOWN)
    excluded_missing_meta = 0
    excluded_unknown = 0
    partition_mismatch = 0
    repos_by_part = defaultdict(set)
    accepted_repos = set()
    accepted_paths = set()
    sha_counts = Counter()
    code_hash_counts = Counter()
    commit_sha_counts = Counter()
    languages_raw = Counter()
    languages_acc = Counter()
    code_lens_acc = []
    doc_lens_acc = []
    per_part = defaultdict(lambda: {
        "raw": 0, "accepted": 0, "invalid_json": 0,
        "empty_code": 0, "empty_docstring": 0,
        "missing_repo": 0, "missing_path": 0, "missing_url": 0, "missing_sha": 0,
        "excluded": Counter(),
    })

    z = zipfile.ZipFile(ARCHIVE)
    members = sorted(
        (m for m in z.namelist() if m.endswith(".jsonl.gz")),
        key=lambda m: m,
    )
    for member in members:
        _peak_samples.append(_current_rss_mb())
        parts = member.split("/")
        idx = parts.index("jsonl")
        partition = parts[idx + 1]  # train/valid/test
        print(f"processing {member} (partition={partition})")
        with z.open(member) as raw:
            with gzip.GzipFile(fileobj=raw) as gz:
                for line in gz:
                    line = line.strip()
                    if not line:
                        continue
                    raw_total += 1
                    per_part[partition]["raw"] += 1
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        invalid_json += 1
                        per_part[partition]["invalid_json"] += 1
                        continue
                    repo = rec.get("repo")
                    path = rec.get("path")
                    url = rec.get("url")
                    sha = rec.get("sha")
                    code = rec.get("code") or ""
                    docstring = rec.get("docstring") or ""
                    language = rec.get("language")
                    rec_partition = rec.get("partition")

                    if not repo:
                        missing_repo += 1
                        per_part[partition]["missing_repo"] += 1
                    if not path:
                        missing_path += 1
                        per_part[partition]["missing_path"] += 1
                    if not url:
                        missing_url += 1
                        per_part[partition]["missing_url"] += 1
                    if not sha:
                        missing_sha += 1
                        per_part[partition]["missing_sha"] += 1
                    if language:
                        languages_raw[language] += 1
                    if code.strip() == "":
                        empty_code += 1
                        per_part[partition]["empty_code"] += 1
                    if docstring.strip() == "":
                        empty_docstring += 1
                        per_part[partition]["empty_docstring"] += 1
                    if rec_partition is not None and rec_partition != partition:
                        partition_mismatch += 1

                    # license filter
                    cat = repo_cats.get(repo) if repo else None
                    if repo not in repo_cats:
                        # missing metadata
                        excluded_missing_meta += 1
                        excluded["METADATA_MISSING"] += 1
                        per_part[partition]["excluded"]["METADATA_MISSING"] += 1
                        continue
                    if cat is None or cat == "":
                        excluded_unknown += 1
                        excluded["UNKNOWN/EMPTY"] += 1
                        per_part[partition]["excluded"]["UNKNOWN/EMPTY"] += 1
                        continue
                    if cat != ACCEPTED_CATEGORY:
                        excluded[cat] += 1
                        per_part[partition]["excluded"][cat] += 1
                        continue

                    # accepted
                    accepted_total += 1
                    per_part[partition]["accepted"] += 1
                    accepted_repos.add(repo)
                    if path:
                        accepted_paths.add(path)
                    if sha:
                        sha_counts[sha] += 1
                    code_hash_counts[hashlib.sha256(code.encode("utf-8", "surrogatepass")).hexdigest()] += 1
                    if language:
                        languages_acc[language] += 1
                    code_lens_acc.append(len(code))
                    doc_lens_acc.append(len(docstring))
                    repos_by_part[partition].add(repo)

                    prov = {
                        "source_dataset": SOURCE_DATASET,
                        "source_doi": SOURCE_DOI,
                        "source_archive_version": SOURCE_ARCHIVE_VERSION,
                        "source_repository": repo,
                        "source_path": path,
                        "source_url": url,
                        "source_partition": partition,
                        "license_category": cat,
                        "license_filter_basis": LICENSE_FILTER_BASIS,
                    }
                    out_rec = dict(rec)
                    out_rec["provenance"] = prov
                    out_files[partition].write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    for f in out_files.values():
        f.close()
    z.close()

    # duplicate stats (accepted). NOTE: the dataset's ``sha`` field is the repository
    # commit hash, so repeated values are expected and are NOT duplicate code.
    commit_sha_repeats = sum(n - 1 for n in sha_counts.values() if n > 1)
    dup_code_hashes = {h: n for h, n in code_hash_counts.items() if n > 1}
    dup_code_extra = sum(n - 1 for n in dup_code_hashes.values())
    dup_shas = {s: n for s, n in sha_counts.items() if n > 1}
    dup_extra = sum(n - 1 for n in dup_shas.values())

    # repo overlap between partitions (accepted)
    parts = ["train", "valid", "test"]
    overlaps = {}
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            inter = repos_by_part[parts[i]] & repos_by_part[parts[j]]
            overlaps[f"{parts[i]}_vs_{parts[j]}"] = len(inter)

    # stats output
    print("\n=== RESULTS ===")
    print("raw_total", raw_total)
    print("accepted_total", accepted_total)
    print("invalid_json", invalid_json)
    print("empty_code", empty_code)
    print("empty_docstring", empty_docstring)
    print("missing_repo", missing_repo)
    print("missing_path", missing_path)
    print("missing_url", missing_url)
    print("missing_sha", missing_sha)
    print("excluded_missing_meta", excluded_missing_meta)
    print("excluded_unknown", excluded_unknown)
    print("excluded breakdown", dict(excluded))
    print("partition_mismatch", partition_mismatch)
    print("unique accepted repos", len(accepted_repos))
    print("unique accepted paths", len(accepted_paths))
    print("commit-sha repeat occurrences (expected, commit hash)", commit_sha_repeats)
    print("distinct code hashes", len(code_hash_counts))
    print("dup distinct code hashes", len(dup_code_hashes), "extra occurrences", dup_code_extra)
    print("languages raw", dict(languages_raw))
    print("languages accepted", dict(languages_acc))
    print("code len avg/median", statistics.mean(code_lens_acc) if code_lens_acc else 0, statistics.median(code_lens_acc) if code_lens_acc else 0)
    print("doc len avg/median", statistics.mean(doc_lens_acc) if doc_lens_acc else 0, statistics.median(doc_lens_acc) if doc_lens_acc else 0)
    print("overlaps", overlaps)
    for p in parts:
        print(p, dict(per_part[p]))
    peak_rss = _peak_rss_mb()
    print("peak RSS MB", peak_rss)
    print("elapsed s", time.perf_counter() - t0)

    # write a stats json for the report
    stats = {
        "raw_total": raw_total,
        "accepted_total": accepted_total,
        "invalid_json": invalid_json,
        "empty_code": empty_code,
        "empty_docstring": empty_docstring,
        "missing_repo": missing_repo,
        "missing_path": missing_path,
        "missing_url": missing_url,
        "missing_sha": missing_sha,
        "excluded_missing_meta": excluded_missing_meta,
        "excluded_unknown": excluded_unknown,
        "excluded": dict(excluded),
        "partition_mismatch": partition_mismatch,
        "unique_accepted_repos": len(accepted_repos),
        "unique_accepted_paths": len(accepted_paths),
        "distinct_code_hashes": len(code_hash_counts),
        "dup_distinct_code_hashes": len(dup_code_hashes),
        "dup_code_extra_occurrences": dup_code_extra,
        "commit_sha_repeat_occurrences": commit_sha_repeats,
        "commit_sha_distinct": len(sha_counts),
        "languages_raw": dict(languages_raw),
        "languages_accepted": dict(languages_acc),
        "code_len_avg": statistics.mean(code_lens_acc) if code_lens_acc else 0,
        "code_len_median": statistics.median(code_lens_acc) if code_lens_acc else 0,
        "doc_len_avg": statistics.mean(doc_lens_acc) if doc_lens_acc else 0,
        "doc_len_median": statistics.median(doc_lens_acc) if doc_lens_acc else 0,
        "overlaps": overlaps,
        "per_part": {k: {kk: (dict(vv) if isinstance(vv, Counter) else vv) for kk, vv in v.items()} for k, v in per_part.items()},
        "peak_rss_mb": _peak_rss_mb(),
        "elapsed_s": time.perf_counter() - t0,
    }
    import json as _j
    with open(ROOT / "data" / "processed" / "codesearchnet_python_filtered" / "_stats.json", "w", encoding="utf-8") as f:
        _j.dump(stats, f, indent=2, ensure_ascii=False)
    print("stats written")


if __name__ == "__main__":
    main()