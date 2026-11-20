"""Phase 2D: stream CodeSearchNet JavaScript JSONL shards from the Zenodo ZIP,
apply repository-level license filtering, write filtered JSONL + provenance.

Outputs are written to ``*.jsonl.tmp`` and only renamed into place after the
whole corpus has been written and validated, so a partially written final
``*.jsonl`` can never exist.
"""

import csv
import gzip
import hashlib
import json
import os
import statistics
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

try:
    import psutil
    _PSUTIL = True
except Exception:
    _PSUTIL = False

LANG = "javascript"
ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data" / "raw" / "codesearchnet_zenodo_javascript" / "javascript.zip"
SUMMARY = ROOT / "data" / "manifests" / "codesearchnet_license_repo_summary.csv"
OUT_DIR = ROOT / "data" / "processed" / "codesearchnet_javascript_filtered"

SOURCE_DATASET = "CodeSearchNet JavaScript"
SOURCE_DOI = "10.5281/zenodo.7908468"
SOURCE_ARCHIVE_VERSION = "v1 (Zenodo record 7908468)"
LICENSE_FILTER_BASIS = (
    "repository-level worst-case license classification from Phase 2B.2 audit "
    "(codesearchnet_license_repo_summary.csv); only CLEAR_PERMISSIVE_CANDIDATE accepted"
)
ACCEPTED_CATEGORY = "CLEAR_PERMISSIVE_CANDIDATE"
PROVENANCE_KEYS = (
    "source_dataset",
    "source_doi",
    "source_archive_version",
    "source_repository",
    "source_path",
    "source_url",
    "source_partition",
    "license_category",
    "license_filter_basis",
)

_peak = []


def rss_mb():
    return psutil.Process().memory_info().rss / (1024 * 1024) if _PSUTIL else 0.0


def peak_mb():
    cur = rss_mb()
    _peak.append(cur)
    return max(_peak)


def load_repo_categories(path, lang):
    cats = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["programming_language"] == lang:
                cats[row["repository"]] = row["repository_category"]
    return cats


def main():
    t0 = time.perf_counter()
    repo_cats = load_repo_categories(SUMMARY, LANG)
    print(f"loaded {len(repo_cats)} {LANG} repo categories")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parts = ("train", "valid", "test")
    tmp_paths = {p: OUT_DIR / f"{p}.jsonl.tmp" for p in parts}
    handles = {p: open(tmp_paths[p], "w", encoding="utf-8") for p in parts}

    raw_total = 0
    accepted_total = 0
    invalid_json = 0
    empty_code = 0
    empty_docstring = 0
    missing_repo = missing_path = missing_url = missing_sha = 0
    excluded = Counter()
    repos_by_part = defaultdict(set)
    accepted_repos = set()
    accepted_paths = set()
    code_hashes = Counter()
    sha_counts = Counter()
    languages_raw = Counter()
    languages_acc = Counter()
    code_lens, doc_lens = [], []
    per_part = defaultdict(lambda: Counter())
    for p in parts:
        per_part[p]["raw"] = 0
        per_part[p]["accepted"] = 0

    z = zipfile.ZipFile(ARCHIVE)
    members = sorted(m for m in z.namelist() if m.endswith(".jsonl.gz"))
    for member in members:
        _peak.append(rss_mb())
        tokens = member.split("/")
        partition = tokens[tokens.index("jsonl") + 1]
        print(f"processing {member} (partition={partition})")
        with z.open(member) as raw, gzip.GzipFile(fileobj=raw) as gz:
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
                if not code.strip():
                    empty_code += 1
                    per_part[partition]["empty_code"] += 1
                if not docstring.strip():
                    empty_docstring += 1
                    per_part[partition]["empty_docstring"] += 1
                if rec.get("partition") not in (None, partition):
                    per_part[partition]["partition_mismatch"] += 1

                cat = repo_cats.get(repo) if repo else None
                if cat is None:
                    excluded["METADATA_MISSING"] += 1
                    per_part[partition]["excluded_METADATA_MISSING"] += 1
                    continue
                if cat == "":
                    excluded["UNKNOWN"] += 1
                    per_part[partition]["excluded_UNKNOWN"] += 1
                    continue
                if cat != ACCEPTED_CATEGORY:
                    excluded[cat] += 1
                    per_part[partition][f"excluded_{cat}"] += 1
                    continue

                accepted_total += 1
                per_part[partition]["accepted"] += 1
                accepted_repos.add(repo)
                repos_by_part[partition].add(repo)
                if path:
                    accepted_paths.add(path)
                if sha:
                    sha_counts[sha] += 1
                if language:
                    languages_acc[language] += 1
                code_hashes[hashlib.sha256(code.encode("utf-8", "surrogatepass")).hexdigest()] += 1
                code_lens.append(len(code))
                doc_lens.append(len(docstring))

                out_rec = dict(rec)
                out_rec["provenance"] = {
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
                handles[partition].write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    for f in handles.values():
        f.flush()
        os.fsync(f.fileno())
        f.close()
    z.close()

    # >>> validate temp files before promoting them to final names <<<
    print("validating temporary outputs ...")
    out_stats = {}
    for p in parts:
        n = 0
        bad = 0
        bad_prov = 0
        bad_cat = 0
        with open(tmp_paths[p], encoding="utf-8") as fh:
            for line in fh:
                n += 1
                rec = json.loads(line)
                prov = rec.get("provenance", {})
                if any(k not in prov for k in PROVENANCE_KEYS):
                    bad_prov += 1
                if prov.get("license_category") != ACCEPTED_CATEGORY:
                    bad_cat += 1
        if bad_prov or bad_cat or bad or n != per_part[p]["accepted"]:
            print(f"TEMP VALIDATION FAILED for {p}: n={n} expected={per_part[p]['accepted']} "
                  f"bad_prov={bad_prov} bad_cat={bad_cat} bad_json={bad}")
            for f in handles.values():
                f.close()
            sys.exit(1)
        out_stats[p] = n
    print("temp validation passed:", out_stats)

    for p in parts:
        os.replace(tmp_paths[p], OUT_DIR / f"{p}.jsonl")
    print("outputs promoted to final names")

    dup_code = {h: n for h, n in code_hashes.items() if n > 1}
    dup_code_extra = sum(n - 1 for n in dup_code.values())
    commit_repeats = sum(n - 1 for n in sha_counts.values() if n > 1)
    overlaps = {
        f"{a}_vs_{b}": len(repos_by_part[a] & repos_by_part[b])
        for a, b in (("train", "valid"), ("train", "test"), ("valid", "test"))
    }

    stats = {
        "language": LANG,
        "archive": str(ARCHIVE).replace("\\", "/"),
        "raw_total": raw_total,
        "accepted_total": accepted_total,
        "invalid_json": invalid_json,
        "empty_code": empty_code,
        "empty_docstring": empty_docstring,
        "missing_repo": missing_repo,
        "missing_path": missing_path,
        "missing_url": missing_url,
        "missing_sha": missing_sha,
        "excluded": dict(excluded),
        "unique_accepted_repos": len(accepted_repos),
        "unique_accepted_paths": len(accepted_paths),
        "distinct_code_hashes": len(code_hashes),
        "dup_distinct_code_hashes": len(dup_code),
        "dup_code_extra_occurrences": dup_code_extra,
        "commit_sha_distinct": len(sha_counts),
        "commit_sha_repeat_occurrences": commit_repeats,
        "languages_raw": dict(languages_raw),
        "languages_accepted": dict(languages_acc),
        "code_len_avg": statistics.mean(code_lens) if code_lens else 0,
        "code_len_median": statistics.median(code_lens) if code_lens else 0,
        "doc_len_avg": statistics.mean(doc_lens) if doc_lens else 0,
        "doc_len_median": statistics.median(doc_lens) if doc_lens else 0,
        "overlaps": overlaps,
        "per_part": {p: dict(per_part[p]) for p in parts},
        "peak_rss_mb": peak_mb(),
        "elapsed_s": time.perf_counter() - t0,
    }
    with open(OUT_DIR / "_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\n=== RESULTS ===")
    for k in ("raw_total", "accepted_total", "invalid_json", "empty_code", "empty_docstring",
              "missing_repo", "missing_path", "missing_url", "missing_sha",
              "unique_accepted_repos", "unique_accepted_paths",
              "distinct_code_hashes", "dup_distinct_code_hashes", "dup_code_extra_occurrences",
              "commit_sha_distinct", "commit_sha_repeat_occurrences"):
        print(f"{k}: {stats[k]}")
    print("excluded:", stats["excluded"])
    print("overlaps:", overlaps)
    for p in parts:
        print(p, stats["per_part"][p])
    print(f"peak RSS MB {stats['peak_rss_mb']:.1f}  elapsed {stats['elapsed_s']:.1f}s")


if __name__ == "__main__":
    main()
