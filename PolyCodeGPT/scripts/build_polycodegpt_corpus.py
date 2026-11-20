"""Phase 2E: build the combined Python + JavaScript corpus with global
repository-level partition assignment (TEST > VALID > TRAIN).

Reads (never modifies) the two filtered CodeSearchNet datasets, resolves
cross-language repository leakage deterministically, and writes
data/processed/polycodegpt_corpus/{train,valid,test}.jsonl + _stats.json.

Outputs are written to *.jsonl.tmp and atomically renamed after validation.
"""

import hashlib
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

try:
    import psutil
    _PSUTIL = True
except Exception:
    _PSUTIL = False

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "processed" / "polycodegpt_corpus"

SOURCES = {
    "python": {
        "dir": ROOT / "data" / "processed" / "codesearchnet_python_filtered",
        "dataset": "CodeSearchNet Python",
    },
    "javascript": {
        "dir": ROOT / "data" / "processed" / "codesearchnet_javascript_filtered",
        "dataset": "CodeSearchNet JavaScript",
    },
}
PARTS = ("train", "valid", "test")
PRIORITY = {"test": 3, "valid": 2, "train": 1}  # TEST > VALIDATION > TRAIN
PROV_REQUIRED = {
    "source_dataset", "source_doi", "source_archive_version", "source_repository",
    "source_path", "source_url", "source_partition", "license_category",
    "license_filter_basis",
}

_peak = []


def rss_mb():
    return psutil.Process().memory_info().rss / (1024 * 1024) if _PSUTIL else 0.0


def norm_repo(repo):
    return (repo or "").strip().lower()


def iter_source(lang):
    src = SOURCES[lang]
    for part in PARTS:
        p = src["dir"] / f"{part}.jsonl"
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                yield part, json.loads(line)


def main():
    t0 = time.perf_counter()

    # ---------- pass 1: repository -> partitions / languages ----------
    print("pass 1: collecting repository partition membership ...")
    repo_parts = defaultdict(set)
    repo_langs = defaultdict(set)
    raw_counts = Counter()
    for lang in SOURCES:
        for part, rec in iter_source(lang):
            raw_counts[(lang, part)] += 1
            key = norm_repo(rec["repo"])
            repo_parts[key].add(part)
            repo_langs[key].add(lang)
        _peak.append(rss_mb())
        print(f"  {lang}: {sum(v for (l, p), v in raw_counts.items() if l == lang):,} records")

    global_part_of = {}
    reassigned_repos = []
    for repo, parts in repo_parts.items():
        target = max(parts, key=lambda p: PRIORITY[p])
        global_part_of[repo] = target
        if len(parts) > 1:
            reassigned_repos.append({"repository": repo, "original_partitions": sorted(parts), "global_partition": target})

    repos_multi_part_before = sum(1 for p in repo_parts.values() if len(p) > 1)
    cross_lang_repos = [r for r, l in repo_langs.items() if len(l) > 1]
    print(f"  repositories: {len(repo_parts):,}  multi-partition before: {repos_multi_part_before:,}  "
          f"cross-language: {len(cross_lang_repos):,}")

    # ---------- pass 2: write combined corpus ----------
    print("pass 2: writing combined corpus ...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = {p: OUT_DIR / f"{p}.jsonl.tmp" for p in PARTS}
    handles = {p: open(tmp[p], "w", encoding="utf-8") for p in PARTS}

    per = defaultdict(lambda: Counter())  # (lang, global_part) -> counters
    lengths = defaultdict(lambda: {"code": [], "doc": []})
    lang_repos = defaultdict(set)
    lang_paths = defaultdict(set)
    repos_by_part = defaultdict(set)
    stats_global = Counter()
    code_hash_partitions = defaultdict(lambda: Counter())  # code_sha -> (lang, part) -> n
    code_hash_meta = {}  # code_sha -> (lang, repo, path, part) example
    missing_prov = 0
    invalid = 0

    for lang in SOURCES:
        src = SOURCES[lang]
        for part, rec in iter_source(lang):
            prov = rec.get("provenance") or {}
            if not PROV_REQUIRED.issubset(prov):
                missing_prov += 1
            repo = prov.get("source_repository") or rec.get("repo")
            gpart = global_part_of[norm_repo(repo)]
            code = rec.get("code") or ""
            doc = rec.get("docstring") or ""

            code_sha = hashlib.sha256(code.encode("utf-8", "surrogatepass")).hexdigest()
            code_hash_partitions[code_sha][(lang, gpart)] += 1
            code_hash_meta.setdefault(code_sha, (lang, repo, rec.get("path"), gpart))

            rid = hashlib.sha256(
                f"{src['dataset']}|{repo}|{rec.get('path')}|{rec.get('func_name')}|{code_sha}".encode(
                    "utf-8", "surrogatepass"
                )
            ).hexdigest()

            out = dict(rec)
            out["record_id"] = rid
            out["code_sha256"] = code_sha
            out["original_partition"] = part
            out["global_partition"] = gpart
            out["source_dataset"] = src["dataset"]
            out["source_doi"] = "10.5281/zenodo.7908468"
            out["source_archive_version"] = "v1 (Zenodo record 7908468)"
            out["license_category"] = prov.get("license_category")
            out["license_filter_basis"] = prov.get("license_filter_basis")
            handles[gpart].write(json.dumps(out, ensure_ascii=False) + "\n")

            c = per[(lang, gpart)]
            c["records"] += 1
            c["code_chars"] += len(code)
            c["doc_chars"] += len(doc)
            c["empty_code"] += 0 if code.strip() else 1
            c["empty_docstring"] += 0 if doc.strip() else 1
            if not prov:
                c["missing_provenance"] += 1
            lengths[(lang, gpart)]["code"].append(len(code))
            lengths[(lang, gpart)]["doc"].append(len(doc))
            lang_repos[(lang, gpart)].add(norm_repo(repo))
            lang_paths[(lang, gpart)].add(rec.get("path"))
            repos_by_part[gpart].add(norm_repo(repo))
            stats_global[gpart] += 1

    for f in handles.values():
        f.flush()
        os.fsync(f.fileno())
        f.close()
    _peak.append(rss_mb())

    # ---------- validate temp files before promotion ----------
    print("validating temporary outputs ...")
    for p in PARTS:
        n = 0
        with open(tmp[p], encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                assert rec["global_partition"] == p, "partition mismatch"
                assert rec["license_category"] == "CLEAR_PERMISSIVE_CANDIDATE", "license violation"
                n += 1
        if n != stats_global[p]:
            print(f"TEMP VALIDATION FAILED {p}: {n} != {stats_global[p]}")
            sys.exit(1)
        print(f"  {p}: {n:,} records validated")
    for p in PARTS:
        os.replace(tmp[p], OUT_DIR / f"{p}.jsonl")
    print("outputs promoted")

    # ---------- duplicate analysis ----------
    dup_within = 0
    dup_cross_partition = 0
    dup_cross_lang = 0
    pair_counts = Counter()
    dup_examples = []
    for code_sha, buckets in code_hash_partitions.items():
        total = sum(buckets.values())
        if total < 2:
            continue
        langs = {l for (l, _p) in buckets}
        parts = {p for (_l, p) in buckets}
        if total > 1:
            dup_within += total - 1
        if len(langs) > 1:
            dup_cross_lang += total
        if len(parts) > 1:
            dup_cross_partition += total
            for a in parts:
                for b in parts:
                    if a != b:
                        pair_counts[f"{a}->{b}"] += 1
            if len(dup_examples) < 20:
                lang, repo, path, _ = code_hash_meta[code_sha]
                dup_examples.append({
                    "code_sha256": code_sha,
                    "occurrences": total,
                    "partitions": dict(buckets),
                    "example_repo": repo,
                    "example_path": path,
                })

    # ---------- quality statistics ----------
    def block(lang, part):
        c = per[(lang, part)]
        cs, ds = lengths[(lang, part)]["code"], lengths[(lang, part)]["doc"]
        return {
            "records": c["records"],
            "unique_repositories": len(lang_repos[(lang, part)]),
            "unique_source_paths": len(lang_paths[(lang, part)]),
            "total_code_chars": c["code_chars"],
            "total_docstring_chars": c["doc_chars"],
            "avg_code_chars": statistics.mean(cs) if cs else 0,
            "median_code_chars": statistics.median(cs) if cs else 0,
            "avg_docstring_chars": statistics.mean(ds) if ds else 0,
            "median_docstring_chars": statistics.median(ds) if ds else 0,
            "empty_code": c["empty_code"],
            "empty_docstring": c["empty_docstring"],
            "missing_provenance": c["missing_provenance"],
            "invalid_json": 0,
        }

    quality = {}
    for lang in SOURCES:
        quality[lang] = {}
        for part in PARTS:
            quality[lang][part] = block(lang, part)

    # ---------- reassignment accounting ----------
    orig_counts = Counter({f"{k[0]}/{k[1]}": v for k, v in raw_counts.items()})
    final_by_orig = Counter()
    records_reassigned = 0
    for lang in SOURCES:
        for part, rec in iter_source(lang):
            g = global_part_of[norm_repo(rec["repo"])]
            final_by_orig[f"{part}->{g}"] += 1
            if g != part:
                records_reassigned += 1

    after_overlaps = {
        f"{a}_vs_{b}": len(repos_by_part[a] & repos_by_part[b])
        for a, b in (("train", "valid"), ("train", "test"), ("valid", "test"))
    }

    stats = {
        "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sources": {l: str(s["dir"]).replace("\\", "/") for l, s in SOURCES.items()},
        "precedence": "TEST > VALID > TRAIN",
        "raw_counts": orig_counts,
        "raw_total": sum(raw_counts.values()),
        "combined_total": sum(stats_global.values()),
        "per_language_totals": {
            l: sum(per[(l, p)]["records"] for p in PARTS) for l in SOURCES
        },
        "final_partition_counts": dict(stats_global),
        "records_reassigned": records_reassigned,
        "repositories_total": len(repo_parts),
        "repositories_multi_partition_before": repos_multi_part_before,
        "repositories_reassigned": len(reassigned_repos),
        "cross_language_repositories": len(cross_lang_repos),
        "cross_language_repository_examples": sorted(cross_lang_repos)[:20],
        "overlaps_after_resolution": after_overlaps,
        "records_by_original_to_final": dict(final_by_orig),
        "duplicates": {
            "distinct_code_hashes": len(code_hash_partitions),
            "records_sharing_code_with_another_record": dup_within,
            "records_in_cross_partition_duplicates": dup_cross_partition,
            "records_in_cross_language_duplicates": dup_cross_lang,
            "cross_partition_pairs": dict(pair_counts),
            "examples": dup_examples,
        },
        "quality": quality,
        "peak_rss_mb": max(_peak) if _peak else rss_mb(),
        "elapsed_s": time.perf_counter() - t0,
    }
    with open(OUT_DIR / "_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\n=== RESULT ===")
    print("raw total", stats["raw_total"], "combined", stats["combined_total"])
    print("per language", stats["per_language_totals"])
    print("final partitions", stats["final_partition_counts"])
    print("repos reassigned", stats["repositories_reassigned"], "records reassigned", records_reassigned)
    print("cross-language repos", stats["cross_language_repositories"])
    print("overlaps after", after_overlaps)
    print("dups", {k: v for k, v in stats["duplicates"].items() if k != "examples"})
    print("elapsed", round(stats["elapsed_s"], 1), "s peak RSS", round(stats["peak_rss_mb"], 1), "MB")


if __name__ == "__main__":
    main()
