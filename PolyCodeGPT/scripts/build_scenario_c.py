#!/usr/bin/env python
"""Production Scenario C function-level dataset extraction pipeline.

Streams CodeParrot Parquet shards, extracts function/method candidates using
tested AST/Tree-sitter extractors, assigns partitions via Scenario B repo sets,
deduplicates across the entire corpus via SQLite, and writes isolated outputs.

Hang protection: each parent file is processed via apply_async with a per-file
wall-clock timeout. On timeout the entire mp.Pool is terminated and rebuilt,
which reliably kills the hung OS worker process on Windows. The offending file
is recorded in quarantine_failures.jsonl and skipped; the rest of the shard
continues.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import multiprocessing as mp
import os
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pyarrow.parquet as pq
from tokenizers import Tokenizer
from tree_sitter import Language, Parser
import tree_sitter_javascript as tsjs

from polycodegpt.data_v2.constants import LICENSE_ALLOWLIST, SCHEMA_VERSION
from polycodegpt.data_v2.language import normalize_repo
from polycodegpt.data_v2.schema import assert_no_forbidden_input, serialize_code
from scripts.audit_scenario_c_extraction import (
    dependency_flags,
    js_candidates,
    python_candidates,
    stats as audit_stats,
)

DEFAULT_SHARD_DIR = PROJECT_ROOT / "data" / "raw" / "v2" / "codeparrot_github_code"
DEFAULT_SCENARIO_B = PROJECT_ROOT / "data" / "processed" / "polycodegpt_v2" / "scenario_b"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "interim" / "scenario_c"
TOKENIZER_PATH = PROJECT_ROOT / "tokenizer" / "artifacts" / "v2" / "polycodegpt_tokenizer_v2.json"
PARROT_SNAPSHOT = "codeparrot/github-code@b5661e6b17396364b2bcf8e68977b0d28e1ebd19"
PARROT_EVIDENCE = (
    "CodeParrot per-example `license` field; only MIT, Apache-2.0, BSD-2-Clause, "
    "BSD-3-Clause, ISC, CC0-1.0, Unlicense accepted; dataset-level 'other' is not permission"
)

# Per-file timeout in seconds. Files that do not complete extraction within
# this budget are quarantined. 60 s is generous — normal files finish in <2 s.
FILE_TIMEOUT_SECONDS = 60

# Global variables for worker processes
_WORKER_TOKENIZER: Tokenizer | None = None
_WORKER_JS_PARSER: Parser | None = None


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_scenario_b_repositories(
    scenario_b_dir: Path, cache_file: Path | None = None
) -> tuple[set[str], set[str]]:
    """Loads repository sets from Scenario B train and valid partitions.

    Ensures zero overlap and never touches test partition.
    """
    train_path = scenario_b_dir / "train.jsonl"
    valid_path = scenario_b_dir / "valid.jsonl"

    assert_no_forbidden_input(str(train_path))
    assert_no_forbidden_input(str(valid_path))

    if cache_file and cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            train_repos = set(data["train"])
            valid_repos = set(data["valid"])
            overlap = train_repos & valid_repos
            if len(overlap) == 0:
                print(
                    f"Loaded repository sets from cache: {len(train_repos):,} train, "
                    f"{len(valid_repos):,} valid",
                    flush=True,
                )
                return train_repos, valid_repos
        except Exception:
            pass

    print("Building repository sets from Scenario B partitions...", flush=True)
    train_repos: set[str] = set()
    valid_repos: set[str] = set()

    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            repo = rec.get("repo")
            if repo:
                train_repos.add(normalize_repo(repo))

    with open(valid_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            repo = rec.get("repo")
            if repo:
                valid_repos.add(normalize_repo(repo))

    overlap = train_repos & valid_repos
    if overlap:
        raise ValueError(f"CRITICAL: Found {len(overlap)} overlapping repos between train and valid!")

    print(
        f"Built repository sets: {len(train_repos):,} train repos, "
        f"{len(valid_repos):,} valid repos, 0 overlap.",
        flush=True,
    )

    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(
                {"train": sorted(train_repos), "valid": sorted(valid_repos)},
                f,
            )

    return train_repos, valid_repos


def init_worker():
    """Initializes worker process resources."""
    global _WORKER_TOKENIZER, _WORKER_JS_PARSER
    _WORKER_TOKENIZER = Tokenizer.from_file(str(TOKENIZER_PATH))
    _WORKER_JS_PARSER = Parser(Language(tsjs.language()))


def extract_parent_candidates(parent: dict) -> dict:
    """Worker task: extracts candidates from one parent file and computes metadata."""
    global _WORKER_TOKENIZER
    if _WORKER_TOKENIZER is None:
        init_worker()

    language = parent.get("language")
    audit_stats.clear()

    if language == "python":
        extracted = python_candidates(parent)
    elif language == "javascript":
        extracted = js_candidates(parent)
    else:
        extracted = []

    parse_stats = dict(audit_stats)
    candidate_occurrences = len(extracted)

    valid_candidates = []
    syntax_invalids = 0
    parent_raw = parent.get("raw_content") or ""
    parent_sha = sha256_text(parent_raw)
    parent_record_id = sha256_text(
        parent.get("source_id", "") + "\0" +
        parent.get("repo", "") + "\0" +
        parent.get("path", "") + "\0" +
        parent_sha
    )

    for item in extracted:
        if not item.get("syntax_valid"):
            syntax_invalids += 1
            continue

        reconstruction = item.get("reconstruction", "")
        code_hash = sha256_text(language + "\0" + reconstruction)
        flags = dependency_flags(language, item["code"])

        serialized = serialize_code(
            language,
            "function",
            file=parent.get("path", ""),
            context=item.get("context", ""),
            imports=item.get("imports", ""),
            docstring=item.get("docstring", ""),
            signature=item.get("signature", ""),
            code=item.get("code", ""),
        )

        token_ids = _WORKER_TOKENIZER.encode(serialized).ids
        token_count = len(token_ids)

        record_id = sha256_text(
            parent.get("source_id", "") + "\0" +
            parent.get("repo", "") + "\0" +
            parent.get("path", "") + "\0" +
            item["category"] + "\0" +
            item["symbol"] + "\0" +
            str(item["start_line"]) + "\0" +
            code_hash
        )

        candidate_record = {
            "record_id": record_id,
            "parent_record_id": parent_record_id,
            "source_id": parent.get("source_id"),
            "source_snapshot": parent.get("source_snapshot"),
            "repo": parent.get("repo"),
            "path": parent.get("path"),
            "license_id": parent.get("license_id"),
            "license_evidence": parent.get("license_evidence"),
            "source_url": parent.get("source_url"),
            "origin_partition": parent.get("partition"),
            "partition": parent.get("partition"),
            "language": language,
            "task": "function",
            "category": item["category"],
            "symbol": item["symbol"],
            "parent_category": parent.get("category"),
            "content_kind": "code",
            "extraction_method": "python_ast" if language == "python" else "tree_sitter",
            "start_line": item["start_line"],
            "end_line": item["end_line"],
            "parent_shard": parent.get("shard"),
            "parent_source_index": parent.get("source_index"),
            "parent_sha256": parent_sha,
            "code_sha256": code_hash,
            "dependency_flags": flags,
            "token_count": token_count,
            "fits_512": token_count <= 512,
            "schema_version": SCHEMA_VERSION,
            "text": serialized,
            "code": item["code"],
            "syntax_valid": item["syntax_valid"],
        }
        valid_candidates.append(candidate_record)

    return {
        "valid_candidates": valid_candidates,
        "parse_stats": parse_stats,
        "candidate_occurrences": candidate_occurrences,
        "syntax_invalids": syntax_invalids,
    }


def _make_pool(workers: int) -> mp.Pool:
    """Creates a fresh worker pool with maxtasksperchild for memory hygiene."""
    # maxtasksperchild=200: workers are recycled after 200 tasks, preventing
    # unbounded memory growth from large AST objects.
    return mp.Pool(processes=workers, initializer=init_worker, maxtasksperchild=200)


class ScenarioCExtractor:
    def __init__(
        self,
        shard_dir: Path,
        scenario_b_dir: Path,
        output_dir: Path,
        workers: int = 8,
        chunksize: int = 1,
        file_timeout: int = FILE_TIMEOUT_SECONDS,
    ):
        self.shard_dir = shard_dir
        self.scenario_b_dir = scenario_b_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.workers = max(1, workers)
        self.chunksize = chunksize
        self.file_timeout = file_timeout

        self.db_path = self.output_dir / "dedup.sqlite"
        self.train_out_path = self.output_dir / "train_functions.jsonl"
        self.valid_out_path = self.output_dir / "valid_functions.jsonl"
        self.unknown_out_path = self.output_dir / "unknown_repositories.jsonl"
        self.quarantine_path = self.output_dir / "quarantine_failures.jsonl"
        self.progress_path = self.output_dir / "progress.json"
        self.report_path = self.output_dir / "extraction_report.json"
        self.cache_repo_path = self.output_dir / "repo_partitions_cache.json"

        self.train_repos: set[str] = set()
        self.valid_repos: set[str] = set()
        self.seen_hashes: set[str] = set()

        self._init_sqlite()

    def _init_sqlite(self):
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS completed_shards (
                        shard_name TEXT PRIMARY KEY,
                        completed_at TEXT,
                        train_bytes INTEGER,
                        valid_bytes INTEGER,
                        parent_files INTEGER,
                        candidate_count INTEGER,
                        retained_count INTEGER,
                        duplicate_count INTEGER,
                        elapsed_seconds REAL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS seen_hashes (
                        code_sha256 TEXT PRIMARY KEY,
                        record_id TEXT,
                        partition TEXT
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS unknown_repositories (
                        repo TEXT PRIMARY KEY,
                        python_files INTEGER DEFAULT 0,
                        javascript_files INTEGER DEFAULT 0
                    )
                    """
                )
        finally:
            conn.close()

    def load_completed_state(self) -> tuple[set[str], int, int]:
        """Recovers state from SQLite and truncates outputs to last committed offsets."""
        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.cursor()
            cur.execute("SELECT shard_name FROM completed_shards")
            completed = {row[0] for row in cur.fetchall()}

            # Load existing seen hashes
            cur.execute("SELECT code_sha256 FROM seen_hashes")
            self.seen_hashes = {row[0] for row in cur.fetchall()}

            # Find last valid offsets
            cur.execute(
                """
                SELECT train_bytes, valid_bytes FROM completed_shards
                ORDER BY rowid DESC LIMIT 1
                """
            )
            row = cur.fetchone()
            train_bytes = row[0] if row else 0
            valid_bytes = row[1] if row else 0
        finally:
            conn.close()

        # Safe truncate output files if they exceed committed offsets
        for path, offset in [(self.train_out_path, train_bytes), (self.valid_out_path, valid_bytes)]:
            if path.exists():
                curr_size = path.stat().st_size
                if curr_size > offset:
                    print(f"Truncating {path.name} from {curr_size:,} to {offset:,} bytes...", flush=True)
                    with open(path, "r+b") as fh:
                        fh.seek(offset)
                        fh.truncate()
            elif offset == 0:
                path.touch()

        return completed, train_bytes, valid_bytes

    def get_shards(self) -> list[Path]:
        shards = sorted(self.shard_dir.glob("*.parquet"))
        if not shards:
            raise FileNotFoundError(f"No parquet shards found in {self.shard_dir}")
        return shards

    def _process_parent_with_timeout(
        self,
        pool: mp.Pool,
        parent: dict,
        quarantine_fh,
        quarantine_counts: Counter,
    ) -> tuple[dict | None, mp.Pool]:
        """
        Submits one parent file to the pool with a wall-clock timeout.

        Returns (result_dict, pool). If the task times out:
        - The pool is terminated and a fresh one is created (killing the hung worker).
        - The file is recorded in the quarantine log.
        - Returns (None, new_pool).

        This is the only reliable way to kill a stuck worker on Windows.
        """
        t0 = time.monotonic()
        async_result = pool.apply_async(extract_parent_candidates, (parent,))
        try:
            result = async_result.get(timeout=self.file_timeout)
            return result, pool
        except mp.TimeoutError:
            elapsed = time.monotonic() - t0
            # TERMINATE the entire pool — this sends SIGTERM to all worker PIDs,
            # which on Windows translates to TerminateProcess(). The hung worker
            # is actually killed, not just abandoned.
            pool.terminate()
            pool.join()
            # Quarantine record
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "failure_type": "timeout",
                "elapsed_seconds": round(elapsed, 2),
                "timeout_budget": self.file_timeout,
                "shard": parent.get("shard"),
                "repo": parent.get("repo"),
                "path": parent.get("path"),
                "language": parent.get("language"),
                "content_len": len(parent.get("raw_content") or ""),
            }
            quarantine_fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            quarantine_fh.flush()
            quarantine_counts["timeout"] += 1
            lang = parent.get("language", "unknown")
            quarantine_counts[f"timeout_{lang}"] += 1
            print(
                f"  [QUARANTINE] Timeout ({elapsed:.1f}s) on {parent.get('path')} "
                f"[{lang}] in {parent.get('shard')} — pool rebuilt.",
                flush=True,
            )
            # Build a fresh pool so remaining files in this shard can proceed
            new_pool = _make_pool(self.workers)
            return None, new_pool
        except Exception as exc:
            elapsed = time.monotonic() - t0
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "failure_type": "worker_exception",
                "error": str(exc)[:500],
                "elapsed_seconds": round(elapsed, 2),
                "shard": parent.get("shard"),
                "repo": parent.get("repo"),
                "path": parent.get("path"),
                "language": parent.get("language"),
                "content_len": len(parent.get("raw_content") or ""),
            }
            quarantine_fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            quarantine_fh.flush()
            quarantine_counts["worker_exception"] += 1
            return None, pool

    def run(
        self,
        shard_limit: int | None = None,
        shard_offset: int = 0,
        max_parents: int | None = None,
    ) -> dict:
        start_time = time.time()
        start_dt = datetime.now(timezone.utc).isoformat()

        # Load repository partitions
        self.train_repos, self.valid_repos = load_scenario_b_repositories(
            self.scenario_b_dir, self.cache_repo_path
        )

        all_shards = self.get_shards()
        total_discovered = len(all_shards)

        if shard_offset:
            all_shards = all_shards[shard_offset:]
        if shard_limit is not None:
            all_shards = all_shards[:shard_limit]

        completed_shards, train_bytes, valid_bytes = self.load_completed_state()
        print(
            f"Discovered {total_discovered} shards. "
            f"Already completed: {len(completed_shards)}. "
            f"Loaded {len(self.seen_hashes):,} existing code hashes.",
            flush=True,
        )

        # Open JSONL files in append mode
        train_fh = open(self.train_out_path, "a", encoding="utf-8", newline="\n")
        valid_fh = open(self.valid_out_path, "a", encoding="utf-8", newline="\n")
        quarantine_fh = open(self.quarantine_path, "a", encoding="utf-8", newline="\n")

        # Global extraction statistics
        shards_completed_this_run = 0
        shards_skipped = 0
        total_parents_scanned = Counter()
        parse_failures = Counter()
        candidate_occurrences = 0
        syntax_invalids = 0
        duplicate_count = 0
        retained_counts = Counter()
        retained_by_type = defaultdict(Counter)
        tokens_by_lang_part = Counter()
        token_histogram = defaultdict(Counter)  # lang -> Counter(token_count -> freq)
        repo_token_counts = Counter()
        dependency_flag_counts = defaultdict(Counter)
        missing_provenance_count = 0
        unknown_repos_memory = defaultdict(lambda: {"python": 0, "javascript": 0})
        total_parents_processed = 0
        quarantine_counts: Counter = Counter()

        # Required provenance keys
        required_provenance = (
            "source_id", "source_snapshot", "repo", "path",
            "license_id", "license_evidence", "source_url",
            "origin_partition", "parent_shard", "parent_sha256", "code_sha256"
        )

        pool = _make_pool(self.workers)

        try:
            for shard_idx, shard_path in enumerate(all_shards, 1):
                shard_name = shard_path.name
                if shard_name in completed_shards:
                    shards_skipped += 1
                    continue

                shard_t0 = time.time()
                shard_parents_scanned = Counter()
                shard_candidates = 0
                shard_retained = 0
                shard_duplicates = 0
                shard_new_hashes: list[tuple[str, str, str]] = []
                shard_pending_hashes: set[str] = set()
                shard_unknown_repos: dict[str, dict[str, int]] = defaultdict(lambda: {"python": 0, "javascript": 0})

                pf = pq.ParquetFile(shard_path)
                num_rgs = pf.num_row_groups

                for rg in range(num_rgs):
                    table = pf.read_row_group(
                        rg, columns=["repo_name", "path", "size", "license", "content"]
                    )
                    records = table.to_pylist()

                    for idx, rec in enumerate(records):
                        path = rec.get("path") or ""
                        suffix = Path(path).suffix.lower()
                        if suffix not in {".py", ".js"}:
                            continue

                        language = "python" if suffix == ".py" else "javascript"
                        total_parents_scanned[language] += 1
                        shard_parents_scanned[language] += 1

                        license_raw = (rec.get("license") or "").strip().lower()
                        if license_raw not in LICENSE_ALLOWLIST:
                            continue

                        repo = normalize_repo(rec.get("repo_name") or "")
                        if not repo:
                            continue

                        # Partition check
                        if repo in self.train_repos:
                            partition = "train"
                        elif repo in self.valid_repos:
                            partition = "valid"
                        else:
                            # Unknown repo -> skip and track
                            unknown_repos_memory[repo][language] += 1
                            shard_unknown_repos[repo][language] += 1
                            continue

                        raw_content = rec.get("content") or ""
                        if not raw_content.strip():
                            continue

                        parent_dict = {
                            "content_kind": "code",
                            "source_id": "CODEPARROT",
                            "source_snapshot": PARROT_SNAPSHOT,
                            "repo": repo,
                            "path": path,
                            "license_id": license_raw,
                            "license_evidence": PARROT_EVIDENCE,
                            "source_url": f"https://github.com/{repo}/blob/HEAD/{path}",
                            "origin_partition": partition,
                            "partition": partition,
                            "language": language,
                            "raw_content": raw_content,
                            "docstring": "",
                            "signature": "",
                            "imports": "",
                            "category": "module",
                            "alignment": "",
                            "source_index": idx,
                            "shard": shard_name,
                        }

                        total_parents_processed += 1

                        # --- Per-file timeout dispatch ---
                        res, pool = self._process_parent_with_timeout(
                            pool, parent_dict, quarantine_fh, quarantine_counts
                        )
                        if res is None:
                            # File was quarantined (timeout or worker crash); skip it
                            continue

                        candidate_occurrences += res["candidate_occurrences"]
                        shard_candidates += res["candidate_occurrences"]
                        syntax_invalids += res["syntax_invalids"]

                        for k, v in res["parse_stats"].items():
                            parse_failures[k] += v

                        for cand in res["valid_candidates"]:
                            c_hash = cand["code_sha256"]
                            if c_hash in self.seen_hashes or c_hash in shard_pending_hashes:
                                duplicate_count += 1
                                shard_duplicates += 1
                                continue

                            shard_pending_hashes.add(c_hash)
                            shard_new_hashes.append((c_hash, cand["record_id"], cand["partition"]))

                            # Check provenance
                            if any(not cand.get(k) for k in required_provenance):
                                missing_provenance_count += 1

                            lang = cand["language"]
                            part = cand["partition"]
                            n_tok = cand["token_count"]
                            cat = cand["category"]

                            retained_counts[f"{lang}_{part}"] += 1
                            retained_counts[lang] += 1
                            retained_counts[part] += 1
                            retained_counts["total"] += 1
                            shard_retained += 1

                            retained_by_type[lang][cat] += 1
                            tokens_by_lang_part[f"{lang}_{part}"] += n_tok
                            tokens_by_lang_part[lang] += n_tok
                            tokens_by_lang_part[part] += n_tok
                            tokens_by_lang_part["total"] += n_tok

                            token_histogram[lang][n_tok] += 1
                            token_histogram["overall"][n_tok] += 1
                            repo_token_counts[cand["repo"]] += n_tok

                            for flag in cand["dependency_flags"]:
                                dependency_flag_counts[lang][flag] += 1

                            line_str = json.dumps(cand, ensure_ascii=False) + "\n"
                            if part == "train":
                                train_fh.write(line_str)
                            else:
                                valid_fh.write(line_str)

                        if max_parents and total_parents_processed >= max_parents:
                            print(f"Reached max parent limit of {max_parents}. Halting shard loop.", flush=True)
                            break

                    if max_parents and total_parents_processed >= max_parents:
                        break

                # Atomic shard flush & commit
                train_fh.flush()
                os.fsync(train_fh.fileno())
                valid_fh.flush()
                os.fsync(valid_fh.fileno())

                new_train_bytes = train_fh.tell()
                new_valid_bytes = valid_fh.tell()
                shard_dt = datetime.now(timezone.utc).isoformat()
                shard_elapsed = time.time() - shard_t0

                conn = sqlite3.connect(self.db_path)
                try:
                    with conn:
                        conn.executemany(
                            "INSERT OR IGNORE INTO seen_hashes (code_sha256, record_id, partition) VALUES (?, ?, ?)",
                            shard_new_hashes,
                        )
                        conn.execute(
                            """
                            INSERT INTO completed_shards (
                                shard_name, completed_at, train_bytes, valid_bytes,
                                parent_files, candidate_count, retained_count,
                                duplicate_count, elapsed_seconds
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                shard_name, shard_dt, new_train_bytes, new_valid_bytes,
                                sum(shard_parents_scanned.values()), shard_candidates,
                                shard_retained, shard_duplicates, shard_elapsed,
                            ),
                        )
                        for repo, lang_counts in shard_unknown_repos.items():
                            conn.execute(
                                """
                                INSERT INTO unknown_repositories (repo, python_files, javascript_files)
                                VALUES (?, ?, ?)
                                ON CONFLICT(repo) DO UPDATE SET
                                    python_files = python_files + excluded.python_files,
                                    javascript_files = javascript_files + excluded.javascript_files
                                """,
                                (repo, lang_counts["python"], lang_counts["javascript"]),
                            )
                finally:
                    conn.close()

                # Update in-memory dedup state only after the SQLite shard commit.
                self.seen_hashes.update(shard_pending_hashes)

                shards_completed_this_run += 1
                completed_shards.add(shard_name)

                # Progress reporting
                elapsed_total = time.time() - start_time
                progress_dict = {
                    "last_completed_shard": shard_name,
                    "completed_shards_total": len(completed_shards),
                    "total_shards": total_discovered,
                    "percent_complete": round(len(completed_shards) / total_discovered * 100, 2),
                    "parent_files_scanned": dict(total_parents_scanned),
                    "candidate_occurrences": candidate_occurrences,
                    "retained_records": dict(retained_counts),
                    "duplicate_count": duplicate_count,
                    "quarantine_counts": dict(quarantine_counts),
                    "train_bytes": new_train_bytes,
                    "valid_bytes": new_valid_bytes,
                    "elapsed_seconds": round(elapsed_total, 2),
                    "updated_at": shard_dt,
                }
                self.progress_path.write_text(json.dumps(progress_dict, indent=2), encoding="utf-8")

                print(
                    f"[{len(completed_shards)}/{total_discovered}] Shard {shard_name} done in {shard_elapsed:.1f}s | "
                    f"Parents: {sum(shard_parents_scanned.values()):,} | "
                    f"Candidates: {shard_candidates:,} | "
                    f"Retained: {shard_retained:,} | "
                    f"Dupes: {shard_duplicates:,} | "
                    f"Quarantined: {quarantine_counts.get('timeout', 0) + quarantine_counts.get('worker_exception', 0)} | "
                    f"Total retained: {retained_counts['total']:,} | "
                    f"Elapsed: {elapsed_total/60:.1f}m",
                    flush=True,
                )

                if max_parents and total_parents_processed >= max_parents:
                    break

        finally:
            pool.terminate()
            pool.join()
            train_fh.close()
            valid_fh.close()
            quarantine_fh.close()

        end_time = time.time()
        end_dt = datetime.now(timezone.utc).isoformat()
        total_elapsed = end_time - start_time

        # Write unknown repositories summary
        with open(self.unknown_out_path, "w", encoding="utf-8") as f:
            for repo, c in sorted(unknown_repos_memory.items()):
                f.write(json.dumps({"repo": repo, "python_files": c["python"], "javascript_files": c["javascript"]}) + "\n")

        # Distribution helper
        def calc_distribution(hist: Counter) -> dict:
            total_items = sum(hist.values())
            if total_items == 0:
                return {"count": 0, "min": 0, "median": 0, "p90": 0, "p95": 0, "max": 0}
            sorted_lens = sorted(hist.keys())
            min_val = sorted_lens[0]
            max_val = sorted_lens[-1]

            def get_percentile(p: float) -> int:
                target = int(total_items * p)
                cumulative = 0
                for length in sorted_lens:
                    cumulative += hist[length]
                    if cumulative >= target:
                        return length
                return max_val

            return {
                "count": total_items,
                "min": min_val,
                "median": get_percentile(0.50),
                "p90": get_percentile(0.90),
                "p95": get_percentile(0.95),
                "max": max_val,
            }

        # Calculate above 512 and 1024
        def count_above(hist: Counter, threshold: int) -> int:
            return sum(count for length, count in hist.items() if length > threshold)

        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(DISTINCT repo) FROM unknown_repositories")
            total_unknown_repos = cur.fetchone()[0]
        finally:
            conn.close()

        top_20_repos = [
            {"repo": repo, "tokens": tokens}
            for repo, tokens in repo_token_counts.most_common(20)
        ]

        # File outputs and sha256
        output_files_info = {}
        for p in [self.train_out_path, self.valid_out_path, self.unknown_out_path,
                  self.quarantine_path, self.progress_path, self.db_path]:
            if p.exists():
                output_files_info[p.name] = {
                    "path": str(p),
                    "size_bytes": p.stat().st_size,
                    "sha256": sha256_file(p),
                }

        report = {
            "title": "Scenario C Production Extraction Report",
            "status": "COMPLETED" if (shards_completed_this_run + shards_skipped) == total_discovered else "PARTIAL",
            "shards": {
                "total_discovered": total_discovered,
                "completed_total": len(completed_shards),
                "completed_this_run": shards_completed_this_run,
                "skipped_already_completed": shards_skipped,
            },
            "parent_files_scanned_by_language": dict(total_parents_scanned),
            "parse_failures_by_language": dict(parse_failures),
            "candidate_occurrences": candidate_occurrences,
            "syntax_invalid_reconstructions": syntax_invalids,
            "duplicate_count": duplicate_count,
            "quarantine": {
                "total_quarantined": quarantine_counts.get("timeout", 0) + quarantine_counts.get("worker_exception", 0),
                "by_type": dict(quarantine_counts),
                "log_path": str(self.quarantine_path),
            },
            "retained_records": {
                "by_language": {
                    "python": retained_counts["python"],
                    "javascript": retained_counts["javascript"],
                },
                "by_partition": {
                    "train": retained_counts["train"],
                    "valid": retained_counts["valid"],
                },
                "by_category": {
                    "python": dict(retained_by_type["python"]),
                    "javascript": dict(retained_by_type["javascript"]),
                },
                "total": retained_counts["total"],
            },
            "repository_counts": {
                "scenario_b_train_repos_pool": len(self.train_repos),
                "scenario_b_valid_repos_pool": len(self.valid_repos),
                "unknown_repositories_encountered": total_unknown_repos,
            },
            "token_totals": {
                "by_language": {
                    "python": tokens_by_lang_part["python"],
                    "javascript": tokens_by_lang_part["javascript"],
                },
                "by_partition": {
                    "train": tokens_by_lang_part["train"],
                    "valid": tokens_by_lang_part["valid"],
                },
                "by_language_and_partition": {
                    "python_train": tokens_by_lang_part["python_train"],
                    "python_valid": tokens_by_lang_part["python_valid"],
                    "javascript_train": tokens_by_lang_part["javascript_train"],
                    "javascript_valid": tokens_by_lang_part["javascript_valid"],
                },
                "total_tokens": tokens_by_lang_part["total"],
            },
            "token_distribution": {
                "overall": calc_distribution(token_histogram["overall"]),
                "python": calc_distribution(token_histogram["python"]),
                "javascript": calc_distribution(token_histogram["javascript"]),
            },
            "token_thresholds": {
                "above_512_overall": count_above(token_histogram["overall"], 512),
                "above_1024_overall": count_above(token_histogram["overall"], 1024),
                "above_512_python": count_above(token_histogram["python"], 512),
                "above_1024_python": count_above(token_histogram["python"], 1024),
                "above_512_javascript": count_above(token_histogram["javascript"], 512),
                "above_1024_javascript": count_above(token_histogram["javascript"], 1024),
            },
            "missing_provenance_count": missing_provenance_count,
            "top_20_repositories_by_tokens": top_20_repos,
            "dependency_flag_distribution": {
                "python": dict(dependency_flag_counts["python"]),
                "javascript": dict(dependency_flag_counts["javascript"]),
            },
            "timing_and_resume": {
                "start_time": start_dt,
                "end_time": end_dt,
                "elapsed_seconds": round(total_elapsed, 2),
                "elapsed_formatted": f"{int(total_elapsed // 60)}m {int(total_elapsed % 60)}s",
                "workers": self.workers,
                "file_timeout_seconds": self.file_timeout,
            },
            "output_files": output_files_info,
        }

        self.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("\nExtraction report saved to:", self.report_path, flush=True)

        return report


def main():
    parser = argparse.ArgumentParser(description="Scenario C Production Extraction Pipeline")
    parser.add_argument("--shard-dir", type=Path, default=DEFAULT_SHARD_DIR, help="Path to CodeParrot parquet shards")
    parser.add_argument("--scenario-b-dir", type=Path, default=DEFAULT_SCENARIO_B, help="Path to Scenario B train/valid jsonl")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory")
    parser.add_argument("--shard-limit", type=int, default=None, help="Max shards to process")
    parser.add_argument("--shard-offset", type=int, default=0, help="Shard start offset")
    parser.add_argument("--max-parents", type=int, default=None, help="Max parent files to process across shards")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4), help="Number of worker processes")
    parser.add_argument("--chunksize", type=int, default=1, help="Batch chunksize for pool dispatch")
    parser.add_argument("--file-timeout", type=int, default=FILE_TIMEOUT_SECONDS, help="Per-file extraction timeout in seconds")

    args = parser.parse_args()

    extractor = ScenarioCExtractor(
        shard_dir=args.shard_dir,
        scenario_b_dir=args.scenario_b_dir,
        output_dir=args.output_dir,
        workers=args.workers,
        chunksize=args.chunksize,
        file_timeout=args.file_timeout,
    )

    report = extractor.run(
        shard_limit=args.shard_limit,
        shard_offset=args.shard_offset,
        max_parents=args.max_parents,
    )

    print("\n========== EXTRACTION SUMMARY ==========")
    print(f"Shards completed total: {report['shards']['completed_total']}/{report['shards']['total_discovered']}")
    print(f"Retained records: {report['retained_records']['total']:,}")
    print(f"Total tokens: {report['token_totals']['total_tokens']:,}")
    print(f"Duplicates removed: {report['duplicate_count']:,}")
    print(f"Quarantined files: {report['quarantine']['total_quarantined']}")
    print(f"Elapsed time: {report['timing_and_resume']['elapsed_formatted']}")


if __name__ == "__main__":
    main()
