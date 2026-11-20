#!/usr/bin/env python
"""Sequential orchestrator for Scenario C extraction (shards 20..79)."""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PY = PROJECT_ROOT / ".venv_scenario_c" / "Scripts" / "python.exe"
SCRIPT = PROJECT_ROOT / "scripts" / "build_scenario_c.py"
DB = PROJECT_ROOT / "data" / "interim" / "scenario_c" / "dedup.sqlite"
OUT = PROJECT_ROOT / "data" / "interim" / "scenario_c"
SHARD_DIR = PROJECT_ROOT / "data" / "raw" / "v2" / "codeparrot_github_code"
LOG_DIR = PROJECT_ROOT / "logs" / "scenario_c_continue"
PROGRESS_LOG = LOG_DIR / "progress.jsonl"
FAILURE_MARKER = LOG_DIR / "FAILED.txt"
DONE_MARKER = LOG_DIR / "DONE.txt"

LOG_DIR.mkdir(parents=True, exist_ok=True)
shards = sorted(SHARD_DIR.glob("*.parquet"))
assert len(shards) == 80, len(shards)


def db_state():
    con = sqlite3.connect(DB)
    try:
        cur = con.cursor()
        cnt = cur.execute("SELECT COUNT(*) FROM completed_shards").fetchone()[0]
        last = cur.execute(
            "SELECT shard_name, train_bytes, valid_bytes FROM completed_shards ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        hashes = cur.execute("SELECT COUNT(*) FROM seen_hashes").fetchone()[0]
        return cnt, last, hashes
    finally:
        con.close()


def main():
    if FAILURE_MARKER.exists():
        FAILURE_MARKER.unlink()
    if DONE_MARKER.exists():
        DONE_MARKER.unlink()

    cnt0, last0, hashes0 = db_state()
    print(f"Starting: completed={cnt0}, last={last0}, hashes={hashes0}", flush=True)

    for offset in range(20, 80):
        expected_name = shards[offset].name

        con = sqlite3.connect(DB)
        already = con.execute(
            "SELECT 1 FROM completed_shards WHERE shard_name=?", (expected_name,)
        ).fetchone()
        con.close()
        if already:
            print(f"Offset {offset} ({expected_name}) already completed; skipping.", flush=True)
            continue

        before_count, before_last, before_hashes = db_state()
        before_train = before_last[1] if before_last else 0
        before_valid = before_last[2] if before_last else 0

        print(f"\n===== OFFSET {offset}: {expected_name} =====", flush=True)
        t0 = time.time()
        log_path = LOG_DIR / f"shard_{offset:03d}.log"
        cmd = [
            str(PY), str(SCRIPT),
            "--shard-offset", str(offset),
            "--shard-limit", "1",
            "--workers", "8",
            "--file-timeout", "300",
        ]
        with open(log_path, "w", encoding="utf-8") as lf:
            proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=str(PROJECT_ROOT))
        elapsed = time.time() - t0
        exit_code = proc.returncode
        print(f"Exit code: {exit_code}, elapsed {elapsed:.1f}s", flush=True)

        if exit_code != 0:
            msg = f"Offset {offset} ({expected_name}) failed with exit code {exit_code}. See {log_path}"
            print("FAILURE:", msg, flush=True)
            FAILURE_MARKER.write_text(msg, encoding="utf-8")
            return 1

        # verify
        cnt, last, hashes = db_state()
        errors = []
        if last is None or last[0] != expected_name:
            errors.append(f"latest shard {last} != expected {expected_name}")
        if cnt != before_count + 1:
            errors.append(f"completed count {cnt} != {before_count}+1")
        train_size = OUT.joinpath("train_functions.jsonl").stat().st_size
        valid_size = OUT.joinpath("valid_functions.jsonl").stat().st_size
        if last is not None:
            if train_size != last[1]:
                errors.append(f"train size {train_size} != committed {last[1]}")
            if valid_size != last[2]:
                errors.append(f"valid size {valid_size} != committed {last[2]}")
        if errors:
            msg = f"Offset {offset} ({expected_name}) verification failed: {errors}. See {log_path}"
            print("FAILURE:", msg, flush=True)
            FAILURE_MARKER.write_text(msg, encoding="utf-8")
            return 1

        # archive report
        try:
            rep_path = OUT / "extraction_report.json"
            if rep_path.exists():
                archive = LOG_DIR / f"shard_{offset:03d}_report.json"
                archive.write_text(rep_path.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception as e:
            print("warn: could not archive report:", e, flush=True)

        con = sqlite3.connect(DB)
        row = con.execute(
            "SELECT shard_name,parent_files,candidate_count,retained_count,duplicate_count,elapsed_seconds,completed_at,train_bytes,valid_bytes FROM completed_shards WHERE shard_name=?",
            (expected_name,),
        ).fetchone()
        con.close()

        entry = {
            "offset": offset,
            "shard": expected_name,
            "completed_count": cnt,
            "unique_hashes": hashes,
            "parent_files": row[1],
            "candidate_count": row[2],
            "retained_count": row[3],
            "duplicate_count": row[4],
            "elapsed_seconds": row[5],
            "completed_at": row[6],
            "train_bytes": row[7],
            "valid_bytes": row[8],
            "verification": "PASS",
            "wall_elapsed_seconds": round(elapsed, 2),
        }
        with open(PROGRESS_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        print(
            f"OK: {expected_name} | parents={row[1]} candidates={row[2]} retained={row[3]} "
            f"dupes={row[4]} | completed={cnt}/80 hashes={hashes}",
            flush=True,
        )

    DONE_MARKER.write_text("All offsets 20..79 completed successfully.\n", encoding="utf-8")
    print("\nALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
