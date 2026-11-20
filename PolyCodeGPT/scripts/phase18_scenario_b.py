#!/usr/bin/env python
"""Create the deterministic 500M-token Scenario B cap from the full v2 pool."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from array import array
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tokenizers import Tokenizer  # noqa: E402

from polycodegpt.data_v2.constants import SPECIAL_TOKENS  # noqa: E402

PROCESSED = ROOT / "data" / "processed" / "polycodegpt_v2"
MANIFESTS = ROOT / "data" / "manifests" / "v2"
TOKENIZER = ROOT / "tokenizer" / "artifacts" / "v2" / "polycodegpt_tokenizer_v2.json"
OUT = PROCESSED / "scenario_b"
CAPS = {"python": 250_000_000, "javascript": 225_000_000, "en": 25_000_000}
SEED = "polycodegpt-v2-scenario-b-500m"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def priority(record_id: str) -> bytes:
    return hashlib.sha256(f"{SEED}|{record_id}".encode("utf-8")).digest()


def main() -> int:
    train = PROCESSED / "train.jsonl"
    valid = PROCESSED / "valid.jsonl"
    for path in (train, valid):
        if path.name == "test.jsonl":
            raise RuntimeError("refusing test partition")
    OUT.mkdir(parents=True, exist_ok=True)

    candidates: dict[str, list[tuple[bytes, int, str]]] = defaultdict(list)
    total_by_lang = Counter()
    with open(train, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            lang = rec["language"]
            n = int(rec["token_count"])
            total_by_lang[lang] += n
            if lang in CAPS:
                candidates[lang].append((priority(rec["record_id"]), n, rec["record_id"]))

    selected: set[str] = set()
    selected_tokens = Counter()
    for lang, cap in CAPS.items():
        items = sorted(candidates.get(lang, []))
        chosen = 0
        for _, n, rid in items:
            if chosen + n > cap:
                continue
            selected.add(rid)
            chosen += n
        selected_tokens[lang] = chosen

    selected_records = 0
    by_source = Counter()
    by_language = Counter()
    by_category = Counter()
    alignment = Counter()
    with open(train, encoding="utf-8") as fh, open(
        OUT / "train.jsonl", "w", encoding="utf-8", newline="\n"
    ) as out:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec["record_id"] not in selected:
                continue
            out.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
            selected_records += 1
            by_source[f"{rec['source_id']}|{rec['language']}"] += 1
            by_language[rec["language"]] += 1
            by_category[f"{rec['language']}|{rec['category']}"] += 1
            alignment[rec.get("alignment_status", "not_applicable")] += 1

    shutil.copyfile(valid, OUT / "valid.jsonl")
    shutil.copyfile(PROCESSED / "valid.bin", OUT / "valid.bin")

    tok = Tokenizer.from_file(str(TOKENIZER))
    for special in SPECIAL_TOKENS:
        if tok.token_to_id(special) is None:
            tok.add_special_tokens([special])
    total = 0
    buffer = array("H")
    with open(OUT / "train.jsonl", encoding="utf-8") as fh, open(
        OUT / "train.bin", "wb"
    ) as out:
        while True:
            batch = []
            for _ in range(512):
                line = fh.readline()
                if not line:
                    break
                if line.strip():
                    batch.append(json.loads(line))
            if not batch:
                break
            for enc in tok.encode_batch([r["text"] for r in batch]):
                buffer.extend(enc.ids)
                total += len(enc.ids)
            if len(buffer) >= 10_000_000:
                out.write(buffer.tobytes())
                buffer = array("H")
        if buffer:
            out.write(buffer.tobytes())

    manifest = {
        "scenario": "B",
        "status": "CAP_APPLIED",
        "seed": SEED,
        "selection": "ascending sha256(seed|record_id) within language, whole records only",
        "target_tokens": CAPS,
        "selected_tokens": dict(selected_tokens),
        "total_selected_tokens": total,
        "full_pool_tokens": dict(total_by_lang),
        "full_pool_total_tokens": sum(total_by_lang.values()),
        "records": selected_records,
        "records_by_source": dict(by_source),
        "records_by_language": dict(by_language),
        "records_by_category": dict(by_category),
        "alignment": dict(alignment),
        "files": {
            "train.jsonl": sha256_file(OUT / "train.jsonl"),
            "train.bin": sha256_file(OUT / "train.bin"),
            "valid.jsonl": sha256_file(OUT / "valid.jsonl"),
            "valid.bin": sha256_file(OUT / "valid.bin"),
        },
        "full_pool_hashes": {
            "train.jsonl": sha256_file(train),
            "valid.jsonl": sha256_file(valid),
        },
        "test_partition_read": False,
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (MANIFESTS / "scenario_b_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
