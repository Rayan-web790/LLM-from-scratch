"""Train and apply the Phase 18 v2 tokenizer (24k ByteLevel BPE)."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from array import array
from pathlib import Path
from typing import Iterator

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from .constants import SCHEMA_VERSION, SPECIAL_TOKENS
from .schema import assert_no_forbidden_input

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "polycodegpt_v2"
MANIFEST_ROOT = PROJECT_ROOT / "data" / "manifests" / "v2"
TRAIN_JSONL = PROCESSED_ROOT / "train.jsonl"
VALID_JSONL = PROCESSED_ROOT / "valid.jsonl"
ARTIFACT_DIR = PROJECT_ROOT / "tokenizer" / "artifacts" / "v2"
TOKENIZER_JSON = ARTIFACT_DIR / "polycodegpt_tokenizer_v2.json"
METADATA_JSON = ARTIFACT_DIR / "tokenizer_metadata_v2.json"
TOKEN_COUNTS = MANIFEST_ROOT / "token_counts.csv"
TRAIN_BIN = PROCESSED_ROOT / "train.bin"
VALID_BIN = PROCESSED_ROOT / "valid.bin"


def _assert_train_only(path: Path) -> None:
    assert_no_forbidden_input(str(path))
    if path.name in {"test.jsonl"}:
        raise RuntimeError(f"refusing evaluation input {path}")


def build_tokenizer(vocab_size: int, min_frequency: int) -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens(list(SPECIAL_TOKENS))
    return tok


def build_trainer(vocab_size: int, min_frequency: int) -> trainers.BpeTrainer:
    return trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )


def iter_texts(path: Path) -> Iterator[str]:
    _assert_train_only(path)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)["text"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def train_tokenizer(vocab_size: int = 24000, min_frequency: int = 2, verbose: bool = True) -> dict:
    _assert_train_only(TRAIN_JSONL)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    train_count = sum(1 for _ in iter_texts(TRAIN_JSONL))
    if verbose:
        print(f"training tokenizer on {train_count:,} training records")
    tok = build_tokenizer(vocab_size, min_frequency)
    trainer = build_trainer(vocab_size, min_frequency)
    t0 = time.perf_counter()
    tok.train_from_iterator(iter_texts(TRAIN_JSONL), trainer=trainer, length=train_count)
    tok.add_special_tokens(list(SPECIAL_TOKENS))
    fit_s = time.perf_counter() - t0
    tok.save(str(TOKENIZER_JSON))
    raw = json.loads(TOKENIZER_JSON.read_text(encoding="utf-8"))
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "tokenizer_version": "2.0",
        "algorithm": "ByteLevel BPE",
        "vocab_size_requested": vocab_size,
        "vocab_size_actual": len(raw["model"]["vocab"]),
        "merge_count": len(raw["model"]["merges"]),
        "added_token_count": len(raw["added_tokens"]),
        "min_frequency": min_frequency,
        "special_tokens": list(SPECIAL_TOKENS),
        "special_token_ids": {t: tok.token_to_id(t) for t in SPECIAL_TOKENS},
        "train_records": train_count,
        "train_file": TRAIN_JSONL.relative_to(PROJECT_ROOT).as_posix(),
        "train_sha256": sha256_file(TRAIN_JSONL),
        "fit_seconds": round(fit_s, 2),
        "artifact": TOKENIZER_JSON.relative_to(PROJECT_ROOT).as_posix(),
        "artifact_sha256": sha256_file(TOKENIZER_JSON),
    }
    METADATA_JSON.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def tokenize_corpus(verbose: bool = True) -> dict:
    _assert_train_only(TRAIN_JSONL)
    _assert_train_only(VALID_JSONL)
    tok = Tokenizer.from_file(str(TOKENIZER_JSON))
    for special in SPECIAL_TOKENS:
        if tok.token_to_id(special) is None:
            tok.add_special_tokens([special])
    counts: dict[str, int] = {}
    rows = []
    for path, bin_path in ((TRAIN_JSONL, TRAIN_BIN), (VALID_JSONL, VALID_BIN)):
        _assert_train_only(path)
        total = 0
        buffer = array("H")
        flush_size = 10_000_000
        tmp = path.with_suffix(".jsonl.tmp")
        batch_size = 512
        with open(path, encoding="utf-8") as fh, open(
            tmp, "w", encoding="utf-8", newline="\n"
        ) as json_out, open(bin_path, "wb") as out:
            while True:
                batch = []
                for _ in range(batch_size):
                    line = fh.readline()
                    if not line:
                        break
                    if line.strip():
                        batch.append(json.loads(line))
                if not batch:
                    break
                encodings = tok.encode_batch([rec["text"] for rec in batch])
                for rec, encoding in zip(batch, encodings):
                    ids = encoding.ids
                    if any(i >= 65536 for i in ids):
                        raise RuntimeError(f"token id exceeds uint16 range in {path}")
                    buffer.extend(ids)
                    total += len(ids)
                    rec["token_count"] = len(ids)
                    rows.append(
                        (
                            rec["record_id"],
                            rec["token_count"],
                            rec["source_id"],
                            rec["language"],
                        )
                    )
                    json_out.write(
                        json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n"
                    )
                if len(buffer) >= flush_size:
                    out.write(buffer.tobytes())
                    buffer = array("H")
            if buffer:
                out.write(buffer.tobytes())
        tmp.replace(path)
        counts[path.name] = total
        if verbose:
            print(f"{path.name}: {total:,} tokens -> {bin_path}")
    with open(TOKEN_COUNTS, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["record_id", "token_count", "source_id", "language"])
        writer.writerows(rows)
    # Update split report token totals.
    split_path = MANIFEST_ROOT / "split_report.json"
    if split_path.is_file():
        report = json.loads(split_path.read_text(encoding="utf-8"))
        report["split"]["tokens"] = {
            "train": counts["train.jsonl"],
            "validation": counts["valid.jsonl"],
        }
        report["token_counts"] = counts
        report["tokenizer"] = {
            "artifact": TOKENIZER_JSON.relative_to(PROJECT_ROOT).as_posix(),
            "vocab_size": len(tok.get_vocab()),
        }
        split_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return counts


def main_cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="PolyCodeGPT Phase 18 v2 tokenizer")
    sub = parser.add_subparsers(dest="command", required=True)
    p_train = sub.add_parser("train")
    p_train.add_argument("--vocab-size", type=int, default=24000)
    p_train.add_argument("--min-frequency", type=int, default=2)
    sub.add_parser("tokenize")
    args = parser.parse_args(argv)
    if args.command == "train":
        result = train_tokenizer(args.vocab_size, args.min_frequency)
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        result = tokenize_corpus()
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
