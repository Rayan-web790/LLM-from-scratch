"""Build binary token streams and a preparation manifest from the approved corpus.

Pipeline per split:
    source record -> serializer v1.1 -> tokenizer v1.1 -> buffered uint16 stream

Nothing here trains, fits, or mutates the tokenizer; the only structural markers
in the stream are the ones the serializer emits, and every example is checked to
start with a language marker and end with exactly one EOS.

Texts are encoded in small batches because ``Tokenizer.encode_batch`` runs the
Rust BPE encoder in parallel and is ~7x faster than per-record ``encode`` on this
machine, while memory stays bounded by ``encode_batch_size`` examples.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np
import tokenizers
import torch

from ..model.config import ModelConfig, PROJECT_ROOT
from ..tokenization import serializer
from .integrity import sha256_file
from .stream import (
    BITS_PER_TOKEN,
    BYTE_ORDER,
    DTYPE_NAME,
    TokenStreamWriter,
    block_geometry,
    dtype_is_safe,
)
from .tokenizer_io import (
    EOS_ID,
    LANG_EN_ID,
    LANG_JAVASCRIPT_ID,
    LANG_MARKER_IDS,
    LANG_PYTHON_ID,
    SPECIAL_TOKEN_IDS,
    TOKENIZER_SHA256,
    TOKENIZER_VERSION,
    TOKENIZER_PATH,
    load_tokenizer,
)

MANIFEST_VERSION = "1.1.0"
DATASET_VERSION = "tokenized_v1_1"

CORPUS_DIR = PROJECT_ROOT / "data" / "processed" / "polycodegpt_corpus"
DOCS_INVENTORY = PROJECT_ROOT / "data" / "manifests" / "python_docs_training_corpus_inventory.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / DATASET_VERSION

#: Measured SHA-256 of each combined corpus partition (data/manifests/polycodegpt_corpus_manifest.csv).
EXPECTED_SPLIT_SHA256: Mapping[str, str] = {
    "train": "2a9c71607a1893711cd7f2963ec172784415f838ec1c633c988949b45e345c74",
    "valid": "2eb02406d22101716214a6509314e9d7e1596e1cc6640d0eb4a886c00fb715af",
    "test": "93f624f6e1f24d8da8d3855fbf407b205671e1f5f1f29ab88914fff16b4f3c2e",
}

DEFAULT_ENCODE_BATCH = 256


class PreparationError(RuntimeError):
    """Raised when a source violates an integrity or structural contract."""


# --------------------------------------------------------------------------- #
# per-example validation
# --------------------------------------------------------------------------- #
def validate_example(ids: Sequence[int], vocab_size: int, where: str) -> None:
    """Enforce the stream contract for one serialized example.

    * every ID is inside ``[0, vocab_size)``
    * the example begins with a language marker
    * the example ends with exactly one EOS and no other EOS
    """
    if not ids:
        raise PreparationError(f"{where}: tokenized to an empty sequence")

    highest = max(ids)
    if highest >= vocab_size:
        raise PreparationError(f"{where}: token ID {highest} outside vocabulary [0, {vocab_size})")
    if min(ids) < 0:
        raise PreparationError(f"{where}: negative token ID {min(ids)}")

    if ids[0] not in LANG_MARKER_IDS:
        raise PreparationError(f"{where}: first token {ids[0]} is not a language marker")
    if ids[-1] != EOS_ID:
        raise PreparationError(f"{where}: last token {ids[-1]} is not EOS ({EOS_ID})")

    eos_count = sum(1 for t in ids if t == EOS_ID)
    if eos_count != 1:
        raise PreparationError(f"{where}: expected exactly one EOS, found {eos_count}")


def _source_sha(path: Path, expected: str | None) -> str:
    actual = sha256_file(path)
    if expected is not None and actual != expected:
        raise PreparationError(
            f"source SHA-256 mismatch for {path}: {actual} != expected {expected}. "
            "Refusing to proceed; sources are never repaired or rewritten."
        )
    return actual


# --------------------------------------------------------------------------- #
# batched encode + write
# --------------------------------------------------------------------------- #
def _encode_batch(tokenizer, texts: list[str]) -> list[list[int] | None]:
    """Encode a list of texts, preserving order; ``None`` marks a failed encode."""
    try:
        return [enc.ids for enc in tokenizer.encode_batch(texts, add_special_tokens=False)]
    except Exception:
        results: list[list[int] | None] = []
        for text in texts:
            try:
                results.append(tokenizer.encode(text, add_special_tokens=False).ids)
            except Exception:
                results.append(None)
        return results


def _write_batched(
    writer: TokenStreamWriter,
    tokenizer,
    vocab_size: int,
    items: Iterable[tuple[dict[str, Any], str]],
    failures: list[dict[str, Any]],
    split: str,
    batch_size: int = DEFAULT_ENCODE_BATCH,
    progress_every: int = 50_000,
    label: str = "",
) -> int:
    """Encode ``(meta, text)`` items in batches and append them to the stream."""
    texts: list[str] = []
    metas: list[dict[str, Any]] = []
    written = 0
    seen = 0
    started = time.perf_counter()

    def flush() -> int:
        nonlocal written
        if not texts:
            return 0
        encodings = _encode_batch(tokenizer, texts)
        batch_written = 0
        for meta, ids in zip(metas, encodings):
            where = f"{split}:{meta.get('record_id', '?')}"
            try:
                if ids is None:
                    raise PreparationError("tokenizer.encode failed")
                validate_example(ids, vocab_size, where)
                if meta.get("expected_first_id") is not None and ids[0] != meta["expected_first_id"]:
                    raise PreparationError(
                        f"{where}: first token {ids[0]} != expected {meta['expected_first_id']}"
                    )
            except Exception as exc:
                failures.append(
                    {
                        "split": split,
                        "source": meta.get("source", ""),
                        "record_index": meta.get("index"),
                        "record_id": meta.get("record_id", ""),
                        "language": meta.get("language"),
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            writer.write(ids)
            batch_written += 1
        written += batch_written
        texts.clear()
        metas.clear()
        return batch_written

    for meta, text in items:
        texts.append(text)
        metas.append(meta)
        seen += 1
        if len(texts) >= batch_size:
            flush()
        if progress_every and seen % progress_every == 0:
            elapsed = time.perf_counter() - started
            print(
                f"    {label}{split}: {seen:,} examples, {written:,} written, "
                f"{elapsed:5.1f}s, {seen / max(elapsed, 1e-9):7.0f} ex/s",
                flush=True,
            )
    flush()
    return written


# --------------------------------------------------------------------------- #
# split preparation
# --------------------------------------------------------------------------- #
def _code_items(
    name: str,
    code_jsonl: Path,
    failures: list[dict[str, Any]],
    languages: set[int] | None = None,
) -> Iterator[tuple[dict[str, Any], str]]:
    """Yield ``(meta, serialized_text)`` for each record, recording serialization errors."""
    for index, record in enumerate(serializer.iter_jsonl(code_jsonl)):
        record_id = f"{record.get('repo')}:{record.get('path')}"
        meta = {
            "index": index,
            "record_id": record_id,
            "language": record.get("language"),
            "source": str(code_jsonl),
        }
        try:
            text = serializer.serialize_code_record(record)
        except Exception as exc:
            failures.append(
                {
                    "split": name,
                    "source": str(code_jsonl),
                    "record_index": index,
                    "record_id": record_id,
                    "language": record.get("language"),
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        yield meta, text


def prepare_code_split(
    name: str,
    code_jsonl: str | Path,
    tokenizer,
    vocab_size: int,
    output_path: str | Path,
    context_length: int,
    expected_sha256: str | None = None,
    buffer_tokens: int = 1 << 20,
    batch_size: int = DEFAULT_ENCODE_BATCH,
    progress_every: int = 50_000,
) -> dict[str, Any]:
    """Tokenize one combined corpus partition into a ``.bin`` stream."""
    code_jsonl = Path(code_jsonl)
    if not code_jsonl.is_file():
        raise FileNotFoundError(f"{name}: corpus file missing: {code_jsonl}")

    source_sha = _source_sha(code_jsonl, expected_sha256)
    failures: list[dict[str, Any]] = []
    started = time.perf_counter()

    with TokenStreamWriter(output_path, vocab_size, buffer_tokens=buffer_tokens) as writer:
        records = _write_batched(
            writer, tokenizer, vocab_size,
            _code_items(name, code_jsonl, failures),
            failures, name, batch_size=batch_size, progress_every=progress_every,
        )
        stats = writer.close()

    geometry = block_geometry(stats["tokens"], context_length)
    return {
        "file": Path(output_path).name,
        "path": str(Path(output_path)),
        "tokens": stats["tokens"],
        "file_size_bytes": stats["file_size_bytes"],
        "sha256": stats["sha256"],
        "max_token_id": stats["max_token_id"],
        "records": records,
        "documents": 0,
        "blocks": geometry["blocks"],
        "dropped_trailing_tokens": geometry["dropped_trailing_tokens"],
        "sources": [
            {
                "split": name,
                "kind": "code_records",
                "path": str(code_jsonl),
                "sha256": source_sha,
                "records": records,
                "partition": name,
            }
        ],
        "failures": failures,
        "elapsed_s": round(time.perf_counter() - started, 2),
    }


def _document_items(
    inventory_path: Path,
    failures: list[dict[str, Any]],
) -> Iterator[tuple[dict[str, Any], str]]:
    """Yield ``(meta, serialized_text)`` for each approved documentation file."""
    docs = serializer.load_doc_inventory(inventory_path, PROJECT_ROOT)
    if not docs:
        raise PreparationError(f"no eligible documentation files in {inventory_path}")

    for index, doc in enumerate(docs):
        path = Path(doc["local_path"])
        record_id = doc["rel_path"]
        meta = {
            "index": index,
            "record_id": record_id,
            "language": "en",
            "source": str(inventory_path),
            "expected_first_id": LANG_EN_ID,
        }
        try:
            if not path.is_file():
                raise PreparationError(f"file missing: {path}")
            raw_sha = sha256_file(path)
            if raw_sha != doc["sha256"]:
                raise PreparationError(f"SHA-256 mismatch {raw_sha} != {doc['sha256']}")
            text = serializer.serialize_documentation(path.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append(
                {
                    "split": "train",
                    "source": str(inventory_path),
                    "record_index": index,
                    "record_id": record_id,
                    "language": "en",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        yield meta, text


def prepare_documents(
    tokenizer,
    vocab_size: int,
    output_path: str | Path,
    context_length: int,
    inventory_path: str | Path = DOCS_INVENTORY,
    buffer_tokens: int = 1 << 20,
    batch_size: int = DEFAULT_ENCODE_BATCH,
    progress_every: int = 0,
) -> dict[str, Any]:
    """Tokenize the approved documentation files into a ``.bin`` stream."""
    inventory_path = Path(inventory_path)
    failures: list[dict[str, Any]] = []
    started = time.perf_counter()

    with TokenStreamWriter(output_path, vocab_size, buffer_tokens=buffer_tokens) as writer:
        included = _write_batched(
            writer, tokenizer, vocab_size,
            _document_items(inventory_path, failures),
            failures, "train", batch_size=batch_size, progress_every=progress_every,
            label="docs/",
        )
        stats = writer.close()

    geometry = block_geometry(stats["tokens"], context_length)
    return {
        "file": Path(output_path).name,
        "path": str(Path(output_path)),
        "tokens": stats["tokens"],
        "file_size_bytes": stats["file_size_bytes"],
        "sha256": stats["sha256"],
        "max_token_id": stats["max_token_id"],
        "records": 0,
        "documents": included,
        "blocks": geometry["blocks"],
        "dropped_trailing_tokens": geometry["dropped_trailing_tokens"],
        "sources": [
            {
                "split": "train",
                "kind": "documentation",
                "path": str(inventory_path),
                "sha256": sha256_file(inventory_path),
                "records": included,
                "partition": "train_only",
            }
        ],
        "failures": failures,
        "elapsed_s": round(time.perf_counter() - started, 2),
    }


def _iter_stream_chunks(path: Path, chunk: int = 1 << 20) -> Iterator[list[int]]:
    """Yield chunks of a stream file without loading it fully."""
    mm = np.memmap(path, dtype="<u2", mode="r")
    try:
        total = int(mm.shape[0])
        for start in range(0, total, chunk):
            yield mm[start : start + chunk].tolist()
    finally:
        del mm


def concatenate_streams(parts: Iterable[Path], output_path: str | Path, vocab_size: int) -> dict[str, Any]:
    """Concatenate binary streams into one without loading them fully into memory."""
    output_path = Path(output_path)
    with TokenStreamWriter(output_path, vocab_size, buffer_tokens=1 << 20) as writer:
        for part in parts:
            for chunk in _iter_stream_chunks(Path(part)):
                writer.write(chunk)
        return writer.close()


# --------------------------------------------------------------------------- #
# full preparation
# --------------------------------------------------------------------------- #
def software_versions() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "tokenizers": tokenizers.__version__,
        "numpy": np.__version__,
    }


def prepare_dataset(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    config_path: str | Path | None = None,
    corpus_dir: str | Path = CORPUS_DIR,
    docs_inventory: str | Path = DOCS_INVENTORY,
    tokenizer_path: str | Path | None = None,
    verify_sources: bool = True,
    buffer_tokens: int = 1 << 20,
    batch_size: int = DEFAULT_ENCODE_BATCH,
    progress_every: int = 50_000,
    include_documents: bool = True,
) -> dict[str, Any]:
    """Build train/valid/test token streams and write ``manifest.json``."""
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    start = time.perf_counter()

    config = ModelConfig.from_yaml(config_path) if config_path else ModelConfig.from_yaml()
    if not dtype_is_safe(config.vocab_size):
        raise PreparationError(f"vocab_size {config.vocab_size} does not fit in {DTYPE_NAME}")

    tokenizer = load_tokenizer(
        tokenizer_path or TOKENIZER_PATH, expected_sha256=TOKENIZER_SHA256, vocab_size=config.vocab_size
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir = Path(corpus_dir)
    expected = EXPECTED_SPLIT_SHA256 if verify_sources else {}

    train_code = prepare_code_split(
        "train", corpus_dir / "train.jsonl", tokenizer, config.vocab_size,
        output_dir / "_train_code.bin", config.context_length,
        expected_sha256=expected.get("train"), buffer_tokens=buffer_tokens,
        batch_size=batch_size, progress_every=progress_every,
    )
    print(f"  train code:    {train_code['tokens']:,} tokens, {train_code['records']:,} records", flush=True)

    sources = list(train_code["sources"])
    failures = list(train_code["failures"])
    documents = 0
    doc_elapsed = 0.0
    doc_parts: list[Path] = []

    if include_documents:
        docs = prepare_documents(
            tokenizer, config.vocab_size, output_dir / "_train_docs.bin", config.context_length,
            inventory_path=docs_inventory, buffer_tokens=buffer_tokens,
            batch_size=batch_size, progress_every=progress_every,
        )
        doc_parts = [Path(docs["path"])]
        sources.extend(docs["sources"])
        failures.extend(docs["failures"])
        documents = docs["documents"]
        doc_elapsed = docs["elapsed_s"]
        print(f"  train docs:    {docs['tokens']:,} tokens, {documents:,} documents", flush=True)

    # Final train.bin keeps the fitting order: code examples first, then documents.
    merged = concatenate_streams(
        [output_dir / "_train_code.bin", *doc_parts], output_dir / "train.bin", config.vocab_size
    )
    (output_dir / "_train_code.bin").unlink(missing_ok=True)
    for part in doc_parts:
        part.unlink(missing_ok=True)
    train_code.update(merged)
    train_code["path"] = str(output_dir / "train.bin")

    train_geometry = block_geometry(train_code["tokens"], config.context_length)
    train_split = {
        **train_code,
        "records": train_code["records"],
        "documents": documents,
        "blocks": train_geometry["blocks"],
        "dropped_trailing_tokens": train_geometry["dropped_trailing_tokens"],
        "sources": sources,
        "failures": failures,
        "elapsed_s": round(train_code["elapsed_s"] + doc_elapsed, 2),
    }

    splits: dict[str, Any] = {"train": train_split}
    for name in ("valid", "test"):
        split = prepare_code_split(
            name, corpus_dir / f"{name}.jsonl", tokenizer, config.vocab_size,
            output_dir / f"{name}.bin", config.context_length,
            expected_sha256=expected.get(name), buffer_tokens=buffer_tokens,
            batch_size=batch_size, progress_every=progress_every,
        )
        print(f"  {name}:         {split['tokens']:,} tokens, {split['records']:,} records", flush=True)
        splits[name] = split

    all_failures = [f for split in splits.values() for f in split["failures"]]
    total_tokens = sum(s["tokens"] for s in splits.values())
    total_blocks = sum(s["blocks"] for s in splits.values())

    manifest: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "dataset_version": DATASET_VERSION,
        "completed_successfully": not all_failures,
        "output_dir": str(output_dir),
        "context_length": config.context_length,
        "window_length": config.context_length + 1,
        "window_stride": config.context_length,
        "window_overlap": 1,
        "block_span": config.context_length + 1,
        "block_scheme": "overlapping",
        "block_indexing": "start = i * window_stride, tokens[start:start + window_length]",
        "shift": "input_ids = block[:-1], targets = block[1:]",
        "dtype": DTYPE_NAME,
        "byte_order": BYTE_ORDER,
        "bits_per_token": BITS_PER_TOKEN,
        "examples_never_split": True,
        "blocks_may_cross_example_boundaries": True,
        "tokenizer": {
            "version": TOKENIZER_VERSION,
            "path": str(tokenizer_path or TOKENIZER_PATH),
            "sha256": TOKENIZER_SHA256,
            "vocab_size": config.vocab_size,
            "special_tokens": dict(SPECIAL_TOKEN_IDS),
            "eos_token_id": EOS_ID,
        },
        "serializer": {
            "version": serializer.SERIALIZER_VERSION,
            "path": "src/polycodegpt/tokenization/serializer.py",
            "sha256": sha256_file(PROJECT_ROOT / "src" / "polycodegpt" / "tokenization" / "serializer.py"),
        },
        "model_config": config.to_dict(),
        "splits": splits,
        "totals": {
            "tokens": total_tokens,
            "blocks": total_blocks,
            "file_size_bytes": sum(s["file_size_bytes"] for s in splits.values()),
            "records": sum(s["records"] for s in splits.values()),
            "documents": sum(s["documents"] for s in splits.values()),
            "failures": len(all_failures),
        },
        "failures": all_failures,
        "software": software_versions(),
        "run": {
            "started_at_utc": started_at,
            "finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "duration_s": round(time.perf_counter() - start, 2),
        },
    }

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=False)

    write_sources_csv(manifest, output_dir / "sources.csv")
    if all_failures:
        with open(output_dir / "failures.jsonl", "w", encoding="utf-8") as fh:
            for failure in all_failures:
                fh.write(json.dumps(failure, ensure_ascii=False) + "\n")

    return manifest


def write_sources_csv(manifest: Mapping[str, Any], path: str | Path) -> None:
    """Write per-source provenance rows (one per input file/inventory)."""
    import csv

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["split", "kind", "path", "sha256", "records", "partition",
             "split_tokens", "split_blocks", "split_dropped_trailing_tokens"]
        )
        for name, split in manifest["splits"].items():
            for source in split["sources"]:
                writer.writerow(
                    [
                        source["split"],
                        source["kind"],
                        source["path"],
                        source["sha256"],
                        source["records"],
                        source["partition"],
                        split["tokens"],
                        split["blocks"],
                        split["dropped_trailing_tokens"],
                    ]
                )


def deterministic_view(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Manifest content that must be identical for identical inputs (no timings/paths)."""
    view = json.loads(json.dumps(manifest))
    view.pop("run", None)
    view.pop("output_dir", None)
    view.pop("software", None)
    for split in view["splits"].values():
        split.pop("elapsed_s", None)
        split.pop("path", None)
        for source in split.get("sources", []):
            source.pop("path", None)
    return view
