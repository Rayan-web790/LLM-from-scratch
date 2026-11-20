"""Materialize Phase 12 experiment datasets from the preserved sources.

The Phase 12 experiments all continue the same checkpoint for 8,130 further
updates, but they differ in *which code records* those updates see.  A token
budget written in a YAML file is not a dataset, so this module does the real work:

1. ``build_language_streams`` tokenizes each source once (Python, original
   JavaScript, curated JavaScript, English documentation) and records the exact
   byte range of every individual record in that stream.
2. ``build_experiment_dataset`` selects record instances so that the language
   reaches its token budget, copies the selected byte ranges into one
   experiment-specific ``train.bin`` and measures the result.
3. ``verify_with_dataloader`` feeds the artifact through the project's real
   ``TokenBlockDataset``/``build_dataloader`` so the claim "the training loader can
   consume it" is a measurement, not an assumption.

Only whole records are ever written.  A language stops at the last record that
fits under its budget, so the materialized total can be a few thousand tokens
below the target; that shortfall is reported, never hidden.

Original sources are opened read-only and never rewritten.  Validation and test
splits are never touched: the experiments are train-only continuations.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

from ..model.config import ModelConfig
from ..tokenization import serializer
from .dataset import TokenBlockDataset, build_dataloader
from .prepare import _encode_batch, validate_example
from .stream import STREAM_DTYPE, TokenStreamWriter, block_geometry
from .tokenizer_io import TOKENIZER_SHA256

PROJECT_ROOT = Path(__file__).resolve().parents[3]

#: Authoritative training partition: the exact records phase 1 trained on.  The
#: per-language filtered files contain 44 extra Python records that the combined
#: corpus excludes, so drawing from them would train on records the split
#: construction deliberately left out.  Everything experiment-side therefore reads
#: this file and filters by the ``language`` field.
CORPUS_TRAIN = PROJECT_ROOT / "data" / "processed" / "polycodegpt_corpus" / "train.jsonl"
PYTHON_TRAIN = CORPUS_TRAIN
JS_ORIGINAL_TRAIN = CORPUS_TRAIN
JS_CURATED_TRAIN = PROJECT_ROOT / "data" / "curated" / "js_v1" / "train.jsonl"
DOCS_INVENTORY = PROJECT_ROOT / "data" / "manifests" / "python_docs_training_corpus_inventory.csv"

#: Declared token budgets from the four Phase 12 experiment configs.
ENGLISH_DOC_TOKENS = 2_847_020
TOTAL_TOKEN_BUDGET = 133_192_957

SOURCES = {
    "python": PYTHON_TRAIN,
    "javascript_original": JS_ORIGINAL_TRAIN,
    "javascript_curated": JS_CURATED_TRAIN,
    "english_docs": DOCS_INVENTORY,
}


# --------------------------------------------------------------------------- #
# experiment definitions
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ExperimentSpec:
    """Which records reach which token budget for one experiment branch."""

    experiment_id: str
    python_tokens: int
    javascript_tokens: int
    javascript_source: str  # "original" | "curated"
    python_sampling: str = "all_records"
    javascript_sampling: str = "all_records"
    notes: str = ""

    @property
    def code_tokens(self) -> int:
        return self.python_tokens + self.javascript_tokens

    @property
    def total_tokens(self) -> int:
        return self.code_tokens + ENGLISH_DOC_TOKENS


EXPERIMENTS: dict[str, ExperimentSpec] = {
    # control: the original training distribution, re-materialized from source.
    "control": ExperimentSpec(
        experiment_id="control",
        python_tokens=98_807_136,
        javascript_tokens=31_538_801,
        javascript_source="original",
        python_sampling="all_records",
        javascript_sampling="all_records",
        notes="original distribution; python:javascript = 3.13:1",
    ),
    # exp_a: curated JavaScript at the original JavaScript token volume.
    "exp_a": ExperimentSpec(
        experiment_id="exp_a",
        python_tokens=98_807_136,
        javascript_tokens=31_538_801,
        javascript_source="curated",
        python_sampling="all_records",
        javascript_sampling="oversample_curated_to_budget",
        notes="quality filter at constant language volume",
    ),
    # exp_b: original records resampled to a 50/50 token split.
    "exp_b": ExperimentSpec(
        experiment_id="exp_b",
        python_tokens=65_172_968,
        javascript_tokens=65_172_969,
        javascript_source="original",
        python_sampling="subsample_to_budget",
        javascript_sampling="oversample_to_budget",
        notes="language balance with original data only",
    ),
    # exp_c: curated JavaScript filling half the code budget.
    "exp_c": ExperimentSpec(
        experiment_id="exp_c",
        python_tokens=65_172_968,
        javascript_tokens=65_172_969,
        javascript_source="curated",
        python_sampling="subsample_to_budget",
        javascript_sampling="oversample_curated_to_budget",
        notes="quality filter plus language balance",
    ),
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _record_ids(code_jsonl: Path) -> Iterator[dict[str, Any]]:
    for record in serializer.iter_jsonl(code_jsonl):
        yield record


def _serialize_items(code_jsonl: Path,
                     language: str | None = None) -> Iterator[tuple[dict[str, Any], str]]:
    """Yield serialized records, optionally restricted to one language."""
    index = -1
    for record in _record_ids(code_jsonl):
        if language is not None and (record.get("language") or "").strip().lower() != language:
            continue
        index += 1
        yield (
            {
                "index": index,
                "record_id": f"{record.get('repo')}:{record.get('path')}:{record.get('func_name')}",
                "language": record.get("language"),
                "source": str(code_jsonl),
            },
            serializer.serialize_code_record(record),
        )


def _doc_items() -> Iterator[tuple[dict[str, Any], str]]:
    docs = serializer.load_doc_inventory(DOCS_INVENTORY, PROJECT_ROOT)
    if not docs:
        raise RuntimeError(f"no eligible documentation files in {DOCS_INVENTORY}")
    for index, doc in enumerate(docs):
        path = Path(doc["local_path"])
        if not path.is_file():
            raise RuntimeError(f"documentation file missing: {path}")
        actual = sha256_file(path)
        if actual != doc["sha256"]:
            raise RuntimeError(f"documentation SHA-256 mismatch for {path}: {actual}")
        raw = path.read_text(encoding="utf-8")
        yield (
            {
                "index": index,
                "record_id": doc["rel_path"],
                "language": "en",
                "source": str(DOCS_INVENTORY),
                "sha256": doc["sha256"],
            },
            serializer.serialize_documentation(raw),
        )


@dataclass
class LanguageStream:
    """One tokenized language plus the byte range of every record in it."""

    name: str
    path: Path
    tokens: int
    sha256: str
    record_offsets: list[int] = field(default_factory=list)
    record_lengths: list[int] = field(default_factory=list)
    record_ids: list[str] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)

    @property
    def records(self) -> int:
        return len(self.record_lengths)

    def token_counts(self) -> np.ndarray:
        return np.asarray(self.record_lengths, dtype=np.int64)

    def to_manifest(self) -> dict[str, Any]:
        counts = self.token_counts()
        return {
            "name": self.name,
            "path": str(self.path),
            "sha256": self.sha256,
            "tokens": self.tokens,
            "records": self.records,
            "record_tokens_min": int(counts.min()) if counts.size else 0,
            "record_tokens_max": int(counts.max()) if counts.size else 0,
            "record_tokens_mean": round(float(counts.mean()), 2) if counts.size else 0.0,
            "failures": len(self.failures),
        }


def build_language_stream(name: str, items: Iterable[tuple[dict[str, Any], str]], out_path: Path,
                          tokenizer, vocab_size: int, batch_size: int = 512,
                          progress_every: int = 25_000) -> LanguageStream:
    """Tokenize ``items`` into ``out_path``, recording each record's token range."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    offsets: list[int] = []
    lengths: list[int] = []
    ids: list[str] = []
    failures: list[dict[str, Any]] = []

    texts: list[str] = []
    metas: list[dict[str, Any]] = []
    started = time.perf_counter()
    seen = 0
    written_tokens = 0
    with TokenStreamWriter(out_path, vocab_size) as writer:
        def flush() -> None:
            nonlocal written_tokens
            if not texts:
                return
            for meta, token_ids in zip(metas, _encode_batch(tokenizer, texts)):
                where = f"{name}:{meta.get('record_id', '?')}"
                try:
                    if token_ids is None:
                        raise ValueError("tokenizer.encode failed")
                    validate_example(token_ids, vocab_size, where)
                except Exception as exc:  # pragma: no cover - source is already validated
                    failures.append({**meta, "reason": f"{type(exc).__name__}: {exc}"})
                    continue
                offsets.append(written_tokens)
                lengths.append(len(token_ids))
                ids.append(str(meta.get("record_id")))
                writer.write(token_ids)
                written_tokens += len(token_ids)
            texts.clear()
            metas.clear()

        for meta, text in items:
            texts.append(text)
            metas.append(meta)
            seen += 1
            if len(texts) >= batch_size:
                flush()
            if progress_every and seen % progress_every == 0:
                elapsed = max(time.perf_counter() - started, 1e-9)
                print(f"    {name}: {seen:,} records, {written_tokens:,} tokens, "
                      f"{seen / elapsed:,.0f} rec/s", flush=True)
        flush()
        stats = writer.close()

    return LanguageStream(
        name=name,
        path=out_path,
        tokens=stats["tokens"],
        sha256=stats["sha256"],
        record_offsets=offsets,
        record_lengths=lengths,
        record_ids=ids,
        failures=failures,
    )


def _selection_order(count: int, budget: int, lengths: Sequence[int], mode: str,
                     rng: np.random.Generator) -> list[int]:
    """Record indices to copy, in order, until ``budget`` tokens are filled.

    ``all_records`` cycles through every record once and then start-over (needed
    when the source is smaller than the budget); ``subsample_to_budget`` shuffles
    and takes a prefix; ``oversample_*`` cycles the shuffled list repeatedly.
    Only whole records are emitted: the last record that would exceed the budget
    is skipped, never truncated.
    """
    if count == 0:
        return []
    lengths_arr = np.asarray(lengths, dtype=np.int64)
    order: list[int] = []
    filled = 0
    if mode == "all_records":
        base = list(range(count))
    else:
        base = list(rng.permutation(count))
    index = 0
    while True:
        if index >= len(base):
            if mode == "subsample_to_budget":
                break
            base = list(rng.permutation(count))  # fresh order each epoch
            index = 0
        candidate = base[index]
        index += 1
        length = int(lengths_arr[candidate])
        if filled + length > budget:
            if mode == "subsample_to_budget":
                continue  # keep looking for a record that fits
            # one pass over the remaining candidates to use up the slack
            remaining = [i for i in base[index:] if filled + int(lengths_arr[i]) <= budget]
            if remaining:
                candidate = remaining[0]
                length = int(lengths_arr[candidate])
            else:
                break
        order.append(candidate)
        filled += length
        if filled >= budget:
            break
    return order


def _copy_records(stream: LanguageStream, order: Sequence[int], writer: TokenStreamWriter,
                  chunk_tokens: int = 1 << 20) -> int:
    """Append the selected record byte ranges of ``stream`` to ``writer``."""
    written = 0
    with open(stream.path, "rb") as fh:
        for record_index in order:
            offset = stream.record_offsets[record_index]
            length = stream.record_lengths[record_index]
            fh.seek(offset * STREAM_DTYPE.itemsize)
            remaining = length
            while remaining > 0:
                take = min(remaining, chunk_tokens)
                data = np.fromfile(fh, dtype=STREAM_DTYPE, count=take)
                if data.size != take:
                    raise RuntimeError(f"short read from {stream.path} at record {record_index}")
                writer.write(data)
                written += int(data.size)
                remaining -= take
    return written


# --------------------------------------------------------------------------- #
# dataset construction
# --------------------------------------------------------------------------- #
def build_experiment_dataset(
    spec: ExperimentSpec,
    output_dir: Path,
    streams: dict[str, LanguageStream],
    vocab_size: int,
    context_length: int,
    seed: int = 42,
) -> dict[str, Any]:
    """Write ``train.bin`` for ``spec`` and return its measured manifest."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "train.bin"
    rng = np.random.default_rng(seed)

    python_stream = streams["python"]
    js_stream = streams["javascript_curated" if spec.javascript_source == "curated" else "javascript_original"]
    docs_stream = streams["english_docs"]

    py_order = _selection_order(python_stream.records, spec.python_tokens,
                               python_stream.record_lengths, spec.python_sampling, rng)
    js_order = _selection_order(js_stream.records, spec.javascript_tokens,
                               js_stream.record_lengths, spec.javascript_sampling, rng)
    doc_order = list(range(docs_stream.records))

    with TokenStreamWriter(out_path, vocab_size) as writer:
        python_tokens = _copy_records(python_stream, py_order, writer)
        javascript_tokens = _copy_records(js_stream, js_order, writer)
        english_tokens = _copy_records(docs_stream, doc_order, writer)
        stats = writer.close()
    if python_tokens + javascript_tokens + english_tokens != stats["tokens"]:
        raise RuntimeError(
            f"language totals {python_tokens + javascript_tokens + english_tokens:,} do not match "
            f"the written stream {stats['tokens']:,}"
        )

    geometry = block_geometry(stats["tokens"], context_length)

    def language_report(name: str, stream: LanguageStream, order: Sequence[int],
                        target: int, sampling: str) -> dict[str, Any]:
        lengths = np.asarray(stream.record_lengths, dtype=np.int64)
        selected = lengths[np.asarray(order, dtype=np.int64)] if order else np.zeros(0, dtype=np.int64)
        unique = len(set(order))
        counts = np.bincount(np.asarray(order, dtype=np.int64), minlength=stream.records) if order else np.zeros(0)
        repeated = int((counts > 1).sum()) if counts.size else 0
        return {
            "source": str(stream.path),
            "source_sha256": stream.sha256,
            "sampling": sampling,
            "budget_target_tokens": int(target),
            "materialized_tokens": int(selected.sum()),
            "unfilled_tokens": int(target - selected.sum()),
            "record_instances": len(order),
            "unique_records_used": unique,
            "available_records": stream.records,
            "unique_coverage": round(unique / stream.records, 6) if stream.records else 0.0,
            "repeated_records": repeated,
            "max_repeats_of_one_record": int(counts.max()) if counts.size else 0,
            "effective_epochs": round(len(order) / stream.records, 4) if stream.records else 0.0,
            "record_tokens_min": int(selected.min()) if selected.size else 0,
            "record_tokens_max": int(selected.max()) if selected.size else 0,
        }

    manifest: dict[str, Any] = {
        "experiment_id": spec.experiment_id,
        "notes": spec.notes,
        "seed": seed,
        "train_bin": {
            "path": str(out_path),
            "sha256": stats["sha256"],
            "file_size_bytes": stats["file_size_bytes"],
            "tokens": stats["tokens"],
        },
        "serialized_token_totals": {
            "python": python_tokens,
            "javascript": javascript_tokens,
            "english_documentation": english_tokens,
            "total": stats["tokens"],
        },
        "declared_budget": {
            "python": spec.python_tokens,
            "javascript": spec.javascript_tokens,
            "english_documentation": ENGLISH_DOC_TOKENS,
            "total": spec.total_tokens,
        },
        "languages": {
            "python": language_report("python", python_stream, py_order, spec.python_tokens,
                                      spec.python_sampling),
            "javascript": language_report("javascript", js_stream, js_order,
                                         spec.javascript_tokens, spec.javascript_sampling),
            "english_documentation": language_report("english", docs_stream, doc_order,
                                                     ENGLISH_DOC_TOKENS, "all_records"),
        },
        "geometry": geometry,
        "context_length": context_length,
        "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def verify_with_dataloader(bin_path: Path, context_length: int, micro_batch_size: int,
                           gradient_accumulation_steps: int, vocab_size: int,
                           seed: int = 42) -> dict[str, Any]:
    """Load the materialized artifact with the real training dataloader."""
    dataset = TokenBlockDataset(bin_path, context_length=context_length, vocab_size=vocab_size)
    loader = build_dataloader(dataset, batch_size=micro_batch_size, shuffle=True, seed=seed,
                              pin_memory=False)
    input_ids, targets = next(iter(loader))
    sequences_per_update = micro_batch_size * gradient_accumulation_steps
    full_updates = dataset.num_blocks // sequences_per_update
    remainder = dataset.num_blocks % sequences_per_update
    return {
        "stream_tokens": dataset.token_count,
        "blocks": dataset.num_blocks,
        "dropped_trailing_tokens": dataset.dropped_trailing_tokens,
        "window_length": dataset.window_length,
        "first_batch_input_shape": list(input_ids.shape),
        "first_batch_target_shape": list(targets.shape),
        "first_batch_input_tokens": int(input_ids.numel()),
        "first_batch_target_tokens": int(targets.numel()),
        "model_input_tokens_per_update": sequences_per_update * context_length,
        "model_target_tokens_per_update": sequences_per_update * context_length,
        "sequences_per_update": sequences_per_update,
        "full_updates": full_updates,
        "final_partial_update_sequences": remainder,
        "total_updates_one_pass": full_updates + (1 if remainder else 0),
        "loader_batches_per_epoch": len(loader),
    }


def stream_cache(work_dir: Path) -> dict[str, LanguageStream]:
    """Reload per-language indexes written by a previous build."""
    out: dict[str, LanguageStream] = {}
    for name in ("python", "javascript_original", "javascript_curated", "english_docs"):
        index = work_dir / f"_{name}.index.json"
        if not index.is_file():
            continue
        data = json.loads(index.read_text(encoding="utf-8"))
        out[name] = LanguageStream(
            name=data["name"],
            path=Path(data["path"]),
            tokens=data["tokens"],
            sha256=data["sha256"],
            record_offsets=data["record_offsets"],
            record_lengths=data["record_lengths"],
            record_ids=data["record_ids"],
            failures=data.get("failures", []),
        )
    return out


def save_stream_index(stream: LanguageStream, work_dir: Path) -> Path:
    path = work_dir / f"_{stream.name}.index.json"
    path.write_text(
        json.dumps(
            {
                "name": stream.name,
                "path": str(stream.path),
                "tokens": stream.tokens,
                "sha256": stream.sha256,
                "record_offsets": stream.record_offsets,
                "record_lengths": stream.record_lengths,
                "record_ids": stream.record_ids,
                "failures": stream.failures,
            }
        ),
        encoding="utf-8",
    )
    return path


def model_settings(config: ModelConfig | None = None) -> tuple[int, int]:
    config = config or ModelConfig.from_yaml()
    return config.vocab_size, config.context_length


TOKENIZER_EXPECTED_SHA = TOKENIZER_SHA256
