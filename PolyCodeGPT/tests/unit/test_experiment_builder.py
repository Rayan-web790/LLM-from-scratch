"""Tests for the Phase 12 experiment dataset builder.

Everything runs on synthetic records with the real tokenizer; the corpus is never
read.  The properties checked are the ones the production build relies on:
records are never split, budgets are respected, sampling frequencies and unique
coverage are measured, and the produced artifact loads through the real dataloader.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from polycodegpt.data.experiment_builder import (  # noqa: E402
    ExperimentSpec,
    LanguageStream,
    _selection_order,
    build_experiment_dataset,
    build_language_stream,
    verify_with_dataloader,
)
from polycodegpt.data.tokenizer_io import load_tokenizer  # noqa: E402


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer(vocab_size=16000)


def write_records(path: Path, language: str, count: int) -> Path:
    with open(path, "w", encoding="utf-8") as fh:
        for index in range(count):
            record = {
                "repo": "example/repo",
                "path": f"file_{index}.py" if language == "python" else f"file_{index}.js",
                "func_name": f"func_{index}",
                "language": language,
                "original_string": f"def func_{index}(x):\n    return x + {index}\n",
                "docstring": f"Adds {index} to x." if index % 3 else "",
                "code": f"return x + {index}",
                "code_tokens": ["return", "x", "+", str(index)],
                "docstring_tokens": [],
            }
            fh.write(json.dumps(record) + "\n")
    return path


def build_stream(tmp_path: Path, name: str, language: str, count: int, tokenizer) -> LanguageStream:
    from polycodegpt.data.experiment_builder import _serialize_items
    from polycodegpt.tokenization import serializer
    from polycodegpt.data.tokenizer_io import LANG_EN_ID

    if language == "en":
        def items():
            for index in range(count):
                yield (
                    {"index": index, "record_id": f"doc_{index}.rst", "language": "en",
                     "source": "synthetic", "expected_first_id": LANG_EN_ID},
                    serializer.serialize_documentation(
                        f"Section {index}\n=========\n\nProse paragraph number {index}.\n"
                    ),
                )
    else:
        source = write_records(tmp_path / f"{name}.jsonl", language, count)
        items = lambda: _serialize_items(source)  # noqa: E731

    return build_language_stream(name, items(), tmp_path / f"_{name}.bin",
                                 tokenizer, 16000, progress_every=0)


# --------------------------------------------------------------------------- #
# selection logic
# --------------------------------------------------------------------------- #
def test_selection_never_splits_a_record_and_respects_budget():
    lengths = [10, 20, 30, 40]
    order = _selection_order(4, 65, lengths, "all_records", None)
    assert sum(lengths[i] for i in order) <= 65
    # 65 cannot be hit exactly: 10+20+30 = 60, +40 = 100 > 65
    assert sum(lengths[i] for i in order) == 60


def test_selection_cycles_when_the_source_is_smaller_than_the_budget():
    import numpy as np

    lengths = [10] * 5
    order = _selection_order(5, 100, lengths, "oversample_curated_to_budget",
                             np.random.default_rng(0))
    assert sum(lengths[i] for i in order) == 100
    assert len(order) == 10
    assert len(set(order)) == 5  # every record is used, twice


def test_subsample_mode_uses_a_subset():
    import numpy as np

    lengths = [10] * 100
    order = _selection_order(100, 200, lengths, "subsample_to_budget", np.random.default_rng(0))
    assert sum(lengths[i] for i in order) == 200
    assert len(set(order)) == len(order) == 20


# --------------------------------------------------------------------------- #
# dataset assembly
# --------------------------------------------------------------------------- #
def test_build_experiment_dataset_accounts_for_every_token(tmp_path, tokenizer):
    streams = {
        "python": build_stream(tmp_path, "python", "python", 20, tokenizer),
        "javascript_original": build_stream(tmp_path, "js_orig", "javascript", 20, tokenizer),
        "javascript_curated": build_stream(tmp_path, "js_cur", "javascript", 10, tokenizer),
        "english_docs": build_stream(tmp_path, "docs_en", "en", 3, tokenizer),
    }
    reference_docs = streams["english_docs"].tokens

    spec = ExperimentSpec(
        experiment_id="unit",
        python_tokens=streams["python"].tokens,
        javascript_tokens=streams["javascript_original"].tokens,
        javascript_source="original",
    )
    manifest = build_experiment_dataset(spec, tmp_path / "out", streams, 16000, 64, seed=7)

    totals = manifest["serialized_token_totals"]
    assert totals["total"] == manifest["train_bin"]["tokens"]
    assert totals["python"] + totals["javascript"] + totals["english_documentation"] == totals["total"]
    assert totals["english_documentation"] == reference_docs
    assert totals["python"] == streams["python"].tokens

    python = manifest["languages"]["python"]
    assert python["unique_records_used"] == 20
    assert python["available_records"] == 20
    assert python["effective_epochs"] == 1.0
    assert python["repeated_records"] == 0
    assert python["unfilled_tokens"] == 0

    # the artifact exists and is block-aligned geometry as reported
    bin_path = Path(manifest["train_bin"]["path"])
    assert bin_path.stat().st_size == totals["total"] * 2
    assert manifest["geometry"]["blocks"] >= 1


def test_oversampled_language_reports_repetition(tmp_path, tokenizer):
    streams = {
        "python": build_stream(tmp_path, "python", "python", 5, tokenizer),
        "javascript_original": build_stream(tmp_path, "js_orig", "javascript", 5, tokenizer),
        "javascript_curated": build_stream(tmp_path, "js_cur", "javascript", 3, tokenizer),
        "english_docs": build_stream(tmp_path, "docs_en", "en", 1, tokenizer),
    }
    curated = streams["javascript_curated"]
    spec = ExperimentSpec(
        experiment_id="oversampled",
        python_tokens=streams["python"].tokens,
        javascript_tokens=curated.tokens * 3,
        javascript_source="curated",
        javascript_sampling="oversample_curated_to_budget",
    )
    manifest = build_experiment_dataset(spec, tmp_path / "out", streams, 16000, 64, seed=3)
    js = manifest["languages"]["javascript"]
    assert js["unique_records_used"] == 3
    assert js["unique_coverage"] == 1.0
    assert js["effective_epochs"] > 1.0
    assert js["repeated_records"] == 3


def test_dataloader_consumes_the_materialized_artifact(tmp_path, tokenizer):
    streams = {
        "python": build_stream(tmp_path, "python", "python", 30, tokenizer),
        "javascript_original": build_stream(tmp_path, "js_orig", "javascript", 30, tokenizer),
        "javascript_curated": build_stream(tmp_path, "js_cur", "javascript", 10, tokenizer),
        "english_docs": build_stream(tmp_path, "docs_en", "en", 2, tokenizer),
    }
    spec = ExperimentSpec(
        experiment_id="loadable",
        python_tokens=streams["python"].tokens,
        javascript_tokens=streams["javascript_original"].tokens,
        javascript_source="original",
    )
    manifest = build_experiment_dataset(spec, tmp_path / "out", streams, 16000, 16, seed=5)
    report = verify_with_dataloader(
        Path(manifest["train_bin"]["path"]), context_length=16,
        micro_batch_size=2, gradient_accumulation_steps=2, vocab_size=16000,
    )
    assert report["stream_tokens"] == manifest["train_bin"]["tokens"]
    assert report["first_batch_input_shape"] == [2, 16]
    assert report["first_batch_target_shape"] == [2, 16]
    assert report["first_batch_input_tokens"] == 32
    assert report["model_input_tokens_per_update"] == 4 * 16
    assert report["total_updates_one_pass"] == report["full_updates"] + (
        1 if report["final_partial_update_sequences"] else 0
    )
    assert report["loader_batches_per_epoch"] > 0
