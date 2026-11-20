"""Lightweight unit tests for Phase 8 validation loss decomposition."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

SCRIPT_PATH = PROJECT_ROOT / "scripts" / "evaluate_validation_loss_breakdown.py"
spec = importlib.util.spec_from_file_location("evaluate_validation_loss_breakdown", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

from polycodegpt.data.tokenizer_io import load_tokenizer, CODE_ID, EOS_ID
from polycodegpt.data.stream import block_geometry


@pytest.fixture(scope="module")
def tok():
    return load_tokenizer(vocab_size=16000)


def make_record(i, lang="python"):
    if lang == "python":
        return {
            "record_id": f"py{i}",
            "language": "python",
            "source_dataset": "CodeSearchNet Python",
            "docstring": f"Return {i}.",
            "code": f"def f{i}(x):\n    if x < 0:\n        return 0\n    total = 0\n    for j in range(x):\n        total += j + {i}\n    return total\n",
        }
    return {
        "record_id": f"js{i}",
        "language": "javascript",
        "source_dataset": "CodeSearchNet JavaScript",
        "docstring": f"Return {i}.",
        "code": f"function f{i}(x) {{\n  if (x < 0) return 0;\n  let total = 0;\n  for (let j = 0; j < x; j++) total += j + {i};\n  return total;\n}}\n",
    }


def test_token_category_assignment(tok):
    rec = make_record(1, "python")
    _text, ids = mod.tokenize_serialized(rec, tok)
    cats = mod.assign_token_categories(ids)
    assert len(cats) == len(ids)
    assert cats[0] == mod.CATEGORY_STRUCTURAL_SPECIAL
    ci = ids.index(CODE_ID)
    # docstring content before code marker
    assert all(c == mod.CATEGORY_PYTHON_DOCSTRING for c in cats[2:ci])
    # code content after code marker (excluding eos)
    assert all(c == mod.CATEGORY_PYTHON_CODE for c in cats[ci + 1 : -1])
    assert cats[-1] == mod.CATEGORY_STRUCTURAL_SPECIAL
    # counts
    totals = mod.category_totals(cats)
    assert totals[mod.CATEGORY_PYTHON_DOCSTRING] == ci - 2
    assert totals[mod.CATEGORY_PYTHON_CODE] == len(ids) - ci - 2


def test_token_stream_reconstruction(tok):
    records = [make_record(i, "python") for i in range(5)] + [make_record(i, "javascript") for i in range(3)]
    ids, cats = mod.reconstruct_token_stream(records, tok)
    # manual flattened
    expected_ids = []
    for r in records:
        _t, rids = mod.tokenize_serialized(r, tok)
        expected_ids.extend(rids)
    assert ids == expected_ids
    assert len(cats) == len(ids)
    # category totals sum
    totals = mod.category_totals(cats)
    assert sum(totals.values()) == len(ids)


def test_window_geometry():
    geo = mod.window_geometry(1000, 8)
    assert geo["window_length"] == 9
    assert geo["window_stride"] == 8
    assert geo["window_overlap"] == 1
    # blocks
    expected = block_geometry(1000, 8)
    assert geo["blocks"] == expected["blocks"]
    assert geo["dropped_trailing_tokens"] == expected["dropped_trailing_tokens"]


def test_target_counting():
    # 7 blocks of context 8 => 56 targets
    assert mod.count_targets(7, 8) == 56
    assert mod.count_targets(num_blocks=7, context_length=8) == 56
    # from token count
    geo = block_geometry(60, 8)
    assert mod.count_targets(token_count=60, context_length=8) == geo["blocks"] * 8
    # target positions tile
    positions = mod.target_positions(2, 4)
    assert positions == [1, 2, 3, 4, 5, 6, 7, 8]


def test_loss_aggregation():
    losses = [1.0, 2.0, 3.0, 4.0]
    cats = [
        mod.CATEGORY_PYTHON_CODE,
        mod.CATEGORY_PYTHON_CODE,
        mod.CATEGORY_JAVASCRIPT_CODE,
        mod.CATEGORY_JAVASCRIPT_CODE,
    ]
    agg = mod.aggregate_losses(losses, cats)
    assert agg[mod.CATEGORY_PYTHON_CODE]["tokens"] == 2
    assert agg[mod.CATEGORY_PYTHON_CODE]["mean_loss"] == pytest.approx(1.5)
    assert agg["overall"]["tokens"] == 4
    assert agg["overall"]["mean_loss"] == pytest.approx(2.5)
    combined = mod.compute_combined_losses(agg)
    assert combined["combined_code_loss"] == pytest.approx(2.5)
    assert combined["combined_code_tokens"] == 4


def test_category_totals():
    cats = [mod.CATEGORY_STRUCTURAL_SPECIAL] * 3 + [mod.CATEGORY_PYTHON_DOCSTRING] * 2
    totals = mod.category_totals(cats)
    assert totals[mod.CATEGORY_STRUCTURAL_SPECIAL] == 3
    assert totals[mod.CATEGORY_PYTHON_DOCSTRING] == 2
    assert sum(totals.values()) == 5


def test_validation_only_data_access():
    src = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "valid.jsonl" in src
    assert "valid.bin" in src
    # no literal test-data references
    assert "test.jsonl" not in src
    assert "test.bin" not in src
    assert mod.VALID_PATH.name == "valid.jsonl"


def test_validation_path_guard():
    assert mod.ensure_validation_path(PROJECT_ROOT / "data" / "processed" / "polycodegpt_corpus" / "valid.jsonl").name == "valid.jsonl"
    with pytest.raises(ValueError):
        mod.ensure_validation_path(PROJECT_ROOT / "data" / "processed" / "polycodegpt_corpus" / "test.jsonl")


def test_verify_reconstruction():
    # matching
    arr = np.array([1, 2, 3], dtype=np.uint16)
    import tempfile, os
    fd, path = tempfile.mkstemp()
    os.close(fd)
    arr.astype("<u2").tofile(path)
    assert mod.verify_reconstruction([1, 2, 3], path) is True
    assert mod.verify_reconstruction([1, 2, 4], path) is False
    os.remove(path)


def test_reconstruction_mismatch_raises(tok):
    # ensure_validation_path not relevant; reconstruct with verify against mismatched bin
    recs = [make_record(0, "python")]
    ids, cats = mod.reconstruct_token_stream(recs, tok)
    import tempfile, os
    fd, path = tempfile.mkstemp()
    os.close(fd)
    np.array([0, 1, 2], dtype=np.uint16).tofile(path)
    with pytest.raises(ValueError):
        mod.reconstruct_token_stream(recs, tok, bin_path=path, verify=True)
    os.remove(path)
