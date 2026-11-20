"""Lightweight unit tests for Phase 6B validation generation diagnostic."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

SCRIPT_PATH = PROJECT_ROOT / "scripts" / "evaluate_validation_generation.py"

spec = importlib.util.spec_from_file_location("evaluate_validation_generation", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

from polycodegpt.data.tokenizer_io import load_tokenizer, CODE_ID, EOS_ID


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


def test_deterministic_sample_selection(tok):
    records = [make_record(i, "python") for i in range(30)] + [make_record(i, "javascript") for i in range(30)]
    sel1, _ = mod.select_validation_samples(records, tok, seed=42, n_per_language=5)
    sel2, _ = mod.select_validation_samples(records, tok, seed=42, n_per_language=5)
    assert [mod.stable_record_id(r) for r in sel1] == [mod.stable_record_id(r) for r in sel2]
    assert len(sel1) == 10


def test_language_balancing(tok):
    records = [make_record(i, "python") for i in range(30)] + [make_record(i, "javascript") for i in range(30)]
    sel, _ = mod.select_validation_samples(records, tok, seed=42, n_per_language=10)
    langs = [r["language"] for r in sel]
    assert langs.count("python") == 10
    assert langs.count("javascript") == 10


def test_validation_only_data_access():
    src = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "valid.jsonl" in src
    assert "test.jsonl" not in src
    assert "test.bin" not in src
    assert mod.VALID_PATH.name == "valid.jsonl"


def test_validation_path_guard():
    assert mod.ensure_validation_path(mod.VALID_PATH).name == "valid.jsonl"
    with pytest.raises(ValueError):
        mod.ensure_validation_path(PROJECT_ROOT / "data" / "processed" / "polycodegpt_corpus" / "test.jsonl")


def test_serializer_prompt_construction(tok):
    rec = make_record(1, "python")
    prompt = mod.build_docstring_prompt(rec)
    assert prompt.startswith("<|lang:python|><|docstring|>")
    assert prompt.endswith("<|code|>")
    assert "Return 1." in prompt
    ids = tok.encode(prompt, add_special_tokens=False).ids
    assert ids[0] == 6
    assert CODE_ID in ids


def test_code_prefix_slicing(tok):
    rec = make_record(5, "python")
    serialized, ids = mod.tokenize_serialized(rec, tok)
    prompt_ids, ref_ids, code_ids, prefix_len = mod.slice_code_prefix(ids)
    assert len(prompt_ids) == ids.index(CODE_ID) + 1 + prefix_len
    assert ref_ids == code_ids[prefix_len:]
    assert prefix_len >= mod.MIN_PREFIX_TOKENS
    assert len(ref_ids) >= mod.MIN_CONTINUATION_TOKENS
    assert code_ids == ids[ids.index(CODE_ID) + 1:-1]
    assert prompt_ids == ids[:len(prompt_ids)]


def test_eos_excluded_from_reference(tok):
    rec = make_record(2, "javascript")
    serialized, ids = mod.tokenize_serialized(rec, tok)
    assert ids[-1] == EOS_ID
    code_tokens = mod.count_code_tokens(ids)
    prompt_ids, ref_ids, code_ids, prefix_len = mod.slice_code_prefix(ids)
    assert EOS_ID not in ref_ids
    assert EOS_ID not in code_ids
    assert code_tokens == len(code_ids)


def test_context_length_protection(tok):
    rec = make_record(0, "python")
    serialized, ids = mod.tokenize_serialized(rec, tok)
    prompt_ids, ref_ids, code_ids, prefix_len = mod.slice_code_prefix(ids)
    with pytest.raises(ValueError):
        mod.protect_context_length(prompt_ids, 1)
    mod.protect_context_length(prompt_ids, len(prompt_ids))


def test_insufficient_eligible_records(tok):
    records = [make_record(i, "python") for i in range(2)] + [make_record(i, "javascript") for i in range(2)]
    with pytest.raises(mod.InsufficientEligibleRecordsError):
        mod.select_validation_samples(records, tok, seed=42, n_per_language=5)


def test_empty_docstring_code_ineligible(tok):
    rec = make_record(0, "python")
    rec["docstring"] = "   "
    assert mod.eligibility_reason(rec, tok) == "empty docstring"
    rec2 = make_record(1, "python")
    rec2["code"] = ""
    assert mod.eligibility_reason(rec2, tok) == "empty code"


def test_count_code_tokens_matches_serialized(tok):
    rec = make_record(3, "python")
    serialized, ids = mod.tokenize_serialized(rec, tok)
    ci = ids.index(CODE_ID)
    assert mod.count_code_tokens(ids) == len(ids[ci + 1:]) - 1


def test_min_code_tokens_filter(tok):
    rec = {"record_id": "short", "language": "python", "source_dataset": "CodeSearchNet Python",
           "docstring": "x", "code": "pass"}
    assert mod.eligibility_reason(rec, tok) is not None
    assert "too few code tokens" in mod.eligibility_reason(rec, tok)


def test_args_defaults():
    args = mod.parse_args([])
    assert args.max_new_tokens == 128
    assert args.seed == 42
    assert args.n_per_language == 10
    assert args.min_code_tokens == 24
    assert args.device == "cuda"
    assert args.min_prefix_tokens == 8
    assert args.max_prefix_tokens == 64
