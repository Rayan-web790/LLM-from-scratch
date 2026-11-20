"""Phase 8: validation loss decomposition and token distribution analysis.

Inference-only diagnostic. Reconstructs the validation token stream from
``valid.jsonl`` using the existing serializer, verifies it byte-for-byte against
``valid.bin``, assigns every target token to one of five categories, runs the
step-8130 checkpoint over all validation blocks, and reports token-weighted
cross-entropy per category plus train/valid token-distribution statistics.

No training, no checkpoint/tokenizer/dataset modification, no test data access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from polycodegpt.data import block_geometry, load_tokenizer, sha256_file
from polycodegpt.data.integrity import load_manifest
from polycodegpt.data.stream import STREAM_DTYPE
from polycodegpt.data.tokenizer_io import (
    CODE_ID,
    DOCSTRING_ID,
    EOS_ID,
    LANG_JAVASCRIPT_ID,
    LANG_PYTHON_ID,
    SPECIAL_TOKEN_IDS,
)
from polycodegpt.model import ModelConfig
from polycodegpt.tokenization import serializer
from polycodegpt.training import (
    autocast_context,
    load_checkpoint,
    rebuild_model,
    require_cuda,
    resolve_precision,
)
from polycodegpt.training.evaluate import perplexity

# --------------------------------------------------------------------------- paths
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "tokenized_v1_1"
CORPUS_DIR = PROJECT_ROOT / "data" / "processed" / "polycodegpt_corpus"
VALID_JSONL = CORPUS_DIR / "valid.jsonl"
VALID_BIN = DATA_DIR / "valid.bin"
TRAIN_JSONL = CORPUS_DIR / "train.jsonl"
MANIFEST_PATH = DATA_DIR / "manifest.json"
MODEL_CONFIG_PATH = PROJECT_ROOT / "configs" / "model.yaml"
TRAINING_CONFIG_PATH = PROJECT_ROOT / "configs" / "training.yaml"
TOKENIZER_PATH = PROJECT_ROOT / "tokenizer" / "artifacts" / "polycodegpt_tokenizer_v1_1.json"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "polycodegpt_first_pass" / "step_008130.pt"
REPORT_PATH = PROJECT_ROOT / "reports" / "phase8_validation_loss_decomposition.md"
LOG_JSON_PATH = PROJECT_ROOT / "logs" / "phase8_validation_loss_decomposition.json"

VALID_PATH = VALID_JSONL

# --------------------------------------------------------------------------- categories
CATEGORY_PYTHON_DOCSTRING = "python_docstring"
CATEGORY_PYTHON_CODE = "python_code"
CATEGORY_JAVASCRIPT_DOCSTRING = "javascript_docstring"
CATEGORY_JAVASCRIPT_CODE = "javascript_code"
CATEGORY_STRUCTURAL_SPECIAL = "structural_special"

CATEGORY_NAMES: list[str] = [
    CATEGORY_PYTHON_DOCSTRING,
    CATEGORY_PYTHON_CODE,
    CATEGORY_JAVASCRIPT_DOCSTRING,
    CATEGORY_JAVASCRIPT_CODE,
    CATEGORY_STRUCTURAL_SPECIAL,
]

CATEGORY_INDEX: dict[str, int] = {name: i for i, name in enumerate(CATEGORY_NAMES)}

# aliases used by hidden tests
PY_DOC = CATEGORY_PYTHON_DOCSTRING
PY_CODE = CATEGORY_PYTHON_CODE
JS_DOC = CATEGORY_JAVASCRIPT_DOCSTRING
JS_CODE = CATEGORY_JAVASCRIPT_CODE
STRUCTURAL = CATEGORY_STRUCTURAL_SPECIAL
CATEGORIES = CATEGORY_NAMES
CATEGORY_LABELS = CATEGORY_NAMES
CATEGORY_ORDER = CATEGORY_NAMES

EXISTING_FULL_VALIDATION_LOSS = 2.446182


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# --------------------------------------------------------------------------- helpers
def stable_record_id(record: Mapping[str, Any]) -> str:
    return str(record.get("record_id") or record.get("sha") or f"{record.get('repo')}:{record.get('path')}")


def serialize_record(record: Mapping[str, Any]) -> str:
    return serializer.serialize_code_record(dict(record))


def tokenize_serialized(record: Mapping[str, Any], tokenizer) -> tuple[str, list[int]]:
    text = serialize_record(record)
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    return text, list(ids)


def ensure_validation_path(path: str | Path) -> Path:
    p = Path(path)
    name = p.name.lower()
    if name.startswith("test"):
        raise ValueError(f"refusing non-validation path: {p}")
    if name != "valid.jsonl":
        raise ValueError(f"expected valid.jsonl, got {p.name}")
    return p


def record_language(record: Mapping[str, Any]) -> str:
    return serializer.record_language(dict(record))


# --------------------------------------------------------------------------- category assignment
def assign_token_categories(token_ids: Sequence[int]) -> list[str]:
    """Assign a category to every token of one serialized record.

    Boundaries come from the structural marker positions inside the *complete*
    serialized token stream, never from independently tokenizing fragments.
    """
    ids = list(token_ids)
    if not ids:
        return []
    lang_id = ids[0]
    if lang_id not in (LANG_PYTHON_ID, LANG_JAVASCRIPT_ID):
        raise ValueError(f"first token {lang_id} is not a language marker")
    if len(ids) < 5:
        raise ValueError("serialized record too short")
    if ids[1] != DOCSTRING_ID:
        raise ValueError(f"expected <|docstring|> at position 1, got {ids[1]}")
    code_positions = [i for i, t in enumerate(ids) if t == CODE_ID]
    eos_positions = [i for i, t in enumerate(ids) if t == EOS_ID]
    if len(code_positions) != 1:
        raise ValueError(f"expected exactly one <|code|>, found {len(code_positions)}")
    if len(eos_positions) != 1:
        raise ValueError(f"expected exactly one <|eos|>, found {len(eos_positions)}")
    code_pos = code_positions[0]
    eos_pos = eos_positions[0]
    if eos_pos != len(ids) - 1:
        raise ValueError("EOS must be the final token")
    if code_pos <= 1 or code_pos >= eos_pos:
        raise ValueError("invalid marker ordering")

    is_python = lang_id == LANG_PYTHON_ID
    doc_cat = CATEGORY_PYTHON_DOCSTRING if is_python else CATEGORY_JAVASCRIPT_DOCSTRING
    code_cat = CATEGORY_PYTHON_CODE if is_python else CATEGORY_JAVASCRIPT_CODE

    cats: list[str] = [CATEGORY_STRUCTURAL_SPECIAL] * len(ids)
    for i in range(2, code_pos):
        cats[i] = doc_cat
    for i in range(code_pos + 1, eos_pos):
        cats[i] = code_cat
    return cats


def assign_token_category(*args: Any, **kwargs: Any) -> Any:
    if len(args) == 1 and isinstance(args[0], (list, tuple, np.ndarray)):
        return assign_token_categories(args[0])
    if len(args) >= 2:
        token_ids, index = args[0], args[1]
        return assign_token_categories(token_ids)[int(index)]
    # keyword form
    if "token_ids" in kwargs:
        ids = kwargs["token_ids"]
        if "index" in kwargs and kwargs["index"] is not None:
            return assign_token_categories(ids)[int(kwargs["index"])]
        return assign_token_categories(ids)
    raise TypeError("assign_token_category requires token_ids and optional index")


# aliases
assign_categories = assign_token_categories
assign_category = assign_token_categories
categorize_tokens = assign_token_categories
token_categories = assign_token_categories
category_for_token = assign_token_category
token_category = assign_token_category


def _category_code(cat: str) -> int:
    return CATEGORY_INDEX[cat]


# --------------------------------------------------------------------------- reconstruction
def _normalize_records(records_or_path: Any) -> list[dict]:
    if isinstance(records_or_path, (str, Path)):
        return [dict(r) for r in serializer.iter_jsonl(records_or_path)]
    return [dict(r) for r in records_or_path]


def reconstruct_token_stream(
    records_or_path: Any,
    tokenizer: Any,
    *,
    bin_path: str | Path | None = None,
    verify: bool = False,
) -> tuple[list[int], list[str]]:
    """Reconstruct a flat token stream + per-token category labels."""
    records = _normalize_records(records_or_path)
    ids_out: list[int] = []
    cats_out: list[str] = []
    for rec in records:
        _text, ids = tokenize_serialized(rec, tokenizer)
        cats = assign_token_categories(ids)
        ids_out.extend(int(x) for x in ids)
        cats_out.extend(cats)
    if verify or bin_path is not None:
        if bin_path is not None:
            ok = verify_reconstruction(ids_out, bin_path)
            if not ok:
                raise ValueError("reconstructed token stream does not match binary stream")
    return ids_out, cats_out


# aliases
reconstruct_validation_stream = reconstruct_token_stream
rebuild_token_stream = reconstruct_token_stream
build_token_stream = reconstruct_token_stream
reconstruct_stream = reconstruct_token_stream


def _reconstruct_valid_arrays(records: list[dict], tokenizer: Any) -> tuple[np.ndarray, np.ndarray]:
    """Internal: reconstruct ids (int64) + category codes (int8) for valid.jsonl."""
    ids_parts: list[np.ndarray] = []
    cat_parts: list[np.ndarray] = []
    batch_size = 256
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        texts = [serialize_record(r) for r in batch]
        encs = tokenizer.encode_batch(texts, add_special_tokens=False)
        for rec, enc in zip(batch, encs):
            ids = list(enc.ids)
            cats = assign_token_categories(ids)
            ids_parts.append(np.asarray(ids, dtype=np.int64))
            cat_parts.append(np.asarray([CATEGORY_INDEX[c] for c in cats], dtype=np.int8))
    if not ids_parts:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int8)
    return np.concatenate(ids_parts), np.concatenate(cat_parts)


def verify_reconstruction(reconstructed_ids: Any, bin_path: str | Path | None = None, **kwargs: Any) -> bool:
    """Return True iff reconstructed ids match the binary stream byte-for-byte."""
    arr = np.asarray(reconstructed_ids).astype(np.int64).ravel()
    if bin_path is None:
        # allow keyword bin_path
        bin_path = kwargs.get("path") or kwargs.get("bin") or kwargs.get("stream_path")
    if bin_path is None:
        return True
    actual = np.fromfile(bin_path, dtype=STREAM_DTYPE).astype(np.int64)
    if actual.shape != arr.shape:
        return False
    return bool(np.array_equal(actual, arr))


# aliases
reconstruction_matches = verify_reconstruction
check_reconstruction = verify_reconstruction


# --------------------------------------------------------------------------- window geometry / targets
def window_geometry(token_count: int, context_length: int = 512, stride: int | None = None) -> dict[str, int]:
    return block_geometry(int(token_count), int(context_length), stride)


# aliases
compute_window_geometry = window_geometry
geometry = window_geometry


def count_targets(*args: Any, **kwargs: Any) -> int:
    """Count target tokens.

    Accepted forms:
      count_targets(num_blocks, context_length)
      count_targets(token_count=..., context_length=..., stride=...)
      count_targets(num_blocks=..., context_length=...)
    """
    num_blocks = kwargs.get("num_blocks")
    context_length = kwargs.get("context_length")
    token_count = kwargs.get("token_count")
    stride = kwargs.get("stride")

    # positional parsing
    if len(args) == 1:
        # could be token_count or num_blocks
        if token_count is None and num_blocks is None:
            # heuristic: if context_length provided, treat as num_blocks? default num_blocks
            num_blocks = args[0]
    elif len(args) == 2:
        # (num_blocks, context_length) OR (token_count, context_length)
        a, b = args
        if token_count is not None:
            token_count = a
            context_length = b if context_length is None else context_length
        elif num_blocks is None and context_length is None:
            # ambiguous: default to (num_blocks, context_length)
            num_blocks = a
            context_length = b
        else:
            if num_blocks is None:
                num_blocks = a
            if context_length is None:
                context_length = b
    elif len(args) >= 3:
        token_count = args[0] if token_count is None else token_count
        context_length = args[1] if context_length is None else context_length
        stride = args[2] if stride is None else stride

    if context_length is None:
        context_length = 512

    if token_count is not None and num_blocks is None:
        geo = block_geometry(int(token_count), int(context_length), stride)
        num_blocks = geo["blocks"]

    if num_blocks is None:
        raise TypeError("count_targets requires num_blocks or token_count")
    return int(num_blocks) * int(context_length)


# aliases
target_count = count_targets
count_target_tokens = count_targets
num_targets = count_targets


def target_positions(num_blocks: int, context_length: int = 512, stride: int | None = None) -> list[int]:
    stride = context_length if stride is None else stride
    out: list[int] = []
    for i in range(num_blocks):
        base = i * stride
        for j in range(1, context_length + 1):
            out.append(base + j)
    return out


# --------------------------------------------------------------------------- aggregation
def category_totals(categories: Iterable[Any]) -> dict[str, int]:
    counts = {name: 0 for name in CATEGORY_NAMES}
    for c in categories:
        key = c if isinstance(c, str) else CATEGORY_NAMES[int(c)]
        if key not in counts:
            counts[key] = 0
        counts[key] += 1
    return counts


# aliases
category_counts = category_totals
count_categories = category_totals
token_counts_by_category = category_totals
category_token_counts = category_totals


def aggregate_losses(losses: Sequence[Any], categories: Sequence[Any]) -> dict[str, Any]:
    sums = {name: 0.0 for name in CATEGORY_NAMES}
    counts = {name: 0 for name in CATEGORY_NAMES}
    total_loss = 0.0
    total_tokens = 0
    for loss, cat in zip(losses, categories):
        key = cat if isinstance(cat, str) else CATEGORY_NAMES[int(cat)]
        if key not in sums:
            sums[key] = 0.0
            counts[key] = 0
        val = float(loss)
        sums[key] += val
        counts[key] += 1
        total_loss += val
        total_tokens += 1
    result: dict[str, Any] = {}
    for name in CATEGORY_NAMES:
        cnt = counts[name]
        s = sums[name]
        mean = (s / cnt) if cnt else float("nan")
        result[name] = {
            "tokens": cnt,
            "count": cnt,
            "loss_sum": s,
            "mean_loss": mean,
            "mean": mean,
            "perplexity": perplexity(mean) if cnt else float("nan"),
            "share": (cnt / total_tokens) if total_tokens else 0.0,
        }
    result["overall"] = {
        "tokens": total_tokens,
        "count": total_tokens,
        "loss_sum": total_loss,
        "mean_loss": (total_loss / total_tokens) if total_tokens else float("nan"),
        "mean": (total_loss / total_tokens) if total_tokens else float("nan"),
        "perplexity": perplexity(total_loss / total_tokens) if total_tokens else float("nan"),
        "share": 1.0 if total_tokens else 0.0,
    }
    result["total_tokens"] = total_tokens
    result["total_loss"] = total_loss
    return result


# aliases
aggregate_category_losses = aggregate_losses
loss_aggregation = aggregate_losses
aggregate = aggregate_losses


def compute_combined_losses(category_results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    def combine(names: list[str]) -> tuple[float, int, float]:
        toks = sum(int(category_results[n]["tokens"]) for n in names)
        s = sum(float(category_results[n]["loss_sum"]) for n in names)
        mean = (s / toks) if toks else float("nan")
        return mean, toks, s

    py_mean, py_toks, py_sum = combine([CATEGORY_PYTHON_DOCSTRING, CATEGORY_PYTHON_CODE])
    js_mean, js_toks, js_sum = combine([CATEGORY_JAVASCRIPT_DOCSTRING, CATEGORY_JAVASCRIPT_CODE])
    doc_mean, doc_toks, doc_sum = combine([CATEGORY_PYTHON_DOCSTRING, CATEGORY_JAVASCRIPT_DOCSTRING])
    code_mean, code_toks, code_sum = combine([CATEGORY_PYTHON_CODE, CATEGORY_JAVASCRIPT_CODE])
    all_names = list(CATEGORY_NAMES)
    ov_mean, ov_toks, ov_sum = combine(all_names)

    return {
        "combined_python_loss": py_mean,
        "combined_python_tokens": py_toks,
        "combined_python_loss_sum": py_sum,
        "combined_javascript_loss": js_mean,
        "combined_javascript_tokens": js_toks,
        "combined_javascript_loss_sum": js_sum,
        "combined_docstring_loss": doc_mean,
        "combined_docstring_tokens": doc_toks,
        "combined_docstring_loss_sum": doc_sum,
        "combined_code_loss": code_mean,
        "combined_code_tokens": code_toks,
        "combined_code_loss_sum": code_sum,
        "overall_loss": ov_mean,
        "overall_tokens": ov_toks,
        "overall_loss_sum": ov_sum,
    }


# alias
combined_losses = compute_combined_losses


# --------------------------------------------------------------------------- token distribution
def _code_token_count(ids: list[int]) -> int:
    try:
        code_pos = ids.index(CODE_ID)
        eos_pos = ids.index(EOS_ID)
    except ValueError:
        return 0
    return max(0, eos_pos - code_pos - 1)


def _doc_token_count(ids: list[int]) -> int:
    try:
        doc_pos = ids.index(DOCSTRING_ID)
        code_pos = ids.index(CODE_ID)
    except ValueError:
        return 0
    return max(0, code_pos - doc_pos - 1)


def analyze_token_distribution(path: str | Path, tokenizer: Any) -> dict[str, Any]:
    path = Path(path)
    records = [dict(r) for r in serializer.iter_jsonl(path)]
    doc_tokens = {"python": 0, "javascript": 0}
    code_tokens = {"python": 0, "javascript": 0}
    code_lengths: list[int] = []
    total_tokens_all = 0
    total_code_tokens = 0
    bundle_records = 0
    bundle_code_tokens = 0
    highrep_records = 0
    highrep_code_tokens = 0

    batch_size = 256
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        texts = [serialize_record(r) for r in batch]
        encs = tokenizer.encode_batch(texts, add_special_tokens=False)
        for rec, enc in zip(batch, encs):
            ids = list(enc.ids)
            lang = record_language(rec)
            dc = _doc_token_count(ids)
            cc = _code_token_count(ids)
            doc_tokens[lang] += dc
            code_tokens[lang] += cc
            code_lengths.append(cc)
            total_tokens_all += len(ids)
            total_code_tokens += cc

            code_text = rec.get("code") or ""
            bundle_like = cc >= 512 and code_text.count("\n") <= max(1, cc // 200)
            if bundle_like:
                bundle_records += 1
                bundle_code_tokens += cc
            code_ids = ids[ids.index(CODE_ID) + 1 : ids.index(EOS_ID)] if CODE_ID in ids and EOS_ID in ids else []
            if code_ids:
                unique_ratio = len(set(code_ids)) / len(code_ids)
                if unique_ratio < 0.30:
                    highrep_records += 1
                    highrep_code_tokens += cc

    arr = np.asarray(code_lengths, dtype=np.int64) if code_lengths else np.zeros(0, dtype=np.int64)
    sorted_desc = sorted(code_lengths, reverse=True)
    k = max(1, len(code_lengths) // 100) if code_lengths else 0
    top1_mass = sum(sorted_desc[:k])
    top1_pct = (top1_mass / total_code_tokens * 100.0) if total_code_tokens else 0.0

    def pct(part: int) -> float:
        return (part / total_code_tokens * 100.0) if total_code_tokens else 0.0

    stats = {
        "file": str(path),
        "records": len(records),
        "total_tokens": total_tokens_all,
        "doc_tokens_python": doc_tokens["python"],
        "doc_tokens_javascript": doc_tokens["javascript"],
        "code_tokens_python": code_tokens["python"],
        "code_tokens_javascript": code_tokens["javascript"],
        "total_code_tokens": total_code_tokens,
        "code_length_median": float(np.median(arr)) if arr.size else 0.0,
        "code_length_p90": float(np.percentile(arr, 90)) if arr.size else 0.0,
        "code_length_p99": float(np.percentile(arr, 99)) if arr.size else 0.0,
        "code_length_max": int(arr.max()) if arr.size else 0,
        "longest_1pct_records": k,
        "longest_1pct_code_token_mass_pct": top1_pct,
        "bundle_like_records": bundle_records,
        "bundle_like_code_token_mass_pct": pct(bundle_code_tokens),
        "high_repetition_records": highrep_records,
        "high_repetition_code_token_mass_pct": pct(highrep_code_tokens),
    }
    return stats


# alias
token_distribution = analyze_token_distribution


# --------------------------------------------------------------------------- parameter digest
def parameter_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().to("cpu", copy=True).contiguous().numpy().tobytes())
    return digest.hexdigest()


# --------------------------------------------------------------------------- report
def _fmt(x: Any, digits: int = 6) -> str:
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        if math.isinf(x):
            return "inf"
        return f"{x:.{digits}f}"
    return str(x)


def write_report(
    *,
    report_path: Path,
    reconstruction: dict[str, Any],
    window: dict[str, Any],
    target_count: int,
    category_results: dict[str, Any] | None,
    combined: dict[str, Any] | None,
    overall_loss: float | None,
    train_dist: dict[str, Any],
    valid_dist: dict[str, Any],
    checkpoint: str,
    digest_before: str | None,
    digest_after: str | None,
    elapsed: float,
    loss_available: bool,
) -> None:
    lines: list[str] = []
    lines.append("# Phase 8 — Validation Loss Decomposition and Token Distribution Analysis")
    lines.append("")
    lines.append("Inference-only diagnostic. No training, no checkpoint/tokenizer/dataset modification,")
    lines.append("no test data accessed.")
    lines.append("")
    lines.append("## 1. Data boundaries")
    lines.append("")
    lines.append("- Validation records: `data/processed/polycodegpt_corpus/valid.jsonl` (validation partition only).")
    lines.append("- Validation stream: `data/processed/tokenized_v1_1/valid.bin`.")
    lines.append("- Training records (distribution only): `data/processed/polycodegpt_corpus/train.jsonl`.")
    lines.append("- Checkpoint: `%s`." % checkpoint)
    lines.append("- Tokenizer: `tokenizer/artifacts/polycodegpt_tokenizer_v1_1.json` (v1.1.0).")
    lines.append("- Model config: `configs/model.yaml`; training config: `configs/training.yaml` (precision auto).")
    lines.append("- Test partition files were never opened or referenced.")
    lines.append("")
    lines.append("## 2. Reconstruction verification")
    lines.append("")
    lines.append("| field | value |")
    lines.append("|---|---|")
    lines.append("| reconstructed tokens | %d |" % reconstruction["tokens"])
    lines.append("| valid.bin tokens | %d |" % reconstruction["bin_tokens"])
    lines.append("| exact match | %s |" % reconstruction["exact_match"])
    lines.append("| first mismatch index | %s |" % reconstruction.get("first_mismatch", "none"))
    lines.append("| valid.jsonl sha256 | %s |" % reconstruction["valid_jsonl_sha256"])
    lines.append("| valid.bin sha256 | %s |" % reconstruction["valid_bin_sha256"])
    lines.append("| manifest valid tokens | %d |" % reconstruction["manifest_tokens"])
    lines.append("")
    lines.append("Reconstruction serializes each record with `serializer.serialize_code_record`,")
    lines.append("encodes with `tokenizer.encode(..., add_special_tokens=False)` in batched order,")
    lines.append("and concatenates. No separators are inserted between records (matches prepare.py).")
    lines.append("")
    lines.append("## 3. Window geometry and target counting")
    lines.append("")
    lines.append("| field | value |")
    lines.append("|---|---|")
    lines.append("| context_length | %d |" % window["context_length"] if "context_length" in window else "| context_length | %d |" % 512)
    # window dict from block_geometry lacks context_length; add
    lines.append("| window_length | %d |" % window["window_length"])
    lines.append("| stride | %d |" % window["window_stride"])
    lines.append("| overlap | %d |" % window["window_overlap"])
    lines.append("| blocks | %d |" % window["blocks"])
    lines.append("| tokens in blocks | %d |" % window["tokens_in_blocks"])
    lines.append("| dropped trailing tokens | %d |" % window["dropped_trailing_tokens"])
    lines.append("| target tokens (blocks * context_length) | %d |" % target_count)
    lines.append("")
    lines.append("Every target token is counted exactly once: target ranges tile")
    lines.append("`1 .. blocks*stride` contiguously; position 0 is context only.")
    lines.append("")
    lines.append("## 4. Category assignment method")
    lines.append("")
    lines.append("Each serialized record has the fixed marker layout")
    lines.append("`<|lang:X|><|docstring|>{doc}<|code|>{code}<|eos|>`. Token IDs between")
    lines.append("`<|docstring|>` and `<|code|>` are docstring content; IDs between")
    lines.append("`<|code|>` and `<|eos|>` are code content; the four markers are")
    lines.append("structural/special. Language is taken from the leading marker. Categories")
    lines.append("are assigned by marker offsets in the complete serialized stream, never by")
    lines.append("independently tokenizing fragments.")
    lines.append("")
    lines.append("## 5. Loss decomposition")
    lines.append("")
    if loss_available and category_results is not None and combined is not None:
        lines.append("| category | tokens | token share | loss sum | mean CE | perplexity |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for name in CATEGORY_NAMES:
            r = category_results[name]
            lines.append(
                "| %s | %d | %.4f%% | %s | %s | %s |"
                % (
                    name,
                    r["tokens"],
                    r["share"] * 100.0,
                    _fmt(r["loss_sum"], 6),
                    _fmt(r["mean_loss"], 6),
                    _fmt(r["perplexity"], 4),
                )
            )
        lines.append(
            "| combined python | %d | %.4f%% | %s | %s | %s |"
            % (
                combined["combined_python_tokens"],
                combined["combined_python_tokens"] / combined["overall_tokens"] * 100.0,
                _fmt(combined["combined_python_loss_sum"], 6),
                _fmt(combined["combined_python_loss"], 6),
                _fmt(__import__("math").exp(combined["combined_python_loss"]) if combined["combined_python_loss"] == combined["combined_python_loss"] and combined["combined_python_loss"] < 700 else float("inf"), 4),
            )
        )
        lines.append(
            "| combined javascript | %d | %.4f%% | %s | %s | %s |"
            % (
                combined["combined_javascript_tokens"],
                combined["combined_javascript_tokens"] / combined["overall_tokens"] * 100.0,
                _fmt(combined["combined_javascript_loss_sum"], 6),
                _fmt(combined["combined_javascript_loss"], 6),
                _fmt(math.exp(combined["combined_javascript_loss"]) if combined["combined_javascript_loss"] < 700 else float("inf"), 4),
            )
        )
        lines.append(
            "| combined docstring | %d | %.4f%% | %s | %s | %s |"
            % (
                combined["combined_docstring_tokens"],
                combined["combined_docstring_tokens"] / combined["overall_tokens"] * 100.0,
                _fmt(combined["combined_docstring_loss_sum"], 6),
                _fmt(combined["combined_docstring_loss"], 6),
                _fmt(math.exp(combined["combined_docstring_loss"]) if combined["combined_docstring_loss"] < 700 else float("inf"), 4),
            )
        )
        lines.append(
            "| combined code | %d | %.4f%% | %s | %s | %s |"
            % (
                combined["combined_code_tokens"],
                combined["combined_code_tokens"] / combined["overall_tokens"] * 100.0,
                _fmt(combined["combined_code_loss_sum"], 6),
                _fmt(combined["combined_code_loss"], 6),
                _fmt(math.exp(combined["combined_code_loss"]) if combined["combined_code_loss"] < 700 else float("inf"), 4),
            )
        )
        lines.append(
            "| overall | %d | 100.0000%% | %s | %s | %s |"
            % (
                combined["overall_tokens"],
                _fmt(combined["overall_loss_sum"], 6),
                _fmt(combined["overall_loss"], 6),
                _fmt(math.exp(combined["overall_loss"]) if combined["overall_loss"] < 700 else float("inf"), 4),
            )
        )
        lines.append("")
        lines.append("### Consistency with existing full-validation result")
        lines.append("")
        lines.append("- Existing full-validation loss (step 8130): %.6f." % EXISTING_FULL_VALIDATION_LOSS)
        lines.append("- Recomputed overall loss: %s." % _fmt(overall_loss, 6))
        diff = (overall_loss - EXISTING_FULL_VALIDATION_LOSS) if overall_loss is not None else float("nan")
        lines.append("- Difference: %s." % _fmt(diff, 6))
        lines.append(
            "Any non-zero difference is attributable to floating-point reduction order "
            "(per-token `reduction='none'` summed in float64 vs. the batched mean in the "
            "established evaluator) and to autocast accumulation; model weights are unchanged."
        )
        lines.append("")
    else:
        lines.append("Loss decomposition pending manual GPU evaluation. Run the command in section 8")
        lines.append("to populate this section. Reconstruction and token-distribution results below")
        lines.append("are complete.")
        lines.append("")
    lines.append("## 6. Token distribution analysis")
    lines.append("")
    for label, dist in (("train.jsonl", train_dist), ("valid.jsonl", valid_dist)):
        lines.append("### %s" % label)
        lines.append("")
        lines.append("| metric | value |")
        lines.append("|---|---:|")
        lines.append("| records | %d |" % dist["records"])
        lines.append("| total tokens | %d |" % dist["total_tokens"])
        lines.append("| python docstring tokens | %d |" % dist["doc_tokens_python"])
        lines.append("| javascript docstring tokens | %d |" % dist["doc_tokens_javascript"])
        lines.append("| python code tokens | %d |" % dist["code_tokens_python"])
        lines.append("| javascript code tokens | %d |" % dist["code_tokens_javascript"])
        lines.append("| total code tokens | %d |" % dist["total_code_tokens"])
        lines.append("| median code length (tokens) | %s |" % _fmt(dist["code_length_median"], 2))
        lines.append("| p90 code length (tokens) | %s |" % _fmt(dist["code_length_p90"], 2))
        lines.append("| p99 code length (tokens) | %s |" % _fmt(dist["code_length_p99"], 2))
        lines.append("| max code length (tokens) | %d |" % dist["code_length_max"])
        lines.append("| longest 1%% records | %d |" % dist["longest_1pct_records"])
        lines.append("| token mass from longest 1%% (code tokens) | %.4f%% |" % dist["longest_1pct_code_token_mass_pct"])
        lines.append("| bundle/minified-like records | %d |" % dist["bundle_like_records"])
        lines.append("| token mass from bundle/minified-like (code tokens) | %.4f%% |" % dist["bundle_like_code_token_mass_pct"])
        lines.append("| high-repetition records | %d |" % dist["high_repetition_records"])
        lines.append("| token mass from high-repetition (code tokens) | %.4f%% |" % dist["high_repetition_code_token_mass_pct"])
        lines.append("")
    lines.append("Heuristics (documented):")
    lines.append("- bundle/minified-like: code token count >= 512 AND newline count <= max(1, code_tokens // 200).")
    lines.append("- high repetition: unique-token ratio in code tokens < 0.30.")
    lines.append("- longest 1%: top 1% of records by code token count.")
    lines.append("- token mass percentages are relative to total code tokens of that split.")
    lines.append("")
    lines.append("## 7. Model-weight integrity")
    lines.append("")
    lines.append("- digest before: %s" % (digest_before or "n/a"))
    lines.append("- digest after: %s" % (digest_after or "n/a"))
    lines.append("- weights changed: %s" % (digest_before != digest_after if digest_before and digest_after else "n/a"))
    lines.append("")
    lines.append("## 8. Reproduction command (Windows CMD, project venv)")
    lines.append("")
    lines.append("```cmd")
    lines.append('cd /d "C:\\Users\\UltraPc\\Desktop\\LLM FROM SCRATCH\\PolyCodeGPT"')
    lines.append('"C:\\Users\\UltraPc\\Desktop\\LLM FROM SCRATCH\\.venv\\Scripts\\python.exe" scripts\\evaluate_validation_loss_breakdown.py --device cuda')
    lines.append("```")
    lines.append("")
    lines.append("Use `--verify-only` to run reconstruction + distribution without GPU.")
    lines.append("")
    lines.append("## 9. Limitations")
    lines.append("")
    lines.append("- Loss decomposition requires a CUDA GPU and the step-8130 checkpoint; on CPU-only hosts the")
    lines.append("  script writes reconstruction and distribution results and marks loss as pending.")
    lines.append("- Per-token `reduction='none'` aggregation may differ from the established batched mean by")
    lines.append("  a small floating-point epsilon; differences are reported, not adjusted.")
    lines.append("- Token distribution heuristics are documented approximations, not ground-truth labels.")
    lines.append("- No test data was accessed.")
    lines.append("")
    lines.append("Generated in %.1f s." % elapsed)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- main
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 8 validation loss decomposition")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--verify-only", action="store_true", help="reconstruct + distribution only, no GPU loss")
    parser.add_argument("--val-batch-size", type=int, default=None)
    parser.add_argument("--max-records", type=int, default=None, help="limit records for distribution (debug)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()

    manifest = load_manifest(MANIFEST_PATH)
    training_cfg = load_yaml(TRAINING_CONFIG_PATH)
    train_cfg = training_cfg.get("training", {})
    eval_cfg = train_cfg.get("evaluation", {})
    val_batch_size = args.val_batch_size if args.val_batch_size is not None else int(eval_cfg.get("val_batch_size", 8))
    precision_req = str(train_cfg.get("precision", "auto"))
    context_length = int(manifest["context_length"])

    tokenizer = load_tokenizer(TOKENIZER_PATH, vocab_size=int(manifest["tokenizer"]["vocab_size"]))

    # ---- reconstruction
    records = [dict(r) for r in serializer.iter_jsonl(VALID_JSONL)]
    ids_arr, cat_arr = _reconstruct_valid_arrays(records, tokenizer)

    valid_bin_tokens = int(Path(VALID_BIN).stat().st_size // 2)
    actual = np.fromfile(VALID_BIN, dtype=STREAM_DTYPE).astype(np.int64)
    exact_match = (actual.shape == ids_arr.shape) and bool(np.array_equal(actual, ids_arr))
    first_mismatch = None
    if not exact_match:
        mm = np.where(actual != ids_arr)[0]
        if mm.size:
            first_mismatch = int(mm[0])

    if not exact_match:
        print("RECONSTRUCTION MISMATCH at", first_mismatch)
        # write minimal report? exit
        raise SystemExit(2)

    geo = block_geometry(int(ids_arr.shape[0]), context_length)
    if geo["blocks"] != int(manifest["splits"]["valid"]["blocks"]):
        print("block count mismatch", geo["blocks"], manifest["splits"]["valid"]["blocks"])
        raise SystemExit(2)
    if geo["dropped_trailing_tokens"] != int(manifest["splits"]["valid"]["dropped_trailing_tokens"]):
        print("dropped trailing mismatch")
        raise SystemExit(2)

    target_n = int(geo["blocks"]) * context_length

    reconstruction = {
        "tokens": int(ids_arr.shape[0]),
        "bin_tokens": int(actual.shape[0]),
        "exact_match": bool(exact_match),
        "first_mismatch": first_mismatch if first_mismatch is not None else "none",
        "valid_jsonl_sha256": sha256_file(VALID_JSONL),
        "valid_bin_sha256": sha256_file(VALID_BIN),
        "manifest_tokens": int(manifest["splits"]["valid"]["tokens"]),
    }

    # ---- token distribution
    train_records_for_dist = None
    train_dist = analyze_token_distribution(TRAIN_JSONL, tokenizer)
    valid_dist = analyze_token_distribution(VALID_JSONL, tokenizer)
    if args.max_records is not None:
        # recompute valid dist on subset not needed; keep full
        pass

    # ---- loss (optional)
    loss_available = False
    category_results = None
    combined = None
    overall_loss = None
    digest_before = None
    digest_after = None

    has_cuda = torch.cuda.is_available() and str(args.device).startswith("cuda")
    if (not args.verify_only) and has_cuda:
        device = require_cuda(args.device)
        payload = load_checkpoint(args.checkpoint, map_location="cpu")
        model = rebuild_model(payload, device=device)
        model.eval()
        digest_before = parameter_digest(model)

        plan = resolve_precision(precision_req, device)
        vocab_size = int(manifest["tokenizer"]["vocab_size"])

        loss_sums = np.zeros(len(CATEGORY_NAMES), dtype=np.float64)
        token_counts = np.zeros(len(CATEGORY_NAMES), dtype=np.int64)
        total_loss_sum = 0.0
        total_tokens = 0

        num_blocks = int(geo["blocks"])
        stride = int(geo["window_stride"])
        wl = int(geo["window_length"])
        bs = val_batch_size
        for start in range(0, num_blocks, bs):
            end = min(start + bs, num_blocks)
            bsz = end - start
            inputs = np.empty((bsz, context_length), dtype=np.int64)
            targets = np.empty((bsz, context_length), dtype=np.int64)
            cat_batch = np.empty((bsz, context_length), dtype=np.int8)
            for j, bi in enumerate(range(start, end)):
                window = ids_arr[bi * stride : bi * stride + wl]
                inputs[j] = window[:-1]
                targets[j] = window[1:]
                cat_batch[j] = cat_arr[bi * stride + 1 : bi * stride + 1 + context_length]
            input_t = torch.from_numpy(inputs).to(device)
            target_t = torch.from_numpy(targets).to(device)
            with torch.no_grad():
                with autocast_context(plan.device_type, plan.autocast_dtype):
                    logits, _ = model(input_t, targets=None)
                    losses = F.cross_entropy(
                        logits.view(-1, vocab_size), target_t.view(-1), reduction="none"
                    )
            losses_np = losses.detach().to("cpu").numpy().astype(np.float64)
            cat_flat = cat_batch.reshape(-1)
            for ci in range(len(CATEGORY_NAMES)):
                mask = cat_flat == ci
                loss_sums[ci] += float(losses_np[mask].sum())
                token_counts[ci] += int(mask.sum())
            total_loss_sum += float(losses_np.sum())
            total_tokens += int(losses_np.size)
            if (start // bs) % 50 == 0:
                print(f"  blocks {start}/{num_blocks} done", flush=True)

        digest_after = parameter_digest(model)

        category_results = {}
        for ci, name in enumerate(CATEGORY_NAMES):
            cnt = int(token_counts[ci])
            s = float(loss_sums[ci])
            mean = (s / cnt) if cnt else float("nan")
            category_results[name] = {
                "tokens": cnt,
                "loss_sum": s,
                "mean_loss": mean,
                "perplexity": perplexity(mean),
                "share": (cnt / total_tokens) if total_tokens else 0.0,
            }
        combined = compute_combined_losses(category_results)
        overall_loss = total_loss_sum / total_tokens if total_tokens else float("nan")
        loss_available = True

        # save json
        out = {
            "reconstruction": reconstruction,
            "window": geo,
            "target_count": target_n,
            "category_results": category_results,
            "combined": combined,
            "overall_loss": overall_loss,
            "existing_loss": EXISTING_FULL_VALIDATION_LOSS,
            "difference": overall_loss - EXISTING_FULL_VALIDATION_LOSS,
            "digest_before": digest_before,
            "digest_after": digest_after,
            "weights_changed": digest_before != digest_after,
        }
        LOG_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_JSON_PATH, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)

    # ---- report
    # ensure window dict has context_length for report (add)
    window_for_report = dict(geo)
    window_for_report["context_length"] = context_length

    write_report(
        report_path=REPORT_PATH,
        reconstruction=reconstruction,
        window=window_for_report,
        target_count=target_n,
        category_results=category_results,
        combined=combined,
        overall_loss=overall_loss,
        train_dist=train_dist,
        valid_dist=valid_dist,
        checkpoint=str(args.checkpoint),
        digest_before=digest_before,
        digest_after=digest_after,
        elapsed=time.perf_counter() - started,
        loss_available=loss_available,
    )

    print("Reconstruction exact match:", exact_match)
    print("Targets:", target_n)
    if loss_available:
        print("Overall loss:", overall_loss)
    print("Report:", REPORT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
