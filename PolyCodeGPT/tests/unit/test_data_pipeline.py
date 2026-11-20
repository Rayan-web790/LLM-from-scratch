"""Unit tests for the Phase 5A tokenized data pipeline.

All fixtures are tiny hand-written records and temporary files; the real corpus
is never read here. The real tokenizer is used so that special-token behaviour
and EOS placement are tested against the artifact that will actually be used.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from polycodegpt.data.dataset import TokenBlockDataset, build_dataloader, open_split
from polycodegpt.data.integrity import sha256_file, verify_outputs
from polycodegpt.data.prepare import (
    PreparationError,
    deterministic_view,
    prepare_code_split,
    prepare_dataset,
    validate_example,
)
from polycodegpt.data.stream import (
    STREAM_DTYPE,
    TokenStreamWriter,
    block_geometry,
    dtype_is_safe,
    stream_token_count,
)
from polycodegpt.data.tokenizer_io import (
    CODE_ID,
    DOCSTRING_ID,
    EOS_ID,
    LANG_EN_ID,
    LANG_JAVASCRIPT_ID,
    LANG_PYTHON_ID,
    SPECIAL_TOKEN_IDS,
    TOKENIZER_METADATA_PATH,
    encode,
    load_tokenizer,
)
from polycodegpt.tokenization.serializer import serialize_code_record, serialize_documentation

# --------------------------------------------------------------------------- #
# helpers / fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def tok():
    return load_tokenizer(vocab_size=16000)


def write_stream(path: Path, ids, vocab_size: int = 16000) -> Path:
    with TokenStreamWriter(path, vocab_size) as writer:
        writer.write(ids)
    return path


def python_record(i: int) -> dict:
    return {
        "language": "python",
        "source_dataset": "CodeSearchNet Python",
        "repo": f"example/repo{i}",
        "path": f"mod_{i}.py",
        "func_name": f"fn_{i}",
        "docstring": f"Return the value {i}.",
        "code": f"def fn_{i}(x):\n    return x + {i}\n",
    }


def javascript_record(i: int) -> dict:
    return {
        "language": "javascript",
        "source_dataset": "CodeSearchNet JavaScript",
        "repo": f"example/js{i}",
        "path": f"mod_{i}.js",
        "func_name": f"fn{i}",
        "docstring": f"Add {i} to a number.",
        "code": f"function fn{i}(x) {{\n  return x + {i};\n}}\n",
    }


def write_jsonl(path: Path, records) -> Path:
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def write_docs_inventory(path: Path, doc_files) -> Path:
    fields = [
        "doc_id", "source_id", "version", "original_repository_url", "local_file_path",
        "source_license", "file_sha256", "file_size_bytes", "char_count",
        "natural_language", "content_type", "corpus_role", "partition_assignment",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for i, doc in enumerate(doc_files):
            writer.writerow(
                {
                    "doc_id": f"doc{i}",
                    "source_id": "PYDOCS",
                    "version": "3.11.9",
                    "original_repository_url": "https://example.invalid/doc.rst",
                    "local_file_path": str(doc),
                    "source_license": "PSF-2.0",
                    "file_sha256": sha256_file(doc),
                    "file_size_bytes": doc.stat().st_size,
                    "char_count": len(doc.read_text(encoding="utf-8")),
                    "natural_language": "en",
                    "content_type": "rst",
                    "corpus_role": "documentation",
                    "partition_assignment": "train_only",
                }
            )
    return path


def write_model_config(path: Path, context_length: int = 8) -> Path:
    path.write_text(
        "model:\n"
        "  architecture: decoder_only_transformer\n"
        "  vocab_size: 16000\n"
        "  d_model: 16\n"
        "  n_layers: 1\n"
        "  n_heads: 4\n"
        "  ffn_dim: 32\n"
        f"  context_length: {context_length}\n"
        "  dropout: 0.0\n"
        "  tie_embeddings: true\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture()
def synthetic_dataset(tmp_path):
    """A complete tiny dataset built through the real preparation path."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    write_jsonl(corpus / "train.jsonl", [python_record(i) for i in range(6)] + [javascript_record(i) for i in range(4)])
    write_jsonl(corpus / "valid.jsonl", [python_record(i) for i in range(100, 104)])
    write_jsonl(corpus / "test.jsonl", [javascript_record(i) for i in range(200, 204)])

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    docs = []
    for i in range(2):
        doc = docs_dir / f"doc{i}.rst"
        doc.write_text(f"Section {i}\n=========\n\nSome documentation text number {i}.\n", encoding="utf-8")
        docs.append(doc)
    inventory = write_docs_inventory(tmp_path / "inventory.csv", docs)

    config = write_model_config(tmp_path / "model.yaml", context_length=8)
    out = tmp_path / "tokenized"
    manifest = prepare_dataset(
        output_dir=out,
        config_path=config,
        corpus_dir=corpus,
        docs_inventory=inventory,
        verify_sources=False,
    )
    return {"dir": out, "manifest": manifest, "config": config}


# --------------------------------------------------------------------------- #
# shift / shapes / dtype
# --------------------------------------------------------------------------- #
def test_next_token_shift_is_exact(tmp_path):
    tokens = list(range(100))
    path = write_stream(tmp_path / "s.bin", tokens)
    ds = TokenBlockDataset(path, context_length=8)
    # stride == context_length: 1 + (100 - 9) // 8 blocks
    assert len(ds) == 12
    for i in range(len(ds)):
        input_ids, targets = ds[i]
        start = i * 8
        assert input_ids.tolist() == tokens[start : start + 8]
        assert targets.tolist() == tokens[start + 1 : start + 9]
        # target[t] is exactly the token after input[t]
        assert targets[:-1].tolist() == input_ids[1:].tolist()
        for t in range(8):
            assert int(targets[t]) == tokens[start + t + 1]


def test_sample_shapes_and_dtypes(tmp_path):
    path = write_stream(tmp_path / "s.bin", list(range(50)))
    ds = TokenBlockDataset(path, context_length=8)
    input_ids, targets = ds[0]
    assert input_ids.shape == (8,)
    assert targets.shape == (8,)
    assert input_ids.dtype == torch.int64
    assert targets.dtype == torch.int64
    assert input_ids.is_contiguous() and targets.is_contiguous()


def test_stream_dtype_is_uint16(tmp_path):
    path = write_stream(tmp_path / "s.bin", [1, 2, 3, 15999])
    assert STREAM_DTYPE.itemsize == 2
    assert path.stat().st_size == 4 * 2
    assert dtype_is_safe(16000) and not dtype_is_safe(70000)


def test_block_counting_and_remainder(tmp_path):
    n = 1000
    path = write_stream(tmp_path / "s.bin", list(range(n)))
    ds = TokenBlockDataset(path, context_length=512)
    assert ds.num_blocks == 1
    assert ds.dropped_trailing_tokens == 1000 - 513
    assert ds.token_count == n

    geometry = block_geometry(1026, 512)
    assert geometry == {
        "window_length": 513,
        "window_stride": 512,
        "window_overlap": 1,
        "block_span": 513,
        "blocks": 2,
        "dropped_trailing_tokens": 1,
        "tokens_in_blocks": 1025,
    }
    # every block is full-length, and dropped tokens complete no block
    assert ds.num_blocks * ds.window_stride + 1 + ds.dropped_trailing_tokens == n


def test_short_stream_yields_no_blocks(tmp_path):
    path = write_stream(tmp_path / "s.bin", list(range(9)))  # 8 + 1 needed
    ds = TokenBlockDataset(path, context_length=8)
    assert len(ds) == 1
    assert ds.dropped_trailing_tokens == 0
    path2 = write_stream(tmp_path / "s2.bin", list(range(8)))
    ds2 = TokenBlockDataset(path2, context_length=8)
    assert len(ds2) == 0
    assert ds2.dropped_trailing_tokens == 8


# --------------------------------------------------------------------------- #
# window stride = context_length (overlap of exactly one token)
# --------------------------------------------------------------------------- #
def test_first_window_uses_tokens_zero_to_context_length(tmp_path):
    tokens = list(range(200))
    ds = TokenBlockDataset(write_stream(tmp_path / "s.bin", tokens), context_length=8)
    input_ids, targets = ds[0]
    assert ds.window_length == 9
    assert input_ids.tolist() == tokens[0:8]
    assert targets.tolist() == tokens[1:9]
    assert ds.block(0).tolist() == tokens[0:9]


def test_second_window_starts_at_context_length(tmp_path):
    tokens = list(range(200))
    ds = TokenBlockDataset(write_stream(tmp_path / "s.bin", tokens), context_length=8)
    input_ids, targets = ds[1]
    assert ds.window_stride == ds.context_length == 8
    assert input_ids.tolist() == tokens[8:16]
    assert targets.tolist() == tokens[9:17]
    assert ds.block(1).tolist() == tokens[8:17]


def test_consecutive_windows_overlap_by_exactly_one_token(tmp_path):
    tokens = list(range(200))
    ds = TokenBlockDataset(write_stream(tmp_path / "s.bin", tokens), context_length=8)
    assert ds.window_overlap == 1
    for i in range(len(ds) - 1):
        current, following = ds.block(i), ds.block(i + 1)
        # the only shared token is the boundary token tokens[(i+1)*stride]
        assert current[-1] == following[0] == tokens[(i + 1) * 8]
        assert set(current[:-1]).isdisjoint(following[1:])


def test_every_target_is_used_exactly_once(tmp_path):
    tokens = list(range(60))
    ds = TokenBlockDataset(write_stream(tmp_path / "s.bin", tokens), context_length=8)
    seen: list[int] = []
    for i in range(len(ds)):
        seen.extend(ds[i][1].tolist())

    used = ds.num_blocks * ds.window_stride + 1
    # every position after the first appears as a target exactly once ...
    assert sorted(seen) == list(range(1, used))
    assert len(seen) == len(set(seen))
    # ... token 0 is context only, and the tail never appears as a target
    assert 0 not in seen
    assert all(t < used for t in seen)
    assert used + ds.dropped_trailing_tokens == ds.token_count


def test_no_target_is_skipped_at_block_boundaries(tmp_path):
    tokens = list(range(60))
    ds = TokenBlockDataset(write_stream(tmp_path / "s.bin", tokens), context_length=8)
    for i in range(len(ds) - 1):
        boundary = int(ds[i][1][-1])
        # the target at the last position of block i is the first input of block i+1
        assert boundary == int(ds[i + 1][0][0])
        assert boundary == tokens[(i + 1) * 8]
        # and the position right after it is predicted by block i+1
        assert int(ds[i + 1][1][0]) == tokens[(i + 1) * 8 + 1]


def test_final_tail_is_counted_and_never_read(tmp_path):
    tokens = list(range(64))
    ds = TokenBlockDataset(write_stream(tmp_path / "s.bin", tokens), context_length=8)
    used = ds.num_blocks * ds.window_stride + 1
    assert ds.num_blocks == 1 + (64 - 9) // 8
    assert used == 57
    assert ds.dropped_trailing_tokens == 64 - 57 == 7
    assert ds.token_count == ds.num_blocks * ds.window_stride + 1 + ds.dropped_trailing_tokens
    # the tail is beyond the last window
    last = ds.block(len(ds) - 1)
    assert int(last[-1]) == used - 1


def test_geometry_rejects_a_stride_that_would_skip_targets():
    with pytest.raises(ValueError, match="would skip targets"):
        block_geometry(1000, 8, stride=10)
    with pytest.raises(ValueError, match="positive"):
        block_geometry(1000, 8, stride=0)
    # stride shorter than the context is allowed (more overlap, repeated targets)
    dense = block_geometry(1000, 8, stride=4)
    assert dense["window_overlap"] == 5 and dense["window_stride"] == 4


def test_open_split_uses_manifest_stride(synthetic_dataset):
    manifest = synthetic_dataset["manifest"]
    root = synthetic_dataset["dir"]
    assert manifest["window_stride"] == manifest["context_length"]
    for split in ("train", "valid", "test"):
        ds = open_split(root, split, manifest)
        assert ds.window_stride == manifest["window_stride"]
        assert ds.window_length == manifest["window_length"]
        assert ds.window_overlap == manifest["window_overlap"] == 1
        assert len(ds) == manifest["splits"][split]["blocks"]

    # a manifest claiming a different stride must be rejected, not silently trusted
    tampered = json.loads(json.dumps(manifest))
    tampered["window_stride"] = manifest["window_stride"] + 1
    with pytest.raises(ValueError):
        open_split(root, "train", tampered)


def test_splits_stay_independent_under_overlap(synthetic_dataset):
    """Overlap is within a split only: every window lives inside its own file."""
    manifest = synthetic_dataset["manifest"]
    root = synthetic_dataset["dir"]
    paths = set()
    for split, info in manifest["splits"].items():
        ds = open_split(root, split, manifest)
        raw = np.memmap(root / info["file"], dtype=STREAM_DTYPE, mode="r")
        paths.add(ds.bin_path)
        for index in range(len(ds)):
            start = index * ds.window_stride
            assert start + ds.window_length <= ds.token_count
            assert ds.block(index).tolist() == raw[start : start + ds.window_length].tolist()
        # a window may never read past its own stream
        assert ds.block(len(ds) - 1)[-1] == int(raw[ds.token_count - 1 - ds.dropped_trailing_tokens])
        ds.close()
    # splits were not merged: each split reads its own file
    assert len(paths) == 3


def test_memmap_offsets_follow_the_stride(tmp_path):
    tokens = list(range(64))
    path = write_stream(tmp_path / "s.bin", tokens)
    ds = TokenBlockDataset(path, context_length=8)
    assert ds._mm is None
    _ = ds[1]
    assert ds._mm is not None
    # block 1 reads mm[8:17], not mm[9:18] as a non-overlapping layout would
    assert ds.mm[8:17].tolist() == tokens[8:17]
    assert ds.mm[9:18].tolist() == tokens[9:18]
    ds.close()


def test_indexing_is_deterministic(tmp_path, synthetic_dataset):
    tokens = list(range(64))
    ds = TokenBlockDataset(write_stream(tmp_path / "s.bin", tokens), context_length=8)
    for index in (0, 1, len(ds) - 1):
        first = ds[index]
        second = ds[index]
        assert first[0].tolist() == second[0].tolist()
        assert first[1].tolist() == second[1].tolist()

    manifest = synthetic_dataset["manifest"]
    train = open_split(synthetic_dataset["dir"], "train", manifest)
    loader = build_dataloader(train, batch_size=2, shuffle=True, pin_memory=False, seed=11)
    epoch_a = [batch[0].tolist() for batch in loader]
    epoch_b = [batch[0].tolist() for batch in loader]
    assert epoch_a != epoch_b  # reshuffled between epochs
    # a loader rebuilt with the same seed reproduces the same order
    reloaded = build_dataloader(train, batch_size=2, shuffle=True, pin_memory=False, seed=11)
    assert [b[0].tolist() for b in reloaded] == epoch_a


# --------------------------------------------------------------------------- #
# invalid IDs / stream writer contract
# --------------------------------------------------------------------------- #
def test_writer_rejects_out_of_vocabulary_ids(tmp_path):
    with pytest.raises(ValueError, match="outside vocabulary"):
        write_stream(tmp_path / "bad.bin", [0, 1, 100], vocab_size=100)
    with pytest.raises(ValueError, match="outside vocabulary"):
        write_stream(tmp_path / "neg.bin", [-1, 5], vocab_size=100)


def test_writer_rejects_non_positive_vocab(tmp_path):
    with pytest.raises(ValueError):
        TokenStreamWriter(tmp_path / "x.bin", vocab_size=0)


def test_no_invalid_ids_in_prepared_stream(synthetic_dataset, tok):
    manifest = synthetic_dataset["manifest"]
    vocab_size = manifest["tokenizer"]["vocab_size"]
    for split, info in manifest["splits"].items():
        arr = np.memmap(synthetic_dataset["dir"] / info["file"], dtype=STREAM_DTYPE, mode="r")
        assert arr.size == info["tokens"]
        assert int(arr.min()) >= 0
        assert int(arr.max()) < vocab_size


# --------------------------------------------------------------------------- #
# example contract / EOS
# --------------------------------------------------------------------------- #
def test_example_contract_accepts_good_example(tok):
    ids = encode(tok, serialize_code_record(python_record(0)))
    validate_example(ids, 16000, "good")  # must not raise
    assert ids[0] == LANG_PYTHON_ID
    assert ids[-1] == EOS_ID


def test_example_contract_rejects_missing_eos(tok):
    ids = encode(tok, serialize_code_record(python_record(0)))[:-1]
    with pytest.raises(PreparationError, match="not EOS"):
        validate_example(ids, 16000, "no-eos")


def test_example_contract_rejects_double_eos(tok):
    ids = encode(tok, serialize_code_record(python_record(0))) + [EOS_ID]
    with pytest.raises(PreparationError, match="exactly one EOS"):
        validate_example(ids, 16000, "double-eos")


def test_example_contract_rejects_bad_first_token(tok):
    ids = encode(tok, serialize_code_record(python_record(0)))[1:]
    with pytest.raises(PreparationError, match="language marker"):
        validate_example(ids, 16000, "no-marker")


def test_example_contract_rejects_out_of_vocab(tok):
    ids = encode(tok, serialize_code_record(python_record(0)))
    ids[3] = 16000
    with pytest.raises(PreparationError, match="outside vocabulary"):
        validate_example(ids, 16000, "oov")


def test_serializer_eos_is_not_duplicated():
    """The serializer already ends with EOS; the pipeline must not add another."""
    for record in (python_record(1), javascript_record(1)):
        text = serialize_code_record(record)
        assert text.endswith("<|eos|>")
        assert text.count("<|eos|>") == 1
    assert serialize_documentation("hello\n").count("<|eos|>") == 1


def test_eos_separates_examples_in_stream(synthetic_dataset, tok):
    manifest = synthetic_dataset["manifest"]
    train = manifest["splits"]["train"]
    arr = np.memmap(synthetic_dataset["dir"] / "train.bin", dtype=STREAM_DTYPE, mode="r")
    eos_total = int(np.count_nonzero(arr == EOS_ID))
    assert eos_total == train["records"] + train["documents"]

    text = tok.decode(arr[:200].tolist(), skip_special_tokens=False)
    assert "<|eos|>" in text
    # EOS never appears doubled back-to-back
    assert "<|eos|><|eos|>" not in text


# --------------------------------------------------------------------------- #
# split separation
# --------------------------------------------------------------------------- #
def test_no_split_mixing(synthetic_dataset):
    manifest = synthetic_dataset["manifest"]
    root = synthetic_dataset["dir"]

    def count(path, token_id):
        arr = np.memmap(path, dtype=STREAM_DTYPE, mode="r")
        return int(np.count_nonzero(arr == token_id))

    train, valid, test = (manifest["splits"][s] for s in ("train", "valid", "test"))
    # documentation exists only in train
    assert count(root / train["file"], LANG_EN_ID) == 2
    assert count(root / valid["file"], LANG_EN_ID) == 0
    assert count(root / test["file"], LANG_EN_ID) == 0
    # code markers match each split's record counts exactly
    assert count(root / train["file"], LANG_PYTHON_ID) == 6
    assert count(root / train["file"], LANG_JAVASCRIPT_ID) == 4
    assert count(root / valid["file"], LANG_PYTHON_ID) == 4
    assert count(root / valid["file"], LANG_JAVASCRIPT_ID) == 0
    assert count(root / test["file"], LANG_JAVASCRIPT_ID) == 4
    assert count(root / test["file"], LANG_PYTHON_ID) == 0


def test_only_approved_documents_are_included(tmp_path, tok):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    write_jsonl(corpus / "train.jsonl", [python_record(0)])
    write_jsonl(corpus / "valid.jsonl", [python_record(1)])
    write_jsonl(corpus / "test.jsonl", [python_record(2)])

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    approved = docs_dir / "approved.rst"
    approved.write_text("Approved documentation.\n", encoding="utf-8")
    excluded = docs_dir / "heldout.rst"
    excluded.write_text("Held-out documentation.\n", encoding="utf-8")

    inventory = write_docs_inventory(tmp_path / "inv.csv", [approved])
    rows = list(csv.DictReader(open(inventory, newline="", encoding="utf-8")))
    rows.append({**rows[0], "doc_id": "heldout", "local_file_path": str(excluded),
                 "file_sha256": sha256_file(excluded), "partition_assignment": "eval_only"})
    with open(inventory, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    config = write_model_config(tmp_path / "m.yaml", context_length=8)
    manifest = prepare_dataset(
        output_dir=tmp_path / "out",
        config_path=config,
        corpus_dir=corpus,
        docs_inventory=inventory,
        verify_sources=False,
    )
    assert manifest["splits"]["train"]["documents"] == 1
    arr = np.memmap(tmp_path / "out" / "train.bin", dtype=STREAM_DTYPE, mode="r")
    assert int(np.count_nonzero(arr == LANG_EN_ID)) == 1
    text = tok.decode(arr.tolist(), skip_special_tokens=False)
    assert "Approved documentation." in text
    assert "Held-out documentation." not in text


# --------------------------------------------------------------------------- #
# dataset / memmap behaviour
# --------------------------------------------------------------------------- #
def test_memmap_indexing_is_lazy_and_bounds_checked(tmp_path):
    path = write_stream(tmp_path / "s.bin", list(range(60)))
    ds = TokenBlockDataset(path, context_length=8)
    assert ds._mm is None  # not opened by construction
    _ = ds[0]
    assert ds._mm is not None

    assert ds[-1][0].tolist() == ds[len(ds) - 1][0].tolist()
    with pytest.raises(IndexError):
        _ = ds[len(ds)]

    ds.close()
    assert ds._mm is None
    # still usable after an explicit close (reopens on demand)
    assert ds[0][0].shape == (8,)


def test_dataset_does_not_hold_tokens_in_memory(tmp_path):
    path = write_stream(tmp_path / "s.bin", list(range(60)))
    ds = TokenBlockDataset(path, context_length=8)
    payload = [v for k, v in ds.__dict__.items() if isinstance(v, (list, tuple, dict, np.ndarray))]
    assert payload == []
    assert isinstance(ds.__getstate__()["_mm"], type(None))


def test_missing_stream_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        TokenBlockDataset(tmp_path / "nope.bin", context_length=8)


# --------------------------------------------------------------------------- #
# manifest / determinism / integrity
# --------------------------------------------------------------------------- #
def test_manifest_has_required_fields(synthetic_dataset):
    manifest = synthetic_dataset["manifest"]
    for key in (
        "manifest_version", "dataset_version", "completed_successfully", "context_length",
        "dtype", "byte_order", "tokenizer", "serializer", "splits", "totals", "failures",
        "software", "run",
    ):
        assert key in manifest, key

    assert manifest["tokenizer"]["version"] == "1.1.0"
    assert manifest["tokenizer"]["sha256"] == "0459f8556a6f79ca1ba1c3dc3f76c26ae2e55653ebbf97beaa9bd23467358a03"
    assert manifest["serializer"]["version"] == "1.1.0"
    assert len(manifest["serializer"]["sha256"]) == 64
    assert manifest["context_length"] == 8
    assert manifest["manifest_version"] == "1.1.0"
    assert manifest["window_length"] == 9
    assert manifest["window_stride"] == 8
    assert manifest["window_overlap"] == 1
    assert manifest["block_span"] == 9
    assert manifest["block_scheme"] == "overlapping"
    assert manifest["dtype"] == "uint16" and manifest["byte_order"] == "little"
    assert manifest["completed_successfully"] is True
    assert manifest["failures"] == []

    for split in manifest["splits"].values():
        for key in ("file", "tokens", "blocks", "dropped_trailing_tokens",
                    "file_size_bytes", "sha256", "records", "documents", "sources"):
            assert key in split, key
        assert split["file_size_bytes"] == split["tokens"] * 2
        assert (
            split["tokens"]
            == split["blocks"] * manifest["window_stride"]
            + manifest["context_length"]
            + 1
            - manifest["window_stride"]
            + split["dropped_trailing_tokens"]
        )


def test_preparation_is_deterministic(tmp_path, synthetic_dataset):
    second = tmp_path / "second"
    manifest2 = prepare_dataset(
        output_dir=second,
        config_path=synthetic_dataset["config"],
        corpus_dir=tmp_path / "corpus",
        docs_inventory=tmp_path / "inventory.csv",
        verify_sources=False,
    )
    assert deterministic_view(manifest2) == deterministic_view(synthetic_dataset["manifest"])
    for split, info in manifest2["splits"].items():
        assert sha256_file(second / info["file"]) == info["sha256"]


def test_verify_outputs_and_corruption_detection(synthetic_dataset):
    manifest = synthetic_dataset["manifest"]
    root = synthetic_dataset["dir"]
    assert verify_outputs(manifest, root)["ok"] is True

    target = root / manifest["splits"]["valid"]["file"]
    data = bytearray(target.read_bytes())
    data[0] ^= 0xFF
    target.write_bytes(bytes(data))

    report = verify_outputs(manifest, root)
    assert report["ok"] is False
    assert any(c["split"] == "valid" and not c["ok"] for c in report["splits"])


def test_source_hash_mismatch_aborts(tmp_path, tok):
    corpus = write_jsonl(tmp_path / "train.jsonl", [python_record(0)])
    with pytest.raises(PreparationError, match="SHA-256 mismatch"):
        prepare_code_split(
            "train", corpus, tok, 16000, tmp_path / "out.bin", 8,
            expected_sha256="0" * 64,
        )


def test_missing_source_file_raises(tmp_path, tok):
    with pytest.raises(FileNotFoundError):
        prepare_code_split("train", tmp_path / "absent.jsonl", tok, 16000, tmp_path / "o.bin", 8)


# --------------------------------------------------------------------------- #
# dataloader
# --------------------------------------------------------------------------- #
def test_dataloader_settings(synthetic_dataset):
    manifest = synthetic_dataset["manifest"]
    root = synthetic_dataset["dir"]

    train_ds = open_split(root, "train", manifest)
    valid_ds = open_split(root, "valid", manifest)

    train_loader = build_dataloader(train_ds, batch_size=1, shuffle=True, pin_memory=False, seed=7)
    valid_loader = build_dataloader(valid_ds, batch_size=1, shuffle=False, pin_memory=False)

    from torch.utils.data import RandomSampler, SequentialSampler

    assert isinstance(train_loader.sampler, RandomSampler)
    assert isinstance(valid_loader.sampler, SequentialSampler)
    assert valid_loader.batch_size == 1
    assert len(train_loader) == len(train_ds) // 1

    def first_indices(loader):
        return [int(batch[0][0, 0]) for batch in loader]

    order_a = first_indices(train_loader)
    order_b = first_indices(train_loader)
    assert len(order_a) == len(train_ds)
    assert sorted(order_a) == sorted(order_b)  # same blocks, different order
    assert order_a != order_b  # reshuffled between epochs

    valid_a = first_indices(valid_loader)
    valid_b = first_indices(valid_loader)
    assert valid_a == valid_b  # never shuffled


def test_dataloader_defaults_pin_memory_and_workers(synthetic_dataset):
    manifest = synthetic_dataset["manifest"]
    ds = open_split(synthetic_dataset["dir"], "valid", manifest)
    loader = build_dataloader(ds, batch_size=2, shuffle=False)
    assert loader.pin_memory == torch.cuda.is_available()
    assert loader.num_workers == 0


def test_dataloader_batch_shapes(synthetic_dataset):
    manifest = synthetic_dataset["manifest"]
    ds = open_split(synthetic_dataset["dir"], "train", manifest)
    loader = build_dataloader(ds, batch_size=2, shuffle=False, pin_memory=False)
    input_ids, targets = next(iter(loader))
    assert input_ids.shape == (2, manifest["context_length"])
    assert targets.shape == (2, manifest["context_length"])
    assert input_ids.dtype == torch.int64


# --------------------------------------------------------------------------- #
# tokenizer contract
# --------------------------------------------------------------------------- #
def test_special_token_ids_match_metadata(tok):
    metadata = json.load(open(TOKENIZER_METADATA_PATH, encoding="utf-8"))
    recorded = metadata["special_tokens"]
    assert recorded == dict(SPECIAL_TOKEN_IDS)
    vocab = tok.get_vocab()
    for token, expected_id in SPECIAL_TOKEN_IDS.items():
        assert vocab[token] == expected_id
        assert tok.encode(token, add_special_tokens=False).ids == [expected_id]
    assert tok.get_vocab_size() == 16000


def test_language_markers_and_role_markers_encode_as_single_ids(tok):
    ids = encode(tok, serialize_code_record(python_record(3)))
    assert ids[0] == LANG_PYTHON_ID
    assert DOCSTRING_ID in ids
    assert CODE_ID in ids
    js = encode(tok, serialize_code_record(javascript_record(3)))
    assert js[0] == LANG_JAVASCRIPT_ID
    doc = encode(tok, serialize_documentation("text\n"))
    assert doc[0] == LANG_EN_ID


def test_encoded_examples_round_trip_through_decoder(tok):
    for record in (python_record(5), javascript_record(5)):
        text = serialize_code_record(record)
        assert tok.decode(encode(tok, text), skip_special_tokens=False) == text
