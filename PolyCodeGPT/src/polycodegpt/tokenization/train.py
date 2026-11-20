"""Train the PolyCodeGPT byte-level BPE tokenizer (serializer 1.1.0).

Inputs (train partition only):
  * data/processed/polycodegpt_corpus/train.jsonl          (392,632 records)
  * 500 eligible Python documentation files listed in
    data/manifests/python_docs_training_corpus_inventory.csv

``valid.jsonl`` and ``test.jsonl`` are never opened. Every path that reaches an
``open()`` call in this module comes from the verified input manifest, and
``_assert_train_only`` rejects anything outside the training partition before a
file is touched.

Outputs (tokenizer/artifacts/):
  polycodegpt_tokenizer_v1_1.json      versioned artifact
  tokenizer_metadata_v1_1.json         versioned metadata
  training_input_manifest_v1_1.csv     versioned input audit manifest
  plus unversioned copies of the same three files for downstream use.
"""

from __future__ import annotations

import csv
import ctypes
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable

import tokenizers
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from .serializer import (
    SERIALIZER_VERSION,
    SPECIAL_TOKENS,
    iter_training_texts,
    load_doc_inventory,
    sha256_file,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAIN_JSONL = PROJECT_ROOT / "data" / "processed" / "polycodegpt_corpus" / "train.jsonl"
CORPUS_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "polycodegpt_corpus_manifest.csv"
DOC_INVENTORY = PROJECT_ROOT / "data" / "manifests" / "python_docs_training_corpus_inventory.csv"
SERIALIZER_PATH = Path(__file__).resolve().parent / "serializer.py"
ARTIFACT_DIR = PROJECT_ROOT / "tokenizer" / "artifacts"

ARTIFACT_VERSION = "1.1"
TOKENIZER_JSON = ARTIFACT_DIR / "polycodegpt_tokenizer_v1_1.json"
METADATA_JSON = ARTIFACT_DIR / "tokenizer_metadata_v1_1.json"
MANIFEST_CSV = ARTIFACT_DIR / "training_input_manifest_v1_1.csv"
CANONICAL_TOKENIZER = ARTIFACT_DIR / "polycodegpt_tokenizer.json"
CANONICAL_METADATA = ARTIFACT_DIR / "tokenizer_metadata.json"
CANONICAL_MANIFEST = ARTIFACT_DIR / "training_input_manifest.csv"

EXPECTED_TRAIN_RECORDS = 392_632
EXPECTED_DOCS = 500
VOCAB_SIZE = 16_000
MIN_FREQUENCY = 2
SEED = 1234
FORBIDDEN_NAMES = {"valid.jsonl", "test.jsonl"}

COLLECTION_CODE = "CodeSearchNet Python + CodeSearchNet JavaScript (Zenodo 10.5281/zenodo.7908468)"
COLLECTION_DOCS = "CPython 3.11.9 Doc/ (PYDOCS, PSF-2.0)"


# --------------------------------------------------------------------------- #
# guards and measurement
# --------------------------------------------------------------------------- #
def _assert_train_only(path: str | Path) -> None:
    """Refuse any path that is not an explicit training-partition input."""
    p = Path(path)
    if p.name in FORBIDDEN_NAMES:
        raise RuntimeError(f"refusing to read evaluation file {p}")
    if "/valid/" in p.as_posix() or "/test/" in p.as_posix():
        raise RuntimeError(f"refusing to read non-training path {p}")


def peak_rss_mb() -> float:
    """Peak resident set size of this process in MiB (0.0 if unavailable)."""
    if os.name == "nt":
        class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        fn = getattr(psapi, "GetProcessMemoryInfo", None) or kernel32.K32GetProcessMemoryInfo
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX), ctypes.c_ulong]
        fn.restype = ctypes.c_int
        handle = kernel32.GetCurrentProcess()
        if fn(handle, ctypes.byref(counters), counters.cb):
            return counters.PeakWorkingSetSize / (1024 * 1024)
        return 0.0
    try:  # pragma: no cover - non-Windows fallback
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:
        return 0.0


# --------------------------------------------------------------------------- #
# input verification and manifest
# --------------------------------------------------------------------------- #
def expected_train_sha256() -> str:
    with open(CORPUS_MANIFEST, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["partition"] == "train":
                return row["sha256"]
    raise RuntimeError("train row missing from corpus manifest")


def verify_inputs(docs: Iterable[dict], verbose: bool = True) -> dict:
    """Verify every training input against its recorded SHA-256."""
    _assert_train_only(TRAIN_JSONL)
    train_sha = sha256_file(TRAIN_JSONL)
    expected = expected_train_sha256()
    if train_sha != expected:
        raise RuntimeError(f"train.jsonl hash mismatch: {train_sha} != {expected}")

    serializer_sha = sha256_file(SERIALIZER_PATH)

    doc_stats = []
    for doc in docs:
        _assert_train_only(doc["local_path"])
        p = Path(doc["local_path"])
        if not p.is_file():
            raise FileNotFoundError(f"inventory document missing: {p}")
        actual = sha256_file(p)
        if actual != doc["sha256"]:
            raise RuntimeError(f"document hash mismatch for {doc['rel_path']}: {actual} != {doc['sha256']}")
        if p.name == "license.rst" and p.parent.name == "Doc":
            raise RuntimeError("Doc/license.rst must never be used")
        doc_stats.append({**doc, "verified_sha256": actual})
    if verbose:
        print(f"  train.jsonl sha256 OK ({train_sha[:16]}…) size={TRAIN_JSONL.stat().st_size:,} B")
        print(f"  serializer.py sha256 {serializer_sha[:16]}… (version {SERIALIZER_VERSION})")
        print(f"  {len(doc_stats)} documentation files verified against the inventory")
    return {"train_sha256": train_sha, "serializer_sha256": serializer_sha, "docs": doc_stats}


def write_input_manifest(docs: Iterable[dict], train_sha256: str, serializer_sha256: str) -> Path:
    columns = [
        "input_kind",
        "source_path",
        "source_sha256",
        "record_count",
        "partition",
        "serializer_version",
        "serializer_sha256",
        "collection_source",
        "documentation_eligibility",
    ]
    rows = [
        {
            "input_kind": "combined_train_jsonl",
            "source_path": TRAIN_JSONL.relative_to(PROJECT_ROOT).as_posix(),
            "source_sha256": train_sha256,
            "record_count": EXPECTED_TRAIN_RECORDS,
            "partition": "train",
            "serializer_version": SERIALIZER_VERSION,
            "serializer_sha256": serializer_sha256,
            "collection_source": COLLECTION_CODE,
            "documentation_eligibility": "n/a",
        },
        {
            "input_kind": "serializer_source",
            "source_path": "src/polycodegpt/tokenization/serializer.py",
            "source_sha256": serializer_sha256,
            "record_count": 0,
            "partition": "train",
            "serializer_version": SERIALIZER_VERSION,
            "serializer_sha256": serializer_sha256,
            "collection_source": "PolyCodeGPT project source (deterministic serialization logic)",
            "documentation_eligibility": "n/a",
        },
    ]
    for doc in docs:
        rows.append(
            {
                "input_kind": "documentation_rst",
                "source_path": doc["rel_path"],
                "source_sha256": doc["sha256"],
                "record_count": 1,
                "partition": "train",
                "serializer_version": SERIALIZER_VERSION,
                "serializer_sha256": serializer_sha256,
                "collection_source": COLLECTION_DOCS,
                "documentation_eligibility": "ELIGIBLE_CANDIDATE (train_only)",
            }
        )
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST_CSV.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, MANIFEST_CSV)
    return MANIFEST_CSV


# --------------------------------------------------------------------------- #
# tokenizer construction and training
# --------------------------------------------------------------------------- #
def build_tokenizer() -> Tokenizer:
    """Byte-level BPE: no normalizer, no lowercasing, indent-preserving."""
    tok = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens(list(SPECIAL_TOKENS))
    return tok


def build_trainer(vocab_size: int = VOCAB_SIZE, min_frequency: int = MIN_FREQUENCY) -> trainers.BpeTrainer:
    return trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )


def train(
    vocab_size: int = VOCAB_SIZE,
    min_frequency: int = MIN_FREQUENCY,
    verbose: bool = True,
) -> dict:
    """Run the full verified training pipeline and save artifacts."""
    t_start = time.perf_counter()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    docs = load_doc_inventory(DOC_INVENTORY, PROJECT_ROOT)
    if len(docs) != EXPECTED_DOCS:
        raise RuntimeError(f"expected {EXPECTED_DOCS} eligible documents, inventory has {len(docs)}")
    if verbose:
        print(f"inputs: {TRAIN_JSONL.relative_to(PROJECT_ROOT)} + {len(docs)} documents")
    verified = verify_inputs(docs, verbose=verbose)
    write_input_manifest(docs, verified["train_sha256"], verified["serializer_sha256"])
    if verbose:
        print(f"  manifest: {MANIFEST_CSV.relative_to(PROJECT_ROOT)}")

    counters: dict = {}
    total = EXPECTED_TRAIN_RECORDS + len(docs)
    texts = iter_training_texts(TRAIN_JSONL, verified["docs"], counters=counters)

    tok = build_tokenizer()
    trainer = build_trainer(vocab_size=vocab_size, min_frequency=min_frequency)
    if verbose:
        print(f"training: vocab_size={vocab_size} min_frequency={min_frequency} "
              f"special_tokens={len(SPECIAL_TOKENS)} serializer={SERIALIZER_VERSION} sequences={total:,}")

    t_fit = time.perf_counter()
    tok.train_from_iterator(texts, trainer=trainer, length=total)
    fit_s = time.perf_counter() - t_fit

    if counters.get("code_records") != EXPECTED_TRAIN_RECORDS:
        raise RuntimeError(f"serialized {counters.get('code_records')} code records, expected {EXPECTED_TRAIN_RECORDS}")
    if counters.get("documents") != len(docs):
        raise RuntimeError(f"serialized {counters.get('documents')} documents, expected {len(docs)}")

    tok.add_special_tokens(list(SPECIAL_TOKENS))

    vocab = tok.get_vocab()
    special_ids = {t: vocab[t] for t in SPECIAL_TOKENS}
    if len(set(special_ids.values())) != len(SPECIAL_TOKENS):
        raise RuntimeError(f"special tokens do not have unique IDs: {special_ids}")

    tok.save(str(TOKENIZER_JSON))
    total_s = time.perf_counter() - t_start

    raw = json.loads(TOKENIZER_JSON.read_text(encoding="utf-8"))
    merge_count = len(raw["model"]["merges"])
    base_entries = len(raw["model"]["vocab"])
    added = len(raw["added_tokens"])

    metadata = {
        "artifact_version": ARTIFACT_VERSION,
        "algorithm": "ByteLevel BPE (tokenizers.models.BPE + ByteLevel pre-tokenizer/decoder)",
        "model_type": "BPE",
        "normalizer": None,
        "lowercase": False,
        "unk_token": raw["model"]["unk_token"],
        "byte_fallback": False,
        "vocab_size_requested": vocab_size,
        "vocab_size_actual": len(vocab),
        "base_vocab_entries": base_entries,
        "added_tokens": added,
        "merge_count": merge_count,
        "min_frequency": min_frequency,
        "special_tokens": special_ids,
        "library": {"tokenizers": tokenizers.__version__, "python": platform.python_version()},
        "training_parameters": {
            "vocab_size": vocab_size,
            "min_frequency": min_frequency,
            "initial_alphabet": "ByteLevel.alphabet() (256 bytes)",
            "add_prefix_space": False,
            "use_regex": True,
            "show_progress": True,
            "length": total,
            "device": "cpu",
        },
        "training_inputs": {
            "combined_train_jsonl": {
                "path": TRAIN_JSONL.relative_to(PROJECT_ROOT).as_posix(),
                "records": counters["code_records"],
                "sha256": verified["train_sha256"],
                "serialized_chars": counters["code_chars"],
            },
            "documentation": {
                "inventory": DOC_INVENTORY.relative_to(PROJECT_ROOT).as_posix(),
                "files": counters["documents"],
                "sha256_list": [d["sha256"] for d in verified["docs"]],
                "serialized_chars": counters["doc_chars"],
            },
            "total_sequences": total,
        },
        "serializer": {
            "version": SERIALIZER_VERSION,
            "path": "src/polycodegpt/tokenization/serializer.py",
            "sha256": verified["serializer_sha256"],
            "escaping": "backslash escaping of '\\\\' and of literal control-token strings (reversible)",
        },
        "input_manifest": MANIFEST_CSV.relative_to(PROJECT_ROOT).as_posix(),
        "training_duration_s": round(total_s, 2),
        "fit_duration_s": round(fit_s, 2),
        "peak_ram_mb": round(peak_rss_mb(), 1),
        "training_date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": SEED,
        "seed_note": (
            "tokenizers BpeTrainer exposes no RNG seed and uses no random sampling: "
            "it counts n-gram frequencies over the iterator in stream order and breaks "
            "frequency ties deterministically. Seed 1234 is recorded for the wider "
            "pipeline (sampling/init) but has no effect on tokenizer fitting."
        ),
        "determinism": "deterministic given identical input order and tokenizers version",
        "evaluation_data_used": False,
        "notes": "Trained only on the train partition; valid.jsonl and test.jsonl were never opened.",
    }
    tmp = METADATA_JSON.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    os.replace(tmp, METADATA_JSON)

    # unversioned copies keep downstream consumers pointed at the current artifact
    shutil.copyfile(TOKENIZER_JSON, CANONICAL_TOKENIZER)
    shutil.copyfile(METADATA_JSON, CANONICAL_METADATA)
    shutil.copyfile(MANIFEST_CSV, CANONICAL_MANIFEST)

    if verbose:
        print(f"\nsaved: {TOKENIZER_JSON.relative_to(PROJECT_ROOT)} "
              f"({TOKENIZER_JSON.stat().st_size:,} B)")
        print(f"saved: {METADATA_JSON.relative_to(PROJECT_ROOT)}")
        print(f"vocab: requested {vocab_size}, actual {len(vocab)} | merges {merge_count} | added {added}")
        print(f"special ids: {special_ids}")
        print(f"duration: {total_s:.1f} s (fit {fit_s:.1f} s), peak RSS {peak_rss_mb():.0f} MB")
    return metadata


if __name__ == "__main__":  # pragma: no cover
    train()
