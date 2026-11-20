"""Deterministic serialization of PolyCodeGPT tokenizer training text.

Single source of truth for how corpus records and documentation become the text
the tokenizer (and, later, the model) sees. Version this file whenever the
serialization changes; the version string is stored in every artifact manifest.

Streams
-------
CodeSearchNet record::

    <|lang:python|><|docstring|>{docstring}<|code|>{code}<|eos|>
    <|lang:javascript|><|docstring|>{docstring}<|code|>{code}<|eos|>

Python documentation file (raw RST, unmodified)::

    <|lang:en|><|text|>{rst_content}<|eos|>

Design notes
------------
* No field is truncated, rewritten, stripped or normalised.
* Docstrings are *not* labelled ``en``: their natural language has not been
  independently verified, so only the reserved ``<|lang:XX|>`` programming
  language marker is emitted for code records.
* Whitespace and indentation are preserved verbatim; the tokenizer's byte-level
  pre-tokenizer is what makes them representable.
* Streamed line by line: nothing here ever materialises the corpus in memory.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator

SERIALIZER_VERSION = "1.0.0"

EOS_TOKEN = "<|eos|>"
CODE_TOKEN = "<|code|>"
DOCSTRING_TOKEN = "<|docstring|>"
TEXT_TOKEN = "<|text|>"

LANG_TOKEN = {
    "python": "<|lang:python|>",
    "javascript": "<|lang:javascript|>",
}
LANG_EN = "<|lang:en|>"

#: The 11 special tokens, in the fixed order used for vocabulary assignment.
SPECIAL_TOKENS: tuple[str, ...] = (
    "<|pad|>",
    "<|unk|>",
    "<|eos|>",
    "<|code|>",
    "<|docstring|>",
    "<|text|>",
    "<|lang:python|>",
    "<|lang:javascript|>",
    "<|lang:en|>",
    "<|lang:fr|>",
    "<|lang:ar|>",
)


def record_language(record: dict) -> str:
    """Return ``python`` or ``javascript`` for a corpus record.

    Uses the explicit ``language`` field, cross-checked against
    ``source_dataset`` so a malformed record cannot be silently mislabelled.
    """
    explicit = (record.get("language") or "").strip().lower()
    dataset = (record.get("source_dataset") or "").lower()
    from_dataset = "python" if "python" in dataset else "javascript" if "javascript" in dataset else ""
    if explicit and from_dataset and explicit != from_dataset:
        raise ValueError(
            f"language/source_dataset mismatch: {explicit!r} vs {dataset!r} "
            f"for {record.get('repo')}:{record.get('path')}"
        )
    lang = explicit or from_dataset
    if lang not in LANG_TOKEN:
        raise ValueError(f"unsupported language {lang!r} in record {record.get('repo')}:{record.get('path')}")
    return lang


def serialize_code_record(record: dict) -> str:
    """Serialize one CodeSearchNet record into tokenizer training text."""
    lang = record_language(record)
    docstring = record.get("docstring") or ""
    code = record.get("code") or ""
    return f"{LANG_TOKEN[lang]}{DOCSTRING_TOKEN}{docstring}{CODE_TOKEN}{code}{EOS_TOKEN}"


def serialize_document(text: str) -> str:
    """Serialize one Python documentation file (raw RST) into training text."""
    return f"{LANG_EN}{TEXT_TOKEN}{text}{EOS_TOKEN}"


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    """Yield parsed records from a JSONL file one line at a time."""
    path = Path(path)
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:  # fail loudly, never skip silently
                raise ValueError(f"{path}:{lineno}: invalid JSON ({exc})") from exc


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def load_doc_inventory(path: str | Path, project_root: str | Path) -> list[dict]:
    """Read the eligible-documentation inventory, resolving local paths.

    Only rows marked ``partition_assignment == train_only`` are accepted; the
    inventory is the sole authority on which documentation files are eligible.
    """
    root = Path(project_root)
    out: list[dict] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["partition_assignment"] != "train_only":
                continue
            local = Path(row["local_file_path"])
            if not local.is_absolute():
                local = root / local
            out.append(
                {
                    "doc_id": row["doc_id"],
                    "original_path": row["original_repository_url"],
                    "local_path": str(local),
                    "rel_path": row["local_file_path"].replace("\\", "/"),
                    "sha256": row["file_sha256"],
                    "source_license": row["source_license"],
                }
            )
    out.sort(key=lambda r: r["rel_path"])
    return out


def iter_training_texts(
    train_jsonl: str | Path,
    docs: Iterable[dict],
    counters: dict | None = None,
) -> Iterator[str]:
    """Yield serialized training texts: code records first, then documents.

    ``counters`` (optional) is updated in place with ``code_records``,
    ``documents`` and ``code_chars``/``doc_chars`` totals so the caller can
    record measured input statistics without a second pass.
    """
    n_code = 0
    n_docs = 0
    code_chars = 0
    doc_chars = 0
    for record in iter_jsonl(train_jsonl):
        text = serialize_code_record(record)
        n_code += 1
        code_chars += len(text)
        yield text
    for doc in docs:
        with open(doc["local_path"], encoding="utf-8") as fh:
            raw = fh.read()
        text = serialize_document(raw)
        n_docs += 1
        doc_chars += len(text)
        yield text
    if counters is not None:
        counters.update(
            code_records=n_code,
            documents=n_docs,
            code_chars=code_chars,
            doc_chars=doc_chars,
        )
