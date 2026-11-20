"""Deterministic serialization of PolyCodeGPT tokenizer training text.

Version 1.1.0 — adds reversible content escaping so that literal control-token
strings inside user content can never be mistaken for structural markers, and
adds a symmetric ``parse_serialized`` decoder.

Streams
-------
Python record::

    <|lang:python|><|docstring|>{docstring}<|code|>{code}<|eos|>

JavaScript record::

    <|lang:javascript|><|docstring|>{docstring}<|code|>{code}<|eos|>

Python documentation (raw RST, unmodified)::

    <|lang:en|><|text|>{rst}<|eos|>

Escaping
--------
Content fields are escaped before insertion, with a single-pass reversible rule
on the backslash escape character ``\\``:

* every literal ``\\`` in content becomes ``\\\\``
* every literal control-token string (e.g. ``<|eos|>``) has its opening ``<|``
  rewritten to ``<\\|``, becoming ``<\\|eos|>``

The second rule **destroys the control-token literal**, so the tokenizer's
added-token matcher can never turn user content into a structural token; the
backslash sits between ``<`` and ``|``, and the string ``<|eos|>`` no longer
occurs anywhere inside the payload.

``unescape_content`` reverses both rules in one left-to-right pass
(``\\\\`` -> ``\\``, ``\\|`` -> ``|``). Because the escape character itself is
always doubled, the mapping is bijective: no two different inputs produce the
same escaped string, and escaping then unescaping is the identity.

Structural markers are inserted **after** escaping, so they are always
unambiguous, and content can never contain one — which also means the
serialized stream can be split on markers without any escape-awareness.
No field is truncated, stripped, normalized or rewritten.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable, Iterator

SERIALIZER_VERSION = "1.1.0"

EOS_TOKEN = "<|eos|>"
CODE_TOKEN = "<|code|>"
DOCSTRING_TOKEN = "<|docstring|>"
TEXT_TOKEN = "<|text|>"
ESCAPE_CHAR = "\\"

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

# Longest first so that a token is matched before any of its prefixes.
_TOKEN_PATTERN = re.compile("(" + "|".join(re.escape(t) for t in sorted(SPECIAL_TOKENS, key=len, reverse=True)) + ")")


# --------------------------------------------------------------------------- #
# escaping
# --------------------------------------------------------------------------- #
def escape_content(text: str) -> str:
    """Make ``text`` inert: escape backslashes, then break control-token literals.

    ``unescape_content(escape_content(x)) == x`` for every string, and the
    result never contains a registered control-token string.
    """
    if not text:
        return ""
    doubled = text.replace(ESCAPE_CHAR, ESCAPE_CHAR * 2)
    return _TOKEN_PATTERN.sub(lambda m: "<" + ESCAPE_CHAR + m.group(1)[1:], doubled)


def unescape_content(text: str) -> str:
    """Inverse of :func:`escape_content` (single left-to-right pass)."""
    if not text:
        return ""
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == ESCAPE_CHAR and i + 1 < n and text[i + 1] in (ESCAPE_CHAR, "|"):
            out.append(text[i + 1])
            i += 2
        else:
            out.append(ch)
            i += 1
    return "".join(out)


# --------------------------------------------------------------------------- #
# serialization
# --------------------------------------------------------------------------- #
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


def serialize_python_record(record: dict) -> str:
    """Serialize one Python CodeSearchNet record."""
    docstring = escape_content(record.get("docstring") or "")
    code = escape_content(record.get("code") or "")
    return f"{LANG_TOKEN['python']}{DOCSTRING_TOKEN}{docstring}{CODE_TOKEN}{code}{EOS_TOKEN}"


def serialize_javascript_record(record: dict) -> str:
    """Serialize one JavaScript CodeSearchNet record."""
    docstring = escape_content(record.get("docstring") or "")
    code = escape_content(record.get("code") or "")
    return f"{LANG_TOKEN['javascript']}{DOCSTRING_TOKEN}{docstring}{CODE_TOKEN}{code}{EOS_TOKEN}"


def serialize_documentation(rst_text: str) -> str:
    """Serialize one Python documentation file (raw RST, escaped only)."""
    return f"{LANG_EN}{TEXT_TOKEN}{escape_content(rst_text)}{EOS_TOKEN}"


def serialize_code_record(record: dict) -> str:
    """Serialize a record, dispatching on its verified programming language."""
    return serialize_python_record(record) if record_language(record) == "python" else serialize_javascript_record(record)


#: Backwards-compatible alias for :func:`serialize_documentation`.
serialize_document = serialize_documentation


# --------------------------------------------------------------------------- #
# deserialization / recovery
# --------------------------------------------------------------------------- #
def parse_serialized(text: str) -> dict:
    """Recover the original content from a serialized stream.

    Returns one of::

        {"kind": "code", "language": "python"|"javascript",
         "docstring": ..., "code": ...}
        {"kind": "documentation", "natural_language": "en", "text": ...}

    Raises ``ValueError`` if the marker structure is malformed, so corruption is
    never silently repaired.
    """
    for lang, marker in LANG_TOKEN.items():
        if text.startswith(marker):
            body = text[len(marker):]
            if not body.startswith(DOCSTRING_TOKEN):
                raise ValueError("code stream missing <|docstring|> marker")
            body = body[len(DOCSTRING_TOKEN):]
            found = body.count(CODE_TOKEN)
            if found != 1:
                raise ValueError(f"code stream must contain exactly one <|code|> marker, found {found}")
            docstring, code = body.split(CODE_TOKEN)
            if not code.endswith(EOS_TOKEN):
                raise ValueError("code stream missing trailing <|eos|>")
            return {
                "kind": "code",
                "language": lang,
                "docstring": unescape_content(docstring),
                "code": unescape_content(code[: -len(EOS_TOKEN)]),
            }
    if text.startswith(LANG_EN):
        body = text[len(LANG_EN):]
        if not body.startswith(TEXT_TOKEN):
            raise ValueError("documentation stream missing <|text|> marker")
        body = body[len(TEXT_TOKEN):]
        if not body.endswith(EOS_TOKEN):
            raise ValueError("documentation stream missing trailing <|eos|>")
        return {"kind": "documentation", "natural_language": "en", "text": unescape_content(body[: -len(EOS_TOKEN)])}
    raise ValueError("stream does not start with a registered language marker")


# --------------------------------------------------------------------------- #
# corpus streaming helpers
# --------------------------------------------------------------------------- #
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

    ``counters`` (optional) is updated in place with record counts and
    ``code_chars``/``doc_chars`` totals so the caller can record measured input
    statistics without a second pass.
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
        text = serialize_documentation(raw)
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
