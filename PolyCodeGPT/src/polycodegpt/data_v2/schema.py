"""Schema 2.0 serialization with reversible content escaping.

Task-first envelope::

    <|bos|><|lang:python|><|task:function|><|file|>...<|imports|>...
    <|docstring|>...<|signature|>...<|code|>...<|eos|>

Documentation::

    <|bos|><|lang:en|><|text|>...<|eos|>

Every content field is escaped before structural markers are inserted, so a
literal control-token string in source code cannot become structure. The
escaping is reversible.
"""

from __future__ import annotations

import hashlib
import re
from typing import Iterable, Iterator

from .constants import (
    LANG_TOKENS,
    SCHEMA_VERSION,
    SECTION_TOKENS,
    SERIALIZED_SECTIONS,
    SPECIAL_TOKENS,
    TASK_TOKENS,
)

BOS = "<|bos|>"
EOS = "<|eos|>"
ESCAPE = "\\"
_TOKEN_PATTERN = re.compile(
    "(" + "|".join(re.escape(t) for t in sorted(SPECIAL_TOKENS, key=len, reverse=True)) + ")"
)


def escape_content(text: str) -> str:
    if not text:
        return ""
    doubled = text.replace(ESCAPE, ESCAPE * 2)
    return _TOKEN_PATTERN.sub(lambda m: "<" + ESCAPE + m.group(1)[1:], doubled)


def unescape_content(text: str) -> str:
    if not text:
        return ""
    out: list[str] = []
    i = 0
    while i < len(text):
        if text[i] == ESCAPE and i + 1 < len(text) and text[i + 1] in (ESCAPE, "|"):
            out.append(text[i + 1])
            i += 2
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def serialize_code(
    language: str,
    task: str,
    *,
    file: str = "",
    context: str = "",
    imports: str = "",
    docstring: str = "",
    signature: str = "",
    prefix: str = "",
    suffix: str = "",
    buggy: str = "",
    tests: str = "",
    code: str = "",
    explanation: str = "",
) -> str:
    if language not in LANG_TOKENS:
        raise ValueError(f"unsupported language: {language}")
    if task not in TASK_TOKENS:
        raise ValueError(f"unsupported task: {task}")
    values = {
        "file": file,
        "context": context,
        "imports": imports,
        "docstring": docstring,
        "signature": signature,
        "prefix": prefix,
        "suffix": suffix,
        "buggy": buggy,
        "tests": tests,
        "code": code,
        "explanation": explanation,
    }
    parts = [BOS, LANG_TOKENS[language], TASK_TOKENS[task]]
    for section in SERIALIZED_SECTIONS:
        value = values.get(section, "")
        if value:
            parts.append(SECTION_TOKENS[section])
            parts.append(escape_content(value))
    parts.append(EOS)
    return "".join(parts)


def serialize_documentation(text: str) -> str:
    return f"{BOS}{LANG_TOKENS['en']}{SECTION_TOKENS['text']}{escape_content(text)}{EOS}"


def parse_serialized(text: str) -> dict:
    """Recover structure and content; raises ValueError on malformed streams."""
    if not text.startswith(BOS):
        raise ValueError("missing <|bos|>")
    body = text[len(BOS) :]
    if body.endswith(EOS):
        body = body[: -len(EOS)]
    else:
        raise ValueError("missing trailing <|eos|>")
    lang = None
    for key, token in LANG_TOKENS.items():
        if body.startswith(token):
            lang = key
            body = body[len(token) :]
            break
    if lang is None:
        raise ValueError("missing language token")
    if lang == "en":
        if not body.startswith(SECTION_TOKENS["text"]):
            raise ValueError("documentation must start with <|text|>")
        return {"kind": "documentation", "language": lang, "sections": {"text": unescape_content(body[len(SECTION_TOKENS["text"]) :])}}
    task = None
    for key, token in TASK_TOKENS.items():
        if body.startswith(token):
            task = key
            body = body[len(token) :]
            break
    if task is None:
        raise ValueError("missing task token")
    # Split only on structural markers; escaped content cannot match them.
    markers = sorted(SECTION_TOKENS.values(), key=len, reverse=True)
    pattern = re.compile("(" + "|".join(re.escape(m) for m in markers) + ")")
    pieces = pattern.split(body)
    sections: dict[str, str] = {}
    current = None
    for piece in pieces:
        if piece in SECTION_TOKENS.values():
            current = next(k for k, v in SECTION_TOKENS.items() if v == piece)
            sections[current] = ""
        elif current is not None:
            sections[current] += piece
    return {
        "kind": "code",
        "language": lang,
        "task": task,
        "sections": {k: unescape_content(v) for k, v in sections.items()},
    }


def make_record_id(source_id: str, repo: str, path: str, code_sha256: str) -> str:
    raw = f"{source_id}|{repo}|{path}|{code_sha256}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:32]


def build_code_record(
    *,
    source_id: str,
    source_snapshot: str,
    repo: str,
    path: str,
    license_id: str,
    license_evidence: str,
    source_url: str,
    origin_partition: str,
    language: str,
    task: str,
    category: str,
    code: str,
    imports: str = "",
    docstring: str = "",
    signature: str = "",
    alignment: str = "not_applicable",
    exact_sha256: str = "",
    normalized_sha256: str = "",
) -> dict:
    text = serialize_code(
        language,
        task,
        file=f"{repo}/{path}",
        imports=imports,
        docstring=docstring,
        signature=signature,
        code=code,
    )
    code_sha = hashlib.sha256(code.encode("utf-8")).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        "record_id": make_record_id(source_id, repo, path, code_sha),
        "content_kind": "code",
        "source_id": source_id,
        "source_snapshot": source_snapshot,
        "repo": repo,
        "path": path,
        "license_id": license_id,
        "license_evidence": license_evidence,
        "source_url": source_url,
        "origin_partition": origin_partition,
        "language": language,
        "task": task,
        "category": category,
        "imports_present": bool(imports),
        "docstring_present": bool(docstring),
        "signature_present": bool(signature),
        "alignment": alignment,
        "code_sha256": code_sha,
        "exact_sha256": exact_sha256,
        "normalized_sha256": normalized_sha256,
        "text": text,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def build_documentation_record(
    *,
    source_id: str,
    source_snapshot: str,
    repo: str,
    path: str,
    license_id: str,
    license_evidence: str,
    source_url: str,
    text: str,
    exact_sha256: str = "",
    normalized_sha256: str = "",
) -> dict:
    serialized = serialize_documentation(text)
    return {
        "schema_version": SCHEMA_VERSION,
        "record_id": make_record_id(source_id, repo, path, hashlib.sha256(text.encode("utf-8")).hexdigest()),
        "content_kind": "documentation",
        "source_id": source_id,
        "source_snapshot": source_snapshot,
        "repo": repo,
        "path": path,
        "license_id": license_id,
        "license_evidence": license_evidence,
        "source_url": source_url,
        "origin_partition": "train_only_documentation",
        "language": "en",
        "task": "text",
        "category": "documentation",
        "imports_present": False,
        "docstring_present": False,
        "signature_present": False,
        "alignment": "not_applicable",
        "code_sha256": "",
        "exact_sha256": exact_sha256,
        "normalized_sha256": normalized_sha256,
        "text": serialized,
        "text_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
    }


def iter_jsonl(path: str) -> Iterator[dict]:
    import json
    from pathlib import Path

    with open(Path(path), encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except Exception as exc:
                    raise ValueError(f"{path}:{lineno}: invalid JSON") from exc


def validate_record(record: dict) -> None:
    required = {
        "schema_version",
        "record_id",
        "content_kind",
        "source_id",
        "repo",
        "path",
        "license_id",
        "language",
        "text",
        "text_sha256",
    }
    missing = required - set(record)
    if missing:
        raise ValueError(f"record missing fields: {sorted(missing)}")
    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"wrong schema {record['schema_version']}")
    if hashlib.sha256(record["text"].encode("utf-8")).hexdigest() != record["text_sha256"]:
        raise ValueError("text hash mismatch")
    parsed = parse_serialized(record["text"])
    if parsed["language"] != record["language"]:
        raise ValueError("language mismatch")
    if record["content_kind"] == "code":
        if parsed["task"] != record["task"]:
            raise ValueError("task mismatch")
    elif record["content_kind"] != "documentation":
        raise ValueError("unknown content kind")


def assert_no_forbidden_input(path: str) -> None:
    p = str(path).replace("\\", "/").lower()
    if p.endswith("/test.jsonl") or "/functional/independent_v3/" in p:
        raise RuntimeError(f"refusing forbidden evaluation input: {path}")


def iter_texts(records: Iterable[dict]) -> Iterator[str]:
    for rec in records:
        yield rec["text"]
