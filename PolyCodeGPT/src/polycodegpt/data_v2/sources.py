"""Licensed source adapters for the v2 corpus.

Only these inputs are opened by the builder:

* existing CodeSearchNet train/valid partitions (never test)
* the approved 500-file CPython documentation inventory
* the acquired Node.js v24.12.0 tagged documentation
* acquired CodeParrot Parquet shards, restricted later by per-file license

Source adapters are deliberately thin: they preserve raw provenance and do not
drop records for quality reasons. The pipeline records every rejection with a
stable reason code.
"""

from __future__ import annotations

import ast
import csv
import json
import re
from pathlib import Path
from typing import Iterator

from .constants import DOC_MAX_CHARS, DOC_MIN_CHARS, DOC_CHUNK_TARGET
from .schema import assert_no_forbidden_input
from .language import (
    classify_code,
    imports_for,
    is_test_path,
    normalize_repo,
    signature_for,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CSN_TRAIN = PROJECT_ROOT / "data" / "processed" / "polycodegpt_corpus" / "train.jsonl"
CSN_VALID = PROJECT_ROOT / "data" / "processed" / "polycodegpt_corpus" / "valid.jsonl"
DOC_INVENTORY = PROJECT_ROOT / "data" / "manifests" / "python_docs_training_corpus_inventory.csv"
NODE_ROOT = PROJECT_ROOT / "data" / "raw" / "v2" / "node_docs" / "node-24.12.0"
CODEPARROT_ROOT = PROJECT_ROOT / "data" / "raw" / "v2" / "codeparrot_github_code"

CSN_SNAPSHOT = "CodeSearchNet Zenodo v1 (record 7908468)"
CSN_LICENSE = "CLEAR_PERMISSIVE_CANDIDATE"
CSN_EVIDENCE = "Phase 2B.2 repository-level worst-case license audit; only CLEAR_PERMISSIVE_CANDIDATE retained"
PYDOCS_SNAPSHOT = "CPython v3.11.9 Doc/"
PYDOCS_LICENSE = "PSF-2.0 (prose); code samples dual PSF-2.0/0BSD"
PYDOCS_EVIDENCE = "data/manifests/python_docs_training_corpus_inventory.csv per-file license, SHA256, eligibility"
NODE_SNAPSHOT = "nodejs/node v24.12.0"
NODE_LICENSE = "MIT"
NODE_EVIDENCE = "nodejs/node v24.12.0 LICENSE (MIT-style permission); docs are part of the tagged repository"
PARROT_SNAPSHOT = "codeparrot/github-code@b5661e6b17396364b2bcf8e68977b0d28e1ebd19"
PARROT_EVIDENCE = (
    "CodeParrot per-example `license` field; only MIT, Apache-2.0, BSD-2-Clause, "
    "BSD-3-Clause, ISC, CC0-1.0, Unlicense accepted; dataset-level 'other' is not permission"
)


def _jsonl(path: Path) -> Iterator[dict]:
    assert_no_forbidden_input(str(path))
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def iter_csn() -> Iterator[dict]:
    for path, origin in ((CSN_TRAIN, "train"), (CSN_VALID, "valid")):
        assert_no_forbidden_input(str(path))
        for index, rec in enumerate(_jsonl(path)):
            source_id = "CSN-PY" if rec.get("language") == "python" else "CSN-JS"
            language = rec.get("language")
            if language not in {"python", "javascript"}:
                continue
            repo = normalize_repo(rec.get("repo") or "")
            code = rec.get("code") or ""
            docstring = (rec.get("docstring") or "").strip()
            yield {
                "content_kind": "code",
                "source_id": source_id,
                "source_snapshot": CSN_SNAPSHOT,
                "repo": repo,
                "path": rec.get("path") or "",
                "license_id": CSN_LICENSE,
                "license_evidence": CSN_EVIDENCE,
                "source_url": (rec.get("provenance") or {}).get("source_url") or rec.get("url") or "",
                "origin_partition": origin,
                "language": language,
                "raw_content": code,
                "docstring": docstring,
                "signature": signature_for(language, code),
                "imports": imports_for(language, code),
                "category": classify_code(language, code, rec.get("path") or ""),
                "alignment": "",
                "source_index": index,
            }


def _split_markdown(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").strip()
    # Drop YAML front matter but preserve the body.
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            text = text[end + 4 :].lstrip()
    sections = re.split(r"(?m)(?=^##\s+)", text)
    chunks: list[str] = []
    for section in sections:
        if len(section) <= DOC_MAX_CHARS:
            if section.strip():
                chunks.append(section.strip())
            continue
        paragraphs = re.split(r"(?m)(?=^#{1,3}\s+)", section)
        current = ""
        for paragraph in paragraphs:
            if len(current) + len(paragraph) <= DOC_CHUNK_TARGET:
                current += paragraph
            else:
                if current.strip():
                    chunks.append(current.strip())
                current = paragraph
        if current.strip():
            chunks.append(current.strip())
    return [c for c in chunks if DOC_MIN_CHARS <= len(c) <= DOC_MAX_CHARS]


def _split_rst(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").strip()
    # RST sections begin with an underlined heading. Keep code-block directives
    # and surrounding explanation together.
    sections = re.split(r"(?m)(?=^[A-Za-z0-9][^\n]*\n[-=~^\"']{3,}\s*$)", text)
    chunks: list[str] = []
    for section in sections:
        if len(section) <= DOC_MAX_CHARS:
            if section.strip():
                chunks.append(section.strip())
            continue
        blocks = re.split(r"(?m)(?=^\.\. code-block::)", section)
        current = ""
        for block in blocks:
            if len(current) + len(block) <= DOC_CHUNK_TARGET:
                current += block
            else:
                if current.strip():
                    chunks.append(current.strip())
                current = block
        if current.strip():
            chunks.append(current.strip())
    return [c for c in chunks if DOC_MIN_CHARS <= len(c) <= DOC_MAX_CHARS]


def _has_code_fence(text: str) -> bool:
    return "```" in text or "~~~" in text


def iter_python_docs() -> Iterator[dict]:
    with open(DOC_INVENTORY, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        if not (row.get("eligible_for_tokenizer_training") or "").upper().startswith("YES"):
            continue
        path = PROJECT_ROOT / row["local_file_path"]
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for idx, chunk in enumerate(_split_rst(text)):
            yield {
                "content_kind": "documentation",
                "source_id": "PYDOCS",
                "source_snapshot": PYDOCS_SNAPSHOT,
                "repo": "python/cpython",
                "path": f"{row['local_file_path']}::chunk{idx:03d}",
                "license_id": PYDOCS_LICENSE,
                "license_evidence": PYDOCS_EVIDENCE,
                "source_url": row.get("original_repository_url") or "",
                "origin_partition": "train_only_documentation",
                "language": "en",
                "raw_content": chunk,
                "docstring": "",
                "signature": "",
                "imports": "",
                "category": "documentation",
                "alignment": "",
                "source_index": idx,
            }


def iter_node_docs() -> Iterator[dict]:
    doc_root = NODE_ROOT / "doc"
    for path in sorted(doc_root.rglob("*.md")):
        rel = path.relative_to(NODE_ROOT).as_posix()
        if "/changelogs/" in f"/{rel}" or rel.endswith("CHANGELOG.md"):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for idx, chunk in enumerate(_split_markdown(text)):
            if not _has_code_fence(chunk):
                continue
            yield {
                "content_kind": "documentation",
                "source_id": "NODE-DOCS",
                "source_snapshot": NODE_SNAPSHOT,
                "repo": "nodejs/node",
                "path": f"{rel}::chunk{idx:03d}",
                "license_id": NODE_LICENSE,
                "license_evidence": NODE_EVIDENCE,
                "source_url": f"https://github.com/nodejs/node/blob/v24.12.0/{rel}",
                "origin_partition": "train_only_documentation",
                "language": "en",
                "raw_content": chunk,
                "docstring": "",
                "signature": "",
                "imports": "",
                "category": "documentation",
                "alignment": "",
                "source_index": idx,
            }


def iter_codeparrot(
    shard_limit: int | None = None, shard_offset: int = 0
) -> Iterator[dict]:
    import pyarrow.parquet as pq

    shards = sorted(CODEPARROT_ROOT.glob("*.parquet"))
    if shard_offset:
        shards = shards[shard_offset:]
    if shard_limit is not None:
        shards = shards[:shard_limit]
    for shard in shards:
        pf = pq.ParquetFile(shard)
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg, columns=["repo_name", "path", "size", "license", "content"])
            for rec in table.to_pylist():
                path = rec.get("path") or ""
                suffix = Path(path).suffix.lower()
                if suffix not in {".py", ".js", ".md"}:
                    continue
                language = "python" if suffix == ".py" else "javascript" if suffix == ".js" else "en"
                content_kind = "documentation" if suffix == ".md" else "code"
                repo = normalize_repo(rec.get("repo_name") or "")
                license_raw = (rec.get("license") or "").strip().lower()
                raw = rec.get("content") or ""
                chunks = _split_markdown(raw) if suffix == ".md" else [raw]
                for idx, chunk in enumerate(chunks):
                    yield {
                        "content_kind": content_kind,
                        "source_id": "CODEPARROT",
                        "source_snapshot": PARROT_SNAPSHOT,
                        "repo": repo,
                        "path": f"{path}::chunk{idx:03d}" if suffix == ".md" else path,
                        "license_id": license_raw or "UNKNOWN",
                        "license_evidence": PARROT_EVIDENCE,
                        "source_url": f"https://github.com/{repo}/blob/HEAD/{path}" if repo else "",
                        "origin_partition": "train",
                        "language": language,
                        "raw_content": chunk,
                        "docstring": "",
                        "signature": "",
                        "imports": "",
                        "category": "documentation" if content_kind == "documentation" else classify_code(language, raw, path),
                        "alignment": "",
                        "source_index": idx,
                        "shard": shard.name,
                    }


def iter_all_sources(
    shard_limit: int | None = None, shard_offset: int = 0
) -> Iterator[dict]:
    yield from iter_csn()
    yield from iter_codeparrot(shard_limit=shard_limit, shard_offset=shard_offset)
    yield from iter_node_docs()
    yield from iter_python_docs()
