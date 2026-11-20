#!/usr/bin/env python
"""Phase 18 source and licensing preflight.

This script does not build or modify the training corpus. It records the
preservation baseline for the frozen v0.1.0 artifacts and the source/license
decisions that the Phase 18 builder is allowed to consume.

The decisions are deliberately conservative: a source is consumed only when
there is an explicit permissive license (or a per-file license field restricted
to an explicit allowlist). Public downloadability alone is never accepted as
evidence of permission.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests" / "v2"
RAW_CODEPARROT = PROJECT_ROOT / "data" / "raw" / "v2" / "codeparrot_github_code"
NODE_ROOT = PROJECT_ROOT / "data" / "raw" / "v2" / "node_docs" / "node-24.12.0"
NODE_TAR = PROJECT_ROOT / "data" / "raw" / "v2" / "node_docs" / "node-v24.12.0.tar.gz"
CODEPARROT_REVISION = "b5661e6b17396364b2bcf8e68977b0d28e1ebd19"
CODEPARROT_CARD = "https://huggingface.co/datasets/codeparrot/github-code"
CODEPARROT_SNAPSHOT_DATE = "2022-03-16"
ACQUISITION_DATE = "2026-11-18"


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def parquet_rows(path: Path) -> int:
    import pyarrow.parquet as pq

    return pq.ParquetFile(path).metadata.num_rows


PRESERVATION_FILES = [
    "release/polycodegpt-exp-c-0.1.0.zip",
    "release/polycodegpt-exp-c-0.1.0/models/exp_c_step_016260.pt",
    "release/polycodegpt-exp-c-0.1.0/tokenizer/polycodegpt_tokenizer_v1_1.json",
    "release/polycodegpt-exp-c-0.1.0/hashes.sha256",
    "checkpoints/polycodegpt_first_pass/best.pt",
    "checkpoints/polycodegpt_first_pass/latest.pt",
    "tokenizer/artifacts/polycodegpt_tokenizer_v1_1.json",
    "tokenizer/artifacts/tokenizer_metadata_v1_1.json",
    "data/processed/polycodegpt_corpus/train.jsonl",
    "data/processed/polycodegpt_corpus/valid.jsonl",
    "data/processed/tokenized_v1_1/train.bin",
    "data/processed/tokenized_v1_1/valid.bin",
    "data/manifests/polycodegpt_corpus_manifest.csv",
    "data/manifests/polycodegpt_corpus_validation.json",
    "data/manifests/source_registry.csv",
    "data/manifests/python_docs_training_corpus_inventory.csv",
    "reports/phase17_data_strategy.md",
]


def write_preservation() -> list[dict]:
    rows = []
    for name in PRESERVATION_FILES:
        p = PROJECT_ROOT / name
        if not p.is_file():
            raise FileNotFoundError(f"preservation baseline missing: {name}")
        rows.append(
            {
                "path": name,
                "bytes": p.stat().st_size,
                "sha256": sha256_file(p),
                "role": "frozen_v0_1_0_artifact" if name.startswith(("release/", "checkpoints/")) else "historical_input_or_report",
                "modified_by_phase18": "NO",
            }
        )
    out = MANIFEST_DIR / "preservation_manifest.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_sources() -> list[dict]:
    # These are corpus-consumption decisions, not legal opinions. The evidence
    # column points to the exact basis available in the repository or acquired
    # source snapshot.
    rows = [
        {
            "source_id": "CSN-PY",
            "source_name": "CodeSearchNet Python filtered train/valid",
            "version_or_snapshot": "Zenodo v1 record 7908468; local Phase 2B.2 filter",
            "acquisition_date": "2026-11-15",
            "repository_identity": "multi-repository; repository recorded per record",
            "license_id": "CLEAR_PERMISSIVE_CANDIDATE (per-repository worst case)",
            "license_evidence": "Phase 2B.2 repository license audit; only CLEAR_PERMISSIVE_CANDIDATE retained; per-record provenance and source DOI preserved",
            "permits_intended_use": "YES_WITH_ATTRIBUTION",
            "language_content": "Python code; function/method snippets; docstrings",
            "decision": "ACCEPT",
            "reason": "Existing seed corpus with recorded permissive per-repository classification. Train and valid only; test is frozen and untouched.",
            "stage_record_count": "392632 train + 20965 valid (existing v1; re-gated and re-split in v2)",
            "stage_token_count": "not used as final; v2 re-tokenization below",
        },
        {
            "source_id": "CSN-JS",
            "source_name": "CodeSearchNet JavaScript filtered train/valid",
            "version_or_snapshot": "Zenodo v1 record 7908468; local Phase 2B.2 filter",
            "acquisition_date": "2026-11-15",
            "repository_identity": "multi-repository; repository recorded per record",
            "license_id": "CLEAR_PERMISSIVE_CANDIDATE (per-repository worst case)",
            "license_evidence": "Phase 2B.2 repository license audit; only CLEAR_PERMISSIVE_CANDIDATE retained; per-record provenance and source DOI preserved",
            "permits_intended_use": "YES_WITH_ATTRIBUTION",
            "language_content": "JavaScript code; function/method snippets; docstrings",
            "decision": "ACCEPT",
            "reason": "Existing seed corpus. Phase 12 curated only a small subset; v2 re-runs syntax and structural gates and does not use the old restriction as a ceiling.",
            "stage_record_count": "102169 train + 6294 valid (existing v1; re-gated and re-split in v2)",
            "stage_token_count": "not used as final; v2 re-tokenization below",
        },
        {
            "source_id": "PYDOCS",
            "source_name": "CPython 3.11.9 English documentation",
            "version_or_snapshot": "v3.11.9; 500 eligibility-audited files",
            "acquisition_date": "2026-11-14",
            "repository_identity": "python/cpython (Doc/)",
            "license_id": "PSF-2.0 (prose); code samples dual PSF-2.0/0BSD",
            "license_evidence": "data/manifests/python_docs_training_corpus_inventory.csv records per-file license, SHA256, and eligibility; Doc/license.rst excluded",
            "permits_intended_use": "YES_WITH_ATTRIBUTION",
            "language_content": "English Python documentation with code examples",
            "decision": "ACCEPT_TRAIN_ONLY",
            "reason": "Existing approved 500-file documentation inventory; train-only and never used for validation or test.",
            "stage_record_count": "500 files",
            "stage_token_count": "~2.85M v1 tokens",
        },
        {
            "source_id": "NODE-DOCS",
            "source_name": "Node.js v24.12.0 documentation",
            "version_or_snapshot": "git tag v24.12.0 (tarball acquired 2026-11-18)",
            "acquisition_date": ACQUISITION_DATE,
            "repository_identity": "nodejs/node (doc/)",
            "license_id": "MIT",
            "license_evidence": "Acquired repository LICENSE begins 'Node.js is licensed ... MIT-style permission ...'; doc/ is part of the tagged repository. LICENSE SHA256 recorded in source_preflight.json.",
            "permits_intended_use": "YES_WITH_ATTRIBUTION",
            "language_content": "English JavaScript/Node.js documentation with code examples",
            "decision": "ACCEPT_TRAIN_ONLY",
            "reason": "Tagged official docs; permissive MIT license; documentation only, train-only.",
            "stage_record_count": "149 Markdown files in doc/ (pre-chunk)",
            "stage_token_count": "measured after chunking",
        },
        {
            "source_id": "CODEPARROT",
            "source_name": "CodeParrot github-code (public snapshot)",
            "version_or_snapshot": f"HF revision {CODEPARROT_REVISION}; dataset card snapshot {CODEPARROT_SNAPSHOT_DATE}",
            "acquisition_date": ACQUISITION_DATE,
            "repository_identity": "multi-repository; repo_name recorded per file",
            "license_id": "per-example GitHub license field; only explicit allowlist",
            "license_evidence": "Dataset card and Parquet field `license`; allowlist MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, ISC, CC0-1.0, Unlicense. Missing/unknown/other/copyleft values rejected. CodeParrot's dataset-level 'other' tag is NOT treated as permission.",
            "permits_intended_use": "YES_ONLY_FOR_ALLOWLISTED_FILES_WITH_ATTRIBUTION",
            "language_content": "Python and JavaScript source files; Markdown documentation used only when code-fenced and allowlisted",
            "decision": "ACCEPT_WITH_PER_FILE_LICENSE_GATE",
            "reason": "Public snapshot with per-file repository license metadata. Only the explicit SPDX allowlist is eligible; every accepted file retains repo/path/license provenance.",
            "stage_record_count": "measured per shard",
            "stage_token_count": "measured after filtering",
        },
        {
            "source_id": "THE-STACK-V1-V2",
            "source_name": "The Stack v1/v2",
            "version_or_snapshot": "gated",
            "acquisition_date": "not acquired",
            "repository_identity": "multi-repository",
            "license_id": "requires access agreement",
            "license_evidence": "Hugging Face API reports gated access; no credentials or agreement in project",
            "permits_intended_use": "UNRESOLVED",
            "language_content": "candidate code",
            "decision": "EXCLUDE",
            "reason": "Access/license agreement not available; cannot establish permission.",
            "stage_record_count": "0",
            "stage_token_count": "0",
        },
        {
            "source_id": "STARCODERDATA",
            "source_name": "StarCoderData",
            "version_or_snapshot": "gated",
            "acquisition_date": "not acquired",
            "repository_identity": "multi-repository",
            "license_id": "requires access agreement",
            "license_evidence": "gated dataset with upstream terms; no project access",
            "permits_intended_use": "UNRESOLVED",
            "language_content": "candidate code",
            "decision": "EXCLUDE",
            "reason": "Access/license agreement not available.",
            "stage_record_count": "0",
            "stage_token_count": "0",
        },
        {
            "source_id": "MDN-PROSE",
            "source_name": "MDN content prose",
            "version_or_snapshot": "not acquired",
            "acquisition_date": "not acquired",
            "repository_identity": "mdn/content",
            "license_id": "CC-BY-SA for prose; code-sample date ambiguity",
            "license_evidence": "Phase 17 notes post-2010 CC0 code samples but historical file dates are not reliably recoverable per snippet",
            "permits_intended_use": "UNRESOLVED",
            "language_content": "documentation",
            "decision": "EXCLUDE",
            "reason": "Do not infer permission for mixed-date prose/code. Node and CPython documentation provide licensed coverage instead.",
            "stage_record_count": "0",
            "stage_token_count": "0",
        },
        {
            "source_id": "STACKOVERFLOW",
            "source_name": "Stack Overflow / Stack Exchange code",
            "version_or_snapshot": "not acquired",
            "acquisition_date": "not acquired",
            "repository_identity": "Stack Exchange",
            "license_id": "CC-BY-SA",
            "license_evidence": "Share-alike attribution terms conflict with intended unrestricted model-data use",
            "permits_intended_use": "NO",
            "language_content": "Q&A/code",
            "decision": "EXCLUDE",
            "reason": "Incompatible share-alike restrictions for this corpus release.",
            "stage_record_count": "0",
            "stage_token_count": "0",
        },
        {
            "source_id": "APPS-MBPP-CODE_CONTESTS",
            "source_name": "APPS / MBPP / CodeContests",
            "version_or_snapshot": "not acquired",
            "acquisition_date": "not acquired",
            "repository_identity": "benchmark datasets",
            "license_id": "mixed/non-permissive or benchmark-only",
            "license_evidence": "Phase 17 source review; problem text and solutions have restrictive or unclear terms",
            "permits_intended_use": "NO",
            "language_content": "benchmark tasks",
            "decision": "EXCLUDE",
            "reason": "Not licensed as bulk training data; benchmark contamination risk.",
            "stage_record_count": "0",
            "stage_token_count": "0",
        },
        {
            "source_id": "BUGSJS-BUGSINPY",
            "source_name": "BugsJS / BugsInPy",
            "version_or_snapshot": "not acquired",
            "acquisition_date": "not acquired",
            "repository_identity": "benchmark repositories",
            "license_id": "mixed repository licenses; benchmark labels",
            "license_evidence": "Phase 17 source review; benchmark/test identity and mixed terms",
            "permits_intended_use": "NO_FOR_BULK_TRAINING",
            "language_content": "bug-fix benchmark",
            "decision": "EXCLUDE",
            "reason": "Consumed historical benchmark material; using it for training would contaminate evaluation.",
            "stage_record_count": "0",
            "stage_token_count": "0",
        },
    ]
    out = MANIFEST_DIR / "source_registry_v2.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_preflight(preservation: list[dict], sources: list[dict]) -> dict:
    shards = []
    for p in sorted(RAW_CODEPARROT.glob("*.parquet")):
        shards.append(
            {
                "path": rel(p),
                "bytes": p.stat().st_size,
                "sha256": sha256_file(p),
                "rows": parquet_rows(p),
                "source": "CODEPARROT",
                "revision": CODEPARROT_REVISION,
            }
        )
    if not shards:
        raise RuntimeError("no CodeParrot shards acquired")
    node_license = NODE_ROOT / "LICENSE"
    if not node_license.is_file():
        raise FileNotFoundError("Node.js LICENSE missing")
    node_docs = sorted((NODE_ROOT / "doc").rglob("*.md"))
    docs_inventory = PROJECT_ROOT / "data" / "manifests" / "python_docs_training_corpus_inventory.csv"
    with open(docs_inventory, newline="", encoding="utf-8") as fh:
        doc_rows = list(csv.DictReader(fh))
    eligible_docs = [r for r in doc_rows if r.get("eligible_for_tokenizer_training", "").upper().startswith("YES")]
    preflight = {
        "phase": 18,
        "preflight_date": ACQUISITION_DATE,
        "preservation": {
            "frozen_release": "release/polycodegpt-exp-c-0.1.0",
            "checkpoint_files_preserved": True,
            "existing_test_partition_read": False,
            "independent_v3_task_content_inspected": False,
            "policy": "No Phase 18 command reads data/processed/polycodegpt_corpus/test.jsonl or functional/independent_v3/ cases. Only train.jsonl and valid.jsonl seed records are read.",
        },
        "sources_examined": len(sources),
        "sources_accepted": [r["source_id"] for r in sources if r["decision"].startswith(("ACCEPT", "ACCEPT_TRAIN_ONLY", "ACCEPT_WITH"))],
        "sources_excluded": [r["source_id"] for r in sources if r["decision"] == "EXCLUDE"],
        "license_allowlist": ["MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC", "CC0-1.0", "Unlicense"],
        "license_allowlist_lower": ["mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc", "cc0-1.0", "unlicense"],
        "codeparrot": {
            "url": CODEPARROT_CARD,
            "revision": CODEPARROT_REVISION,
            "snapshot_date": CODEPARROT_SNAPSHOT_DATE,
            "shards_acquired": len(shards),
            "shard_bytes": sum(s["bytes"] for s in shards),
            "shard_rows": sum(s["rows"] for s in shards),
            "shards": shards,
        },
        "node_docs": {
            "version": "v24.12.0",
            "tarball": rel(NODE_TAR),
            "tarball_bytes": NODE_TAR.stat().st_size,
            "tarball_sha256": sha256_file(NODE_TAR),
            "license_path": rel(node_license),
            "license_sha256": sha256_file(node_license),
            "markdown_files": len(node_docs),
            "markdown_bytes": sum(p.stat().st_size for p in node_docs),
        },
        "python_docs": {
            "inventory": rel(docs_inventory),
            "inventory_sha256": sha256_file(docs_inventory),
            "eligible_files": len(eligible_docs),
            "inventory_rows": len(doc_rows),
        },
        "preservation_manifest": preservation,
    }
    out = MANIFEST_DIR / "source_preflight.json"
    out.write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return preflight


def main() -> int:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    preservation = write_preservation()
    sources = write_sources()
    preflight = write_preflight(preservation, sources)
    print(f"wrote {len(preservation)} preservation hashes")
    print(f"wrote {len(sources)} source decisions; accepted={preflight['sources_accepted']}")
    print(f"CodeParrot shards={preflight['codeparrot']['shards_acquired']} rows={preflight['codeparrot']['shard_rows']:,} bytes={preflight['codeparrot']['shard_bytes']:,}")
    print(f"Node docs={preflight['node_docs']['markdown_files']} files {preflight['node_docs']['markdown_bytes']:,} B")
    print(f"manifest dir: {rel(MANIFEST_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
