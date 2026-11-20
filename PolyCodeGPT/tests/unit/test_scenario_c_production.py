"""Unit and integration tests for production Scenario C extraction pipeline."""

import ast
import json
import sqlite3
import tempfile
from pathlib import Path
from collections import Counter

import pytest
from tokenizers import Tokenizer
from tree_sitter import Language, Parser
import tree_sitter_javascript as tsjs

from polycodegpt.data_v2.constants import LICENSE_ALLOWLIST
from polycodegpt.data_v2.language import normalize_repo
from scripts.audit_scenario_c_extraction import (
    dependency_flags,
    js_candidates,
    python_candidates,
    tokenizer as v2_tokenizer,
)
from scripts.build_scenario_c import (
    ScenarioCExtractor,
    extract_parent_candidates,
    load_scenario_b_repositories,
    sha256_text,
)

SAMPLE_PATH = Path("data/interim/scenario_c_audit/parent_sample.jsonl")
AUDIT_CANDIDATES = Path("data/interim/scenario_c_audit/extracted_candidates.jsonl")
SCENARIO_B_DIR = Path("data/processed/polycodegpt_v2/scenario_b")
TOKENIZER_PATH = Path("tokenizer/artifacts/v2/polycodegpt_tokenizer_v2.json")


def test_repo_partition_assignment_no_overlap():
    """Confirms train and valid repositories have zero overlap."""
    train_repos, valid_repos = load_scenario_b_repositories(SCENARIO_B_DIR)
    assert len(train_repos) > 0
    assert len(valid_repos) > 0
    overlap = train_repos & valid_repos
    assert len(overlap) == 0, f"Found unexpected repo overlap: {overlap}"


def test_sample_extraction_matches_audit():
    """Confirms candidate extraction logic matches the audit results."""
    assert SAMPLE_PATH.exists()
    with open(SAMPLE_PATH, "r", encoding="utf-8") as f:
        parents = [json.loads(line) for line in f if line.strip()]

    seen_hashes = set()
    retained_records = []
    duplicates = 0

    for parent in parents:
        lang = parent.get("language")
        parent["partition"] = "train"  # For testing sample
        res = extract_parent_candidates(parent)
        for cand in res["valid_candidates"]:
            h = cand["code_sha256"]
            if h in seen_hashes:
                duplicates += 1
                continue
            seen_hashes.add(h)
            retained_records.append(cand)

    assert len(retained_records) == 5185, f"Expected 5185 retained, got {len(retained_records)}"
    assert duplicates == 81, f"Expected 81 duplicates, got {duplicates}"

    py_count = sum(1 for r in retained_records if r["language"] == "python")
    js_count = sum(1 for r in retained_records if r["language"] == "javascript")
    assert py_count == 3990
    assert js_count == 1195


def test_record_id_deterministic_and_unique():
    """Confirms deterministic record_id and parent_record_id generation."""
    sample_parent = {
        "source_id": "CODEPARROT",
        "repo": "test/repo",
        "path": "module/sample.py",
        "raw_content": "def add(a, b):\n    return a + b\n",
        "partition": "train",
        "language": "python",
    }
    res1 = extract_parent_candidates(sample_parent)
    res2 = extract_parent_candidates(sample_parent)

    assert len(res1["valid_candidates"]) == 1
    c1 = res1["valid_candidates"][0]
    c2 = res2["valid_candidates"][0]

    assert c1["record_id"] == c2["record_id"]
    assert c1["parent_record_id"] == c2["parent_record_id"]
    assert len(c1["record_id"]) == 64
    assert len(c1["parent_record_id"]) == 64


def test_syntax_validity_and_token_count_reencoding():
    """Confirms AST syntax validity and token count re-encoding match."""
    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    js_parser = Parser(Language(tsjs.language()))

    sample_py = {
        "source_id": "CODEPARROT",
        "repo": "test/repo",
        "path": "sample.py",
        "raw_content": "def calc(x):\n    '''Sample doc.'''\n    return x * 2\n",
        "partition": "train",
        "language": "python",
    }
    sample_js = {
        "source_id": "CODEPARROT",
        "repo": "test/repo",
        "path": "sample.js",
        "raw_content": "function compute(a, b) {\n    return a + b;\n}\n",
        "partition": "valid",
        "language": "javascript",
    }

    for sample in (sample_py, sample_js):
        res = extract_parent_candidates(sample)
        assert len(res["valid_candidates"]) == 1
        cand = res["valid_candidates"][0]

        # Token count re-encoding
        actual_tokens = len(tok.encode(cand["text"]).ids)
        assert actual_tokens == cand["token_count"]

        # Syntax validity
        if cand["language"] == "python":
            ast.parse(cand["code"])
        else:
            tree = js_parser.parse(cand["code"].encode("utf-8"))
            assert not tree.root_node.has_error


def test_provenance_fields_present():
    """Confirms all required provenance fields are populated in extracted records."""
    required = (
        "record_id", "parent_record_id", "source_id", "source_snapshot", "repo", "path",
        "license_id", "license_evidence", "source_url", "origin_partition", "partition",
        "parent_sha256", "code_sha256", "text", "code", "schema_version"
    )
    sample = {
        "source_id": "CODEPARROT",
        "source_snapshot": "snapshot-1",
        "repo": "user/repo",
        "path": "app.py",
        "license_id": "mit",
        "license_evidence": "evidence",
        "source_url": "https://example.com",
        "raw_content": "def foo(): pass\n",
        "partition": "train",
        "language": "python",
    }
    res = extract_parent_candidates(sample)
    cand = res["valid_candidates"][0]
    for key in required:
        assert cand.get(key) is not None, f"Missing required provenance key: {key}"


def test_deduplication_and_resume_safety():
    """Tests SQLite deduplication and resume without duplicate records."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        extractor = ScenarioCExtractor(
            shard_dir=Path("data/raw/v2/codeparrot_github_code"),
            scenario_b_dir=SCENARIO_B_DIR,
            output_dir=tmp_path,
            workers=2,
            chunksize=5,
        )

        # Run bounded on 1 shard with 50 parents
        report1 = extractor.run(shard_limit=1, max_parents=50)
        retained1 = report1["retained_records"]["total"]

        # Read lines in output files
        train_lines_1 = tmp_path.joinpath("train_functions.jsonl").read_text(encoding="utf-8").splitlines()
        valid_lines_1 = tmp_path.joinpath("valid_functions.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(train_lines_1) + len(valid_lines_1) == retained1

        # Check SQLite completed shards
        conn = sqlite3.connect(tmp_path / "dedup.sqlite")
        try:
            cur = conn.cursor()
            cur.execute("SELECT shard_name FROM completed_shards")
            completed = [row[0] for row in cur.fetchall()]
            assert len(completed) == 1
        finally:
            conn.close()

        # Now simulate resuming: run again with shard_limit=1
        extractor2 = ScenarioCExtractor(
            shard_dir=Path("data/raw/v2/codeparrot_github_code"),
            scenario_b_dir=SCENARIO_B_DIR,
            output_dir=tmp_path,
            workers=2,
            chunksize=5,
        )
        report2 = extractor2.run(shard_limit=1)

        # Confirm shard was skipped because it was already completed
        assert report2["shards"]["skipped_already_completed"] == 1
        assert report2["shards"]["completed_this_run"] == 0

        # Output lines must not have been duplicated
        train_lines_2 = tmp_path.joinpath("train_functions.jsonl").read_text(encoding="utf-8").splitlines()
        valid_lines_2 = tmp_path.joinpath("valid_functions.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(train_lines_2) == len(train_lines_1)
        assert len(valid_lines_2) == len(valid_lines_1)
