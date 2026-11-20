"""Phase 14 tooling compatibility tests.

These tests do not run model generation or execute candidate code. They verify
that the Phase 10/11 defaults are preserved and that the frozen independent_v3
suite can be loaded by the updated tooling.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from functional.evaluate_functional import (  # noqa: E402
    build_candidate_indexes,
    load_manifest,
    resolve_candidate,
)
from functional.sandbox import runner  # noqa: E402
from scripts.evaluate_phase10_benchmark import (  # noqa: E402
    _phase10_generation_tasks,
    load_task_manifest,
)

INDEPENDENT_MANIFEST = PROJECT_ROOT / "functional" / "independent_v3" / "manifest.json"
PHASE11_MANIFEST = PROJECT_ROOT / "functional" / "manifest.json"


def test_default_generation_task_set_is_phase10() -> None:
    tasks = _phase10_generation_tasks()
    assert len(tasks) == 100
    assert sum(t["language"] == "python" for t in tasks) == 50
    assert sum(t["language"] == "javascript" for t in tasks) == 50
    assert all(t["task_id"] is None for t in tasks)


def test_independent_generation_manifest_loads_frozen_ids() -> None:
    tasks, suite, manifest_hash = load_task_manifest(INDEPENDENT_MANIFEST)
    assert suite == "independent_v3"
    assert len(tasks) == 100
    assert sum(t["language"] == "python" for t in tasks) == 50
    assert sum(t["language"] == "javascript" for t in tasks) == 50
    assert [t["task_id"] for t in tasks][:3] == ["py_v2_01", "py_v2_02", "py_v2_03"]
    assert manifest_hash == hashlib.sha256(INDEPENDENT_MANIFEST.read_bytes()).hexdigest()


def test_default_functional_manifest_is_phase11() -> None:
    manifest = load_manifest()
    tasks = manifest["tasks"]
    assert len(tasks) == 100
    assert sum(t["language"] == "python" for t in tasks) == 50
    assert sum(t["language"] == "javascript" for t in tasks) == 50
    total_cases = sum(len(t["cases"]) for t in tasks)
    assert total_cases == 420
    assert sum(len(t["cases"]) for t in tasks if t["language"] == "python") == 210
    assert sum(len(t["cases"]) for t in tasks if t["language"] == "javascript") == 210


def test_independent_functional_manifest_counts() -> None:
    manifest = load_manifest(INDEPENDENT_MANIFEST)
    tasks = manifest["tasks"]
    assert len(tasks) == 100
    assert sum(t["language"] == "python" for t in tasks) == 50
    assert sum(t["language"] == "javascript" for t in tasks) == 50
    assert sum(len(t["cases"]) for t in tasks) == 328
    assert sum(len(t["cases"]) for t in tasks if t["language"] == "python") == 164
    assert sum(len(t["cases"]) for t in tasks if t["language"] == "javascript") == 164
    assert tasks[0]["task_id"] == "py_v2_01"
    assert tasks[50]["task_id"] == "js_v2_01"


def test_no_implicit_fallback_to_phase11() -> None:
    independent = load_manifest(INDEPENDENT_MANIFEST)
    phase11 = load_manifest(PHASE11_MANIFEST)
    assert independent["tasks"][0]["task_id"] != phase11["tasks"][0]["task_id"]
    assert independent["tasks"][0]["task_id"].startswith("py_v2_")
    assert phase11["tasks"][0]["task_id"].startswith("py_")


def test_candidate_matching_by_independent_task_id() -> None:
    manifest = load_manifest(INDEPENDENT_MANIFEST)
    task = manifest["tasks"][0]
    candidate = {
        "task_id": task["task_id"],
        "language": task["language"],
        "docstring": task["docstring"],
        "code_stub": task["signature"],
        "syntax_valid": True,
        "syntax_candidate": "def placeholder():\n    return None",
    }
    report = {"per_task_results": [candidate]}
    by_key, by_id = build_candidate_indexes(report, manifest["tasks"])
    assert resolve_candidate(task, by_key, by_id) is candidate


def test_missing_candidate_returns_none() -> None:
    manifest = load_manifest(INDEPENDENT_MANIFEST)
    task = manifest["tasks"][0]
    report = {"per_task_results": []}
    by_key, by_id = build_candidate_indexes(report, manifest["tasks"])
    assert resolve_candidate(task, by_key, by_id) is None


def test_duplicate_candidate_task_id_fails() -> None:
    manifest = load_manifest(INDEPENDENT_MANIFEST)
    task = manifest["tasks"][0]
    candidate = {
        "task_id": task["task_id"],
        "language": task["language"],
        "docstring": task["docstring"],
        "code_stub": task["signature"],
    }
    report = {"per_task_results": [candidate, dict(candidate)]}
    with pytest.raises(ValueError, match="duplicate candidate task_id"):
        build_candidate_indexes(report, manifest["tasks"])


def test_unknown_candidate_task_id_fails() -> None:
    manifest = load_manifest(INDEPENDENT_MANIFEST)
    candidate = {
        "task_id": "not_a_real_task",
        "language": "python",
        "docstring": manifest["tasks"][0]["docstring"],
        "code_stub": manifest["tasks"][0]["signature"],
    }
    report = {"per_task_results": [candidate]}
    with pytest.raises(ValueError, match="unknown candidate task_id"):
        build_candidate_indexes(report, manifest["tasks"])


def test_mismatched_candidate_metadata_fails() -> None:
    manifest = load_manifest(INDEPENDENT_MANIFEST)
    task = manifest["tasks"][0]
    candidate = {
        "task_id": task["task_id"],
        "language": "javascript",
        "docstring": task["docstring"],
        "code_stub": task["signature"],
    }
    report = {"per_task_results": [candidate]}
    by_key, by_id = build_candidate_indexes(report, manifest["tasks"])
    with pytest.raises(ValueError, match="language mismatch"):
        resolve_candidate(task, by_key, by_id)


def test_sandbox_configuration_unchanged() -> None:
    assert runner.IMAGES == {
        "python": "polycodegpt-sandbox-python:v1",
        "javascript": "polycodegpt-sandbox-node:v1",
    }
    assert runner.LIMITS["memory"] == "128m"
    assert runner.LIMITS["cpus"] == "0.50"
    assert runner.DEFAULT_TIMEOUT_S == 10.0
    safety = json.loads((PROJECT_ROOT / "reports" / "phase11_sandbox_security.json").read_text())
    assert safety["passed"] == safety["total"] == 16
