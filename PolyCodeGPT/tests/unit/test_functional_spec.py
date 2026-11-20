"""Unit tests for the Phase 11 functional benchmark specification and manifest."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from functional.spec.build_manifest import (  # noqa: E402
    MANIFEST_PATH,
    build,
    parse_signature,
    stats,
    validate,
)
from functional.spec.javascript_specs import (  # noqa: E402
    DOC as JS_DOC,
    JS_SPECS,
    STUB as JS_STUB,
)
from functional.spec.python_specs import (  # noqa: E402
    DOC as PY_DOC,
    PY_SPECS,
    STUB as PY_STUB,
)
from scripts.phase10_tasks import JS_TASKS, PYTHON_TASKS  # noqa: E402


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def all_tasks() -> list[dict]:
    return build("python", PY_SPECS, PY_DOC, PY_STUB) + build(
        "javascript", JS_SPECS, JS_DOC, JS_STUB
    )


def test_manifest_has_100_tasks_50_per_language(manifest: dict) -> None:
    tasks = manifest["tasks"]
    assert len(tasks) == 100
    assert sum(t["language"] == "python" for t in tasks) == 50
    assert sum(t["language"] == "javascript" for t in tasks) == 50
    assert manifest["success_rule"] == "all cases pass"


def test_manifest_matches_phase10_inventory_positionally(manifest: dict) -> None:
    py = [t for t in manifest["tasks"] if t["language"] == "python"]
    js = [t for t in manifest["tasks"] if t["language"] == "javascript"]
    for task, (docstring, stub) in zip(py, PYTHON_TASKS):
        assert task["docstring"] == docstring
        assert task["signature"] == stub
    for task, (docstring, stub) in zip(js, JS_TASKS):
        assert task["docstring"] == docstring
        assert task["signature"] == stub


def test_specs_validate_without_problems() -> None:
    assert validate(all_tasks()) == []


def test_case_and_task_counts(manifest: dict) -> None:
    summary = stats(manifest["tasks"])
    assert summary["python"]["tasks"] == 50
    assert summary["javascript"]["tasks"] == 50
    assert summary["python"]["cases"] == summary["javascript"]["cases"] == 210
    assert summary["python"]["error"] == 2
    assert summary["javascript"]["error"] == 2


def test_every_task_has_normal_and_edge_cases(manifest: dict) -> None:
    for task in manifest["tasks"]:
        kinds = [c["kind"] for c in task["cases"]]
        assert kinds.count("normal") >= 2, task["task_id"]
        assert {"boundary", "error"} & set(kinds), task["task_id"]


def test_entries_unique_per_language(manifest: dict) -> None:
    keys = {(t["language"], t["entry"]) for t in manifest["tasks"]}
    assert len(keys) == 100


def test_convention_cases_have_documented_resolution(manifest: dict) -> None:
    for task in manifest["tasks"]:
        if any(c.get("convention") for c in task["cases"]):
            assert task["resolution"], task["task_id"]
            assert task["convention_note"], task["task_id"]


def test_error_cases_declare_expect_error_only(manifest: dict) -> None:
    for task in manifest["tasks"]:
        for case in task["cases"]:
            has_expected = "expected" in case
            has_error = "expect_error" in case
            assert has_expected != has_error, (task["task_id"], case["name"])
            if has_error:
                assert case["kind"] == "error"
                assert case["expect_error"]


def test_python_error_contract_is_value_error(manifest: dict) -> None:
    task = next(t for t in manifest["tasks"] if t["task_id"] == "py_22")
    errors = [c["expect_error"] for c in task["cases"] if "expect_error" in c]
    assert errors and all(e == "ValueError" for e in errors)


def test_javascript_error_contract_accepts_error_subclasses(manifest: dict) -> None:
    task = next(t for t in manifest["tasks"] if t["task_id"] == "js_22")
    errors = [c["expect_error"] for c in task["cases"] if "expect_error" in c]
    assert errors and all(
        isinstance(e, list) and "Error" in e and "RangeError" in e for e in errors
    )


def test_parse_signature_handles_defaults() -> None:
    assert parse_signature("python", "def fib_memo(n, memo={}):") == ("fib_memo", 2, 1)
    assert parse_signature("javascript", "function fibMemo(n, memo) {") == ("fibMemo", 2, 0)
    assert parse_signature("python", "def no_args():") == ("no_args", 0, 0)


def test_case_argument_counts_fit_signatures(manifest: dict) -> None:
    for task in manifest["tasks"]:
        required = task["params"] - task["params_with_defaults"]
        for case in task["cases"]:
            assert required <= len(case["args"]) <= task["params"], (
                task["task_id"], case["name"], task["signature"]
            )


def test_fib_memo_cases_call_with_single_argument_in_python() -> None:
    task = next(t for t in all_tasks() if t["task_id"] == "py_24")
    assert all(len(c["args"]) == 1 for c in task["cases"])


def test_javascript_fib_memo_passes_memo_explicitly() -> None:
    task = next(t for t in all_tasks() if t["task_id"] == "js_24")
    assert all(len(c["args"]) == 2 and c["args"][1] == {} for c in task["cases"])


def test_report_markdown_exists_and_mentions_ambiguity_policy() -> None:
    report = PROJECT_ROOT / "reports" / "phase11_functional_spec.md"
    text = report.read_text(encoding="utf-8")
    assert "Ambiguity policy" in text
    assert text.count("### py_") == 50
    assert text.count("### js_") == 50
