"""Unit tests for Phase 11 outcome classification and metric computation.

No Docker and no candidate code is involved: sandbox results are constructed
directly, and the syntax-failure path is exercised through evaluate_task, which
returns before touching the sandbox.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from functional.evaluate_functional import (  # noqa: E402
    OUTCOMES,
    TaskOutcome,
    candidate_key,
    classify,
    compute_metrics,
    evaluate_task,
    preflight_safety,
)
from functional.sandbox.runner import SandboxResult  # noqa: E402


CASES = [
    {"name": "normal", "kind": "normal", "args": [2], "expected": 4},
    {"name": "boundary", "kind": "boundary", "args": [0], "expected": 0},
]


def done(*events: dict) -> SandboxResult:
    return SandboxResult(events=[*events, {"type": "done"}], exit_code=0, duration_s=0.1)


def case(name: str, status: str, **extra) -> dict:
    return {"type": "case", "name": name, "kind": "normal", "status": status, **extra}


def test_success_requires_every_case_to_pass() -> None:
    result = done(case("normal", "pass"), case("boundary", "pass"))
    outcome, detail, cases = classify(result, CASES)
    assert outcome == "success"
    assert detail == ""
    assert [c["status"] for c in cases] == ["pass", "pass"]


def test_wrong_value_is_incorrect_output() -> None:
    result = done(case("normal", "pass"),
                  case("boundary", "fail", got="1", expected="0"))
    outcome, detail, cases = classify(result, CASES)
    assert outcome == "incorrect_output"
    assert "boundary" in detail
    assert [c["status"] for c in cases] == ["pass", "fail"]


def test_case_exception_is_runtime_error() -> None:
    result = done(case("normal", "exception", error="ZeroDivisionError"),
                  case("boundary", "pass"))
    outcome, detail, _ = classify(result, CASES)
    assert outcome == "runtime_error"
    assert "ZeroDivisionError" in detail


def test_missing_entry_is_distinct_from_runtime_error() -> None:
    result = done({"type": "entry_missing", "entry": "square"})
    outcome, detail, cases = classify(result, CASES)
    assert outcome == "missing_entry"
    assert "square" in detail
    assert all(c["status"] == "not_run" for c in cases)


def test_exec_error_is_runtime_error() -> None:
    result = SandboxResult(
        events=[{"type": "exec_error", "error": "SyntaxError", "message": "bad"}],
        exit_code=0,
    )
    outcome, detail, _ = classify(result, CASES)
    assert outcome == "runtime_error"
    assert "SyntaxError" in detail


def test_timeout_wins_over_other_signals() -> None:
    result = SandboxResult(events=[case("normal", "pass")], exit_code=137, timed_out=True)
    outcome, _, cases = classify(result, CASES)
    assert outcome == "timeout"
    assert [c["status"] for c in cases] == ["pass", "not_run"]


def test_oom_and_exit_137_are_resource_limit() -> None:
    oom = SandboxResult(events=[], exit_code=137, oom_killed=True)
    assert classify(oom, CASES)[0] == "resource_limit"
    exit137 = SandboxResult(events=[], exit_code=137, oom_killed=False)
    assert classify(exit137, CASES)[0] == "resource_limit"


def test_docker_failure_is_infrastructure_error_not_a_model_failure() -> None:
    result = SandboxResult(docker_error="no such image", exit_code=125)
    assert classify(result, CASES)[0] == "infrastructure_error"


def test_incomplete_run_without_events_is_runtime_error() -> None:
    result = SandboxResult(events=[], exit_code=1, stderr_tail="killed")
    assert classify(result, CASES)[0] == "runtime_error"


def test_outcome_taxonomy_is_complete() -> None:
    assert OUTCOMES == (
        "syntax_failure", "missing_entry", "runtime_error", "timeout",
        "resource_limit", "incorrect_output", "success",
    )


def make_task(task_id: str, language: str = "python", cases=CASES) -> dict:
    return {
        "task_id": task_id, "language": language, "entry": "square",
        "docstring": "Implement a function that returns the square of a number.",
        "signature": "def square(x):", "cases": cases,
    }


def test_syntax_failure_is_never_executed() -> None:
    task = make_task("py_49")
    candidate = {"syntax_valid": False, "syntax_candidate": "def square(x):\n  return x*"}
    outcome = evaluate_task(task, candidate, timeout_s=5)
    assert outcome.outcome == "syntax_failure"
    assert outcome.executed is False
    assert all(c["status"] == "not_run" for c in outcome.cases)


def test_empty_candidate_is_syntax_failure() -> None:
    task = make_task("py_49")
    outcome = evaluate_task(task, {"syntax_valid": True, "syntax_candidate": "   "}, 5)
    assert outcome.outcome == "syntax_failure"
    assert outcome.candidate_sha256 is None


def outcome(task_id: str, language: str, kind: str, statuses: list[str],
            kinds: list[str] | None = None) -> TaskOutcome:
    kinds = kinds or ["normal"] * len(statuses)
    cases = [
        {"name": f"c{i}", "kind": kinds[i], "status": s, "expected": "", "got": ""}
        for i, s in enumerate(statuses)
    ]
    return TaskOutcome(
        task_id=task_id, language=language, entry="f", docstring="d",
        outcome=kind, executed=kind not in ("syntax_failure",), duration_s=0.1, cases=cases,
    )


def test_metrics_use_all_tasks_as_the_denominator() -> None:
    outcomes = [
        outcome("py_1", "python", "success", ["pass", "pass", "pass"]),
        outcome("py_2", "python", "incorrect_output", ["pass", "fail", "pass"]),
        outcome("py_3", "python", "syntax_failure", ["not_run", "not_run", "not_run"]),
    ]
    metrics = compute_metrics(outcomes)["python"]
    assert metrics["tasks"] == 3
    assert metrics["functional_success_rate"] == pytest.approx(0.3333, abs=1e-4)
    assert metrics["syntax_valid_rate"] == pytest.approx(0.6667, abs=1e-4)
    assert metrics["test_case_pass_rate"] == pytest.approx(5 / 9, abs=1e-4)
    assert metrics["syntax_failure_rate"] == pytest.approx(0.3333, abs=1e-4)


def test_completion_rate_counts_fully_executed_case_sets() -> None:
    outcomes = [
        outcome("py_1", "python", "success", ["pass", "pass"]),
        outcome("py_2", "python", "timeout", ["pass", "not_run"]),
        outcome("py_3", "python", "syntax_failure", ["not_run", "not_run"]),
    ]
    metrics = compute_metrics(outcomes)["python"]
    assert metrics["completion_rate"] == pytest.approx(0.3333, abs=1e-4)
    assert metrics["timeout_rate"] == pytest.approx(0.3333, abs=1e-4)


def test_edge_case_success_rate_counts_boundary_and_error_cases() -> None:
    cases = ["normal", "boundary", "error"]
    outcomes = [
        outcome("py_1", "python", "incorrect_output", ["pass", "fail", "pass"], cases),
    ]
    metrics = compute_metrics(outcomes)["python"]
    assert metrics["edge_case_success_rate"] == pytest.approx(0.5)


def test_convention_sensitivity_ignores_convention_cases() -> None:
    task_cases = [
        {"name": "found", "kind": "normal", "status": "pass"},
        {"name": "miss", "kind": "normal", "status": "fail", "convention": True},
    ]
    o = TaskOutcome(task_id="py_5", language="python", entry="f", docstring="d",
                    outcome="incorrect_output", executed=True, duration_s=0.0, cases=task_cases)
    metrics = compute_metrics([o])["python"]
    assert metrics["functional_success_rate"] == 0.0
    assert metrics["convention_sensitivity"]["tasks_passing_all_non_convention_cases"] == 1
    assert metrics["convention_sensitivity"]["rate"] == 1.0


def test_infrastructure_errors_are_excluded_from_scoring() -> None:
    outcomes = [
        outcome("py_1", "python", "success", ["pass"]),
        outcome("py_2", "python", "infrastructure_error", ["not_run"]),
    ]
    metrics = compute_metrics(outcomes)["python"]
    assert metrics["tasks"] == 2
    assert metrics["tasks_scored"] == 1
    assert metrics["infrastructure_errors"] == 1
    assert metrics["functional_success_rate"] == pytest.approx(0.5)


def test_metrics_are_per_language() -> None:
    outcomes = [
        outcome("py_1", "python", "success", ["pass"]),
        outcome("js_1", "javascript", "incorrect_output", ["fail"]),
    ]
    metrics = compute_metrics(outcomes)
    assert metrics["python"]["functional_success_rate"] == 1.0
    assert metrics["javascript"]["functional_success_rate"] == 0.0


def test_candidate_key_includes_language_and_stub() -> None:
    a = candidate_key("python", "doc", "def f(x):")
    b = candidate_key("javascript", "doc", "def f(x):")
    assert a != b


def test_preflight_refuses_missing_safety_report(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        preflight_safety(tmp_path / "absent.json")


def test_preflight_refuses_failed_probes(tmp_path: Path) -> None:
    path = tmp_path / "safety.json"
    path.write_text(json.dumps({"passed": 3, "total": 4}), encoding="utf-8")
    with pytest.raises(SystemExit):
        preflight_safety(path)


def test_preflight_accepts_fully_passed_probes(tmp_path: Path) -> None:
    path = tmp_path / "safety.json"
    path.write_text(json.dumps({"passed": 16, "total": 16}), encoding="utf-8")
    assert preflight_safety(path)["passed"] == 16


def test_committed_safety_report_passed_all_probes() -> None:
    payload = preflight_safety(PROJECT_ROOT / "reports" / "phase11_sandbox_security.json")
    assert payload["passed"] == payload["total"] >= 10
