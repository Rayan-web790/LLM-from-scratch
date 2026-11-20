"""Phase 11 — functional correctness evaluator.

Reads the official Phase 10 generation report (never regenerates anything),
takes each task's complete ``syntax_candidate``, rejects invalid syntax without
executing it, and runs every remaining candidate against its task's test cases
inside the verified Docker sandbox.

    python functional/evaluate_functional.py                 # full 100-task run
    python functional/evaluate_functional.py --limit 5       # quick smoke run

Outcomes per task (exactly one):

``syntax_failure``   candidate did not compile — never executed
``missing_entry``    compiled, but the required function is not defined
``runtime_error``    raised while loading the source or during a case
``timeout``          exceeded the wall-clock ceiling (host kills the container)
``resource_limit``   OOM-killed or exited 137 inside the container
``incorrect_output`` ran to completion but produced a wrong value
``success``          every case passed

Anything that goes wrong with Docker itself is reported as ``infrastructure_error``
and excluded from the candidate metrics rather than being blamed on the model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from functional.sandbox import runner  # noqa: E402

MANIFEST = ROOT / "functional" / "manifest.json"
OFFICIAL_REPORT = ROOT / "reports" / "phase10_generation_evaluation_20261117_153043.json"
SAFETY_REPORT = ROOT / "reports" / "phase11_sandbox_security.json"
OUT_JSON = ROOT / "reports" / "phase11_functional_results.json"
OUT_MD = ROOT / "reports" / "phase11_functional_results.md"

OUTCOMES = (
    "syntax_failure",
    "missing_entry",
    "runtime_error",
    "timeout",
    "resource_limit",
    "incorrect_output",
    "success",
)
OUTCOME_LABELS = {
    "syntax_failure": "Syntax failure (not executed)",
    "missing_entry": "Missing function / entry point",
    "runtime_error": "Runtime exception",
    "timeout": "Timeout",
    "resource_limit": "Resource-limit violation",
    "incorrect_output": "Incorrect output",
    "success": "Complete task success",
}


@dataclass
class TaskOutcome:
    task_id: str
    language: str
    entry: str
    docstring: str
    outcome: str
    executed: bool
    duration_s: float
    cases: list[dict] = field(default_factory=list)
    detail: str = ""
    candidate_sha256: str | None = None
    candidate_source: str | None = None

    @property
    def mandatory_cases(self) -> int:
        return len(self.cases)

    @property
    def passed_cases(self) -> int:
        return sum(1 for c in self.cases if c.get("status") == "pass")

    @property
    def edge_cases(self) -> list[dict]:
        return [c for c in self.cases if c.get("kind") in ("boundary", "error")]

    def all_non_convention_pass(self) -> bool:
        relevant = [c for c in self.cases if not c.get("convention")]
        return bool(relevant) and all(c.get("status") == "pass" for c in relevant)


def load_manifest(path: Path | str | None = None) -> dict:
    """Load a functional task manifest.

    ``path`` defaults to the historical Phase 11 manifest. A manifest path is
    never replaced silently with the default.
    """
    manifest_path = MANIFEST if path is None else Path(path)
    if not manifest_path.is_file():
        raise SystemExit(f"manifest not found: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read manifest {manifest_path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), list):
        raise SystemExit(f"manifest has no task list: {manifest_path}")
    return payload


def load_candidates(report_path: Path) -> dict[str, dict]:
    """Index candidates by (language, docstring, code_stub)."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    index: dict[str, dict] = {}
    for item in report["per_task_results"]:
        key = candidate_key(item["language"], item["docstring"], item["code_stub"])
        if key in index:
            raise ValueError(f"duplicate candidate key: {key}")
        index[key] = item
    return index


def candidate_key(language: str, docstring: str, code_stub: str) -> str:
    return f"{language}\x1f{docstring}\x1f{code_stub}"


def _sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_candidate_indexes(report: dict, tasks: list[dict]) -> tuple[dict[str, dict], dict[str, dict]]:
    """Index candidates by key and, when available, by stable task ID.

    Raises ``ValueError`` for duplicate or unknown candidate task IDs.
    """
    task_ids = {task.get("task_id") for task in tasks if task.get("task_id") is not None}
    by_key: dict[str, dict] = {}
    by_task_id: dict[str, dict] = {}
    for item in report.get("per_task_results", []):
        if not isinstance(item, dict):
            raise ValueError("candidate entry is not an object")
        language = item.get("language")
        docstring = item.get("docstring")
        code_stub = item.get("code_stub")
        if not isinstance(language, str) or not isinstance(docstring, str) or not isinstance(code_stub, str):
            raise ValueError("candidate entry is missing language/docstring/code_stub")
        task_id = item.get("task_id")
        if task_id is not None:
            if not isinstance(task_id, str) or not task_id.strip():
                raise ValueError("candidate has an invalid task_id")
            if task_id in by_task_id:
                raise ValueError(f"duplicate candidate task_id: {task_id}")
            if task_ids and task_id not in task_ids:
                raise ValueError(f"unknown candidate task_id: {task_id}")
        key = candidate_key(language, docstring, code_stub)
        if key in by_key:
            raise ValueError(f"duplicate candidate key: {key}")
        by_key[key] = item
        if task_id is not None:
            by_task_id[task_id] = item
    return by_key, by_task_id


def resolve_candidate(task: dict, by_key: dict[str, dict], by_task_id: dict[str, dict] | None = None) -> dict | None:
    """Resolve one task to a candidate, validating identity metadata."""
    by_task_id = by_task_id or {}
    task_id = task.get("task_id")
    task_stub = task.get("signature", task.get("code_stub"))

    if task_id is not None and by_task_id:
        candidate = by_task_id.get(task_id)
    else:
        if not isinstance(task_stub, str):
            return None
        candidate = by_key.get(candidate_key(task["language"], task["docstring"], task_stub))
    if candidate is None:
        return None

    if task_id is not None and candidate.get("task_id") is not None and candidate["task_id"] != task_id:
        raise ValueError(
            f"task {task_id}: candidate task_id mismatch "
            f"({candidate['task_id']!r} != {task_id!r})"
        )
    if candidate.get("language") is not None and candidate["language"] != task["language"]:
        raise ValueError(
            f"task {task.get('task_id', task.get('entry'))}: candidate language mismatch "
            f"({candidate['language']!r} != {task['language']!r})"
        )
    if candidate.get("docstring") is not None and candidate["docstring"] != task["docstring"]:
        raise ValueError(
            f"task {task.get('task_id', task.get('entry'))}: candidate docstring mismatch"
        )
    candidate_stub = candidate.get("code_stub")
    if candidate_stub is not None and task_stub is not None and candidate_stub != task_stub:
        raise ValueError(
            f"task {task.get('task_id', task.get('entry'))}: candidate code stub mismatch"
        )
    return candidate


# Backward/forward-compatible aliases for tests and callers.
find_candidate = resolve_candidate
match_candidate = resolve_candidate

def preflight_safety(path: Path = SAFETY_REPORT) -> dict:
    """Part F gate: refuse to execute candidates unless probes all passed."""
    if not path.exists():
        raise SystemExit(
            f"refusing to evaluate: {path.name} is missing. "
            "Run functional/sandbox/probes.py first (Part F)."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    passed, total = payload.get("passed", 0), payload.get("total", 0)
    if total == 0 or passed != total:
        raise SystemExit(
            f"refusing to evaluate: containment probes passed {passed}/{total}. "
            "Fix the sandbox before executing candidate code."
        )
    return payload


def classify(res: runner.SandboxResult, cases: list[dict]) -> tuple[str, str, list[dict]]:
    """Map a sandbox run onto exactly one outcome category."""
    executed_cases = res.by_type("case")
    by_name = {c.get("name"): c for c in executed_cases}
    merged: list[dict] = []
    for case in cases:
        event = by_name.get(case["name"])
        entry = {
            "name": case["name"],
            "kind": case.get("kind"),
            "status": event.get("status", "not_run") if event else "not_run",
            "got": event.get("got", "") if event else "",
            "expected": event.get("expected", "") if event else "",
        }
        # Spec metadata the harness cannot know about must survive the merge.
        if case.get("convention"):
            entry["convention"] = True
        merged.append(entry)

    if res.timed_out or res.by_type("timeout"):
        return "timeout", "program exceeded the execution time budget", merged
    if any(c.get("status") == "timeout" for c in executed_cases):
        return "timeout", "at least one case exceeded the execution time budget", merged
    if res.oom_killed or res.exit_code == 137:
        return "resource_limit", "container OOM-killed or exited 137", merged
    if res.exit_code == 134 or "out of memory" in res.stderr_tail.lower():
        return "resource_limit", "runtime aborted after exhausting its memory ceiling", merged
    if res.docker_error and not res.events:
        return "infrastructure_error", res.docker_error, merged
    missing = res.by_type("entry_missing")
    if missing:
        return "missing_entry", f"function {missing[0].get('entry')!r} is not defined", merged
    exec_errors = res.by_type("exec_error")
    if exec_errors:
        first = exec_errors[0]
        return "runtime_error", f"{first.get('error')}: {first.get('message')}", merged
    if not res.completed:
        if res.exit_code == 137:
            return "resource_limit", "container exited 137", merged
        return "runtime_error", f"run did not complete (exit={res.exit_code})", merged

    exceptions = [c for c in executed_cases if c.get("status") == "exception"]
    if exceptions:
        first = exceptions[0]
        return (
            "runtime_error",
            f"{first.get('error')} on case {first.get('name')}: {first.get('message')}",
            merged,
        )
    failures = [c for c in executed_cases if c.get("status") == "fail"]
    if failures:
        first = failures[0]
        return (
            "incorrect_output",
            f"{len(failures)} case(s) wrong, first: {first.get('name')} "
            f"got {first.get('got')} want {first.get('expected')}",
            merged,
        )
    if all(c.get("status") == "pass" for c in merged):
        return "success", "", merged
    return "runtime_error", "harness returned no usable case results", merged


def evaluate_task(task: dict, candidate: dict, timeout_s: float) -> TaskOutcome:
    source = candidate.get("syntax_candidate") or ""
    base = dict(
        task_id=task["task_id"], language=task["language"], entry=task["entry"],
        docstring=task["docstring"],
        candidate_sha256=(
            hashlib.sha256(source.encode("utf-8")).hexdigest() if source.strip() else None
        ),
    )
    if not candidate.get("syntax_valid") or not source.strip():
        return TaskOutcome(
            outcome="syntax_failure", executed=False, duration_s=0.0,
            detail="candidate did not parse; not executed", cases=[
                {**case, "status": "not_run", "got": "", "expected": ""}
                for case in task["cases"]
            ],
            candidate_source=source, **base,
        )

    res = runner.run_code(
        task["language"], source, task["entry"], task["cases"], timeout_s=timeout_s
    )
    outcome, detail, cases = classify(res, task["cases"])
    return TaskOutcome(
        outcome=outcome, executed=True, duration_s=res.duration_s,
        cases=cases, detail=detail, candidate_source=source, **base,
    )


def _rate(numerator: float, denominator: float) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def compute_metrics(outcomes: list[TaskOutcome]) -> dict[str, dict]:
    """Per-language metrics. Tasks without usable candidates count against rates."""
    metrics: dict[str, dict] = {}
    for language in ("python", "javascript"):
        subset = [o for o in outcomes if o.language == language]
        if not subset:
            continue
        tasks = len(subset)
        counted = [o for o in subset if o.outcome != "infrastructure_error"]
        cases = [c for o in counted for c in o.cases]
        edge = [c for o in counted for c in o.edge_cases]
        executed = [o for o in counted if o.executed]
        counts = {name: sum(1 for o in counted if o.outcome == name) for name in OUTCOMES}

        metrics[language] = {
            "tasks": tasks,
            "tasks_scored": len(counted),
            "infrastructure_errors": tasks - len(counted),
            "syntax_valid_rate": _rate(len(executed), tasks),
            "functional_success_rate": _rate(counts["success"], tasks),
            "test_case_pass_rate": _rate(sum(o.passed_cases for o in counted), len(cases)),
            "runtime_error_rate": _rate(counts["runtime_error"], tasks),
            "timeout_rate": _rate(counts["timeout"], tasks),
            "completion_rate": _rate(
                sum(1 for o in counted if o.cases and all(
                    c.get("status") != "not_run" for c in o.cases)),
                tasks,
            ),
            "edge_case_success_rate": _rate(
                sum(1 for c in edge if c.get("status") == "pass"), len(edge)
            ),
            "missing_entry_rate": _rate(counts["missing_entry"], tasks),
            "resource_limit_rate": _rate(counts["resource_limit"], tasks),
            "syntax_failure_rate": _rate(counts["syntax_failure"], tasks),
            "functional_success_rate_executed_only": _rate(
                sum(1 for o in executed if o.outcome == "success"), len(executed)
            ),
            "convention_sensitivity": {
                "tasks_passing_all_non_convention_cases": sum(
                    1 for o in counted if o.all_non_convention_pass()
                ),
                "rate": _rate(
                    sum(1 for o in counted if o.all_non_convention_pass()), tasks
                ),
            },
            "case_counts": {
                "total": len(cases),
                "passed": sum(o.passed_cases for o in counted),
                "normal": sum(1 for c in cases if c.get("kind") == "normal"),
                "boundary": sum(1 for c in cases if c.get("kind") == "boundary"),
                "error": sum(1 for c in cases if c.get("kind") == "error"),
                "convention": sum(1 for c in cases if c.get("convention")),
            },
            "outcome_counts": counts,
        }
    return metrics


def write_markdown(payload: dict, path: Path) -> None:
    metrics = payload["metrics"]
    outcomes = payload["task_outcomes"]
    lines = [
        "# Phase 11 — Functional Correctness Results",
        "",
        f"Generated {payload['timestamp']} from the frozen Phase 10 report "
        f"`{Path(payload['candidate_source']).name}` "
        f"(checkpoint step {payload['checkpoint_step']}, seed {payload['seed']}). "
        "No model was loaded, no generation was re-run, and no checkpoint, tokenizer or "
        "dataset file was touched.",
        "",
        "Execution was performed exclusively inside the verified Docker sandbox "
        "(`reports/phase11_sandbox_security.md`: 16/16 containment probes passed). "
        "Candidates with invalid syntax were rejected without execution.",
        "",
        "## Headline metrics",
        "",
        "| metric | Python | JavaScript |",
        "|---|---|---|",
    ]
    rows = [
        ("syntax-valid rate", "syntax_valid_rate"),
        ("functional task success rate", "functional_success_rate"),
        ("test-case pass rate", "test_case_pass_rate"),
        ("runtime-error rate", "runtime_error_rate"),
        ("timeout rate", "timeout_rate"),
        ("completion rate", "completion_rate"),
        ("edge-case success rate", "edge_case_success_rate"),
        ("functional success (executed candidates only)", "functional_success_rate_executed_only"),
        ("success excluding convention cases", None),
        ("missing-entry rate", "missing_entry_rate"),
        ("resource-limit rate", "resource_limit_rate"),
        ("syntax-failure rate", "syntax_failure_rate"),
    ]
    for label, key in rows:
        cells = []
        for language in ("python", "javascript"):
            row = metrics.get(language, {})
            value = (
                row.get("convention_sensitivity", {}).get("rate")
                if key is None else row.get(key)
            )
            cells.append(f"{value:.1%}" if isinstance(value, (int, float)) else "n/a")
        lines.append(f"| {label} | {cells[0]} | {cells[1]} |")

    lines += [
        "",
        "Rates are over all 50 tasks per language; a task that never compiled is a failure "
        "for every execution-based metric, never a silent pass.",
        "",
        "## Outcome distribution",
        "",
        "| outcome | Python | JavaScript |",
        "|---|---|---|",
    ]
    for outcome in OUTCOMES:
        counts = [metrics.get(l, {}).get("outcome_counts", {}).get(outcome, 0) for l in ("python", "javascript")]
        lines.append(f"| {OUTCOME_LABELS[outcome]} | {counts[0]} | {counts[1]} |")

    for language in ("python", "javascript"):
        lang_outcomes = [o for o in outcomes if o["language"] == language]
        lines += [
            "",
            f"## {language.capitalize()} per-task results",
            "",
            "| task | entry | outcome | cases passed | detail |",
            "|---|---|---|---|---|",
        ]
        for o in lang_outcomes:
            passed = sum(1 for c in o["cases"] if c.get("status") == "pass")
            detail = (o["detail"] or "").replace("|", "\\|")[:150]
            lines.append(
                f"| {o['task_id']} | `{o['entry']}` | {o['outcome']} | "
                f"{passed}/{len(o['cases'])} | {detail} |"
            )

    lines += [
        "",
        "## Limitations",
        "",
        "- Behaviour is judged only through the documented specification; style constraints "
        "that are not observable (`without len()`, `without %`, recursiveness) are not enforced.",
        "- Convention cases (`-1` miss value, mixed-case Caesar) are included in the primary "
        "success rate and reported separately for sensitivity.",
        "- The sandbox shares the Docker Desktop VM kernel, so containment is process/namespace "
        "level, not a hardware boundary.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=OFFICIAL_REPORT)
    parser.add_argument("--manifest", type=Path, default=None, help="functional task manifest to evaluate")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N tasks")
    parser.add_argument("--timeout", type=float, default=runner.DEFAULT_TIMEOUT_S)
    parser.add_argument("--json-out", type=Path, default=OUT_JSON)
    parser.add_argument("--md-out", type=Path, default=OUT_MD)
    args = parser.parse_args(argv)

    safety = preflight_safety()
    ok, detail = runner.docker_available()
    if not ok:
        print(f"refusing to evaluate: docker unavailable ({detail})")
        return 2
    for image in runner.IMAGES.values():
        if not runner.image_exists(image):
            print(f"refusing to evaluate: missing image {image}")
            return 2

    manifest_path = args.manifest if args.manifest is not None else MANIFEST
    manifest = load_manifest(manifest_path)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    report_meta = report
    candidates_by_key, candidates_by_id = build_candidate_indexes(report, manifest["tasks"])

    tasks = manifest["tasks"]
    if args.limit:
        tasks = tasks[: args.limit]

    outcomes: list[TaskOutcome] = []
    for task in tasks:
        try:
            candidate = resolve_candidate(task, candidates_by_key, candidates_by_id)
        except ValueError as exc:
            print(f"[INVALID] {task.get('task_id', task.get('entry'))}: {exc}")
            return 1
        if candidate is None:
            print(f"[MISSING] {task['task_id']}: no candidate in the generation report")
            continue
        result = evaluate_task(task, candidate, args.timeout)
        outcomes.append(result)
        print(
            f"[{result.outcome}] {task['task_id']} {task['entry']} "
            f"({result.passed_cases}/{len(result.cases)}) {result.detail[:110]}"
        )

    metrics = compute_metrics(outcomes)
    payload = {
        "phase": 11,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "candidate_source": str(args.report),
        "manifest": str(manifest_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "manifest_hash": _sha256_file(manifest_path),
        "checkpoint_step": report_meta.get("checkpoint_step"),
        "seed": report_meta.get("seed"),
        "safety_probes": {"passed": safety["passed"], "total": safety["total"]},
        "sandbox": {"images": runner.IMAGES, "limits": runner.LIMITS,
                    "timeout_s": args.timeout, "network": "none"},
        "success_rule": "a task succeeds only if every case passes",
        "metrics": metrics,
        "task_outcomes": [
            {
                "task_id": o.task_id, "language": o.language, "entry": o.entry,
                "docstring": o.docstring, "outcome": o.outcome, "executed": o.executed,
                "duration_s": o.duration_s, "detail": o.detail,
                "candidate_sha256": o.candidate_sha256,
                "candidate_source": o.candidate_source,
                "cases": o.cases,
            }
            for o in outcomes
        ],
    }
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    write_markdown(payload, args.md_out)

    for language, row in metrics.items():
        print(f"{language}: success {row['functional_success_rate']:.1%}, "
              f"case pass {row['test_case_pass_rate']:.1%}, "
              f"syntax-valid {row['syntax_valid_rate']:.1%}")
    print(f"wrote {args.json_out.name} and {args.md_out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
