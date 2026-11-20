"""Validate the independent benchmark's reference implementations in the sandbox."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from functional.sandbox import runner

MANIFEST = ROOT / "functional" / "independent" / "manifest.json"
REFS = {
    "python": ROOT / "functional" / "independent" / "reference" / "python_refs.py",
    "javascript": ROOT / "functional" / "independent" / "reference" / "javascript_refs.js",
}


@dataclass
class TaskResult:
    task_id: str
    passed: bool
    cases_passed: int
    cases_total: int
    detail: str


def validate(timeout_s=runner.DEFAULT_TIMEOUT_S):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    sources = {lang: path.read_text(encoding="utf-8") for lang, path in REFS.items()}
    results = []
    for task in manifest["tasks"]:
        res = runner.run_code(task["language"], sources[task["language"]], task["entry"],
                              task["cases"], timeout_s=timeout_s)
        case_events = res.by_type("case")
        passed = sum(1 for c in case_events if c.get("status") == "pass")
        total = len(task["cases"])
        ok = res.completed and passed == total and not res.timed_out
        detail = ""
        if not ok:
            problems = [c for c in case_events if c.get("status") != "pass"]
            exec_errors = res.by_type("exec_error")
            missing = res.by_type("entry_missing")
            if missing:
                detail = f"entry missing: {task['entry']}"
            elif exec_errors:
                detail = f"exec error: {exec_errors[0].get('error')}: {exec_errors[0].get('message')}"
            elif res.timed_out:
                detail = "timed out"
            elif problems:
                detail = "; ".join(f"{c.get('name')}: got {c.get('got')} want {c.get('expected')}"
                                    for c in problems[:3])
            else:
                detail = f"incomplete run (exit={res.exit_code})"
        results.append(TaskResult(task["task_id"], ok, passed, total, detail))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    ok, detail = runner.docker_available()
    if not ok:
        print(f"docker unavailable: {detail}"); return 2
    for image in runner.IMAGES.values():
        if not runner.image_exists(image):
            print(f"missing image {image}"); return 2
    results = validate()
    failures = [r for r in results if not r.passed]
    total_cases = sum(r.cases_total for r in results)
    passed_cases = sum(r.cases_passed for r in results)
    for r in results:
        if not r.passed:
            print(f"[FAIL] {r.task_id}: {r.cases_passed}/{r.cases_total} — {r.detail}")
    print(f"reference validation: {len(results)-len(failures)}/{len(results)} tasks, "
          f"{passed_cases}/{total_cases} cases")
    if args.json_out:
        args.json_out.write_text(json.dumps({
            "tasks": len(results), "tasks_passed": len(results)-len(failures),
            "cases": total_cases, "cases_passed": passed_cases,
            "failures": [{"task_id": r.task_id, "detail": r.detail,
                          "cases_passed": r.cases_passed, "cases_total": r.cases_total}
                         for r in failures],
        }, indent=2) + "\n", encoding="utf-8")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
