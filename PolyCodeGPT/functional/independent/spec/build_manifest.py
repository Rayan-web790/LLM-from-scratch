"""Build and validate the independent (Phase 12B) functional benchmark manifest.

    python functional/independent/spec/build_manifest.py
    python functional/independent/spec/build_manifest.py --check
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from functional.independent.spec.javascript_specs import DOC as JS_DOC, STUB as JS_STUB, JS_SPECS
from functional.independent.spec.python_specs import DOC as PY_DOC, STUB as PY_STUB, PY_SPECS

MANIFEST_PATH = ROOT / "functional" / "independent" / "manifest.json"
REPORT_PATH = ROOT / "reports" / "phase12b_independent_spec.md"

SIGNATURE_RE = {"python": re.compile(r"def\s+(\w+)\s*\(([^)]*)\)"),
                "javascript": re.compile(r"function\s+(\w+)\s*\(([^)]*)\)")}


@dataclass
class Problem:
    task_id: str
    message: str


def parse_signature(language, stub):
    match = SIGNATURE_RE[language].search(stub)
    if not match:
        raise ValueError(f"cannot parse stub: {stub!r}")
    name, params = match.group(1), match.group(2).strip()
    if not params:
        return name, 0, 0
    parts = [p.strip() for p in params.split(",") if p.strip()]
    defaults = sum(1 for p in parts if "=" in p)
    return name, len(parts), defaults


def build(language, specs, docs, stubs):
    tasks = []
    for index, (spec, docstring, stub) in enumerate(zip(specs, docs, stubs), start=1):
        task_id = f"{'py' if language == 'python' else 'js'}_ind_{index:02d}"
        name, n_params, n_defaults = parse_signature(language, stub)
        assert name == spec["entry"], (task_id, name, spec["entry"])
        tasks.append({
            "task_id": task_id,
            "language": language,
            "index": index,
            "entry": name,
            "signature": stub,
            "docstring": docstring,
            "behavior": spec["behavior"],
            "resolution": spec["resolution"],
            "convention_note": spec["convention_note"],
            "params": n_params,
            "params_with_defaults": n_defaults,
            "cases": spec["cases"],
            "benchmark": "independent_v1",
        })
    return tasks


def validate(tasks):
    problems = []
    seen = set()
    for task in tasks:
        tid = task["task_id"]
        key = (task["language"], task["entry"])
        if key in seen:
            problems.append(Problem(tid, f"duplicate entry {task['entry']}"))
        seen.add(key)
        kinds = [c["kind"] for c in task["cases"]]
        if kinds.count("normal") < 2:
            problems.append(Problem(tid, "needs at least two normal cases"))
        if not ({"boundary", "error"} & set(kinds)):
            problems.append(Problem(tid, "needs a boundary or error case"))
        if not task["behavior"] or len(task["behavior"]) < 20:
            problems.append(Problem(tid, "behavior description missing or too vague"))
        for case in task["cases"]:
            args = case["args"]
            if not isinstance(args, list):
                problems.append(Problem(tid, f"{case['name']}: args must be a list"))
            required = task["params"] - task["params_with_defaults"]
            if not (required <= len(args) <= task["params"]):
                problems.append(Problem(tid, f"{case['name']}: {len(args)} args for signature {task['signature']}"))
            has_expected = "expected" in case
            has_error = "expect_error" in case
            if has_expected == has_error:
                problems.append(Problem(tid, f"{case['name']}: exactly one of expected/expect_error"))
            if has_error and case["kind"] != "error":
                problems.append(Problem(tid, f"{case['name']}: expect_error case must have kind=error"))
            if case["kind"] not in {"normal", "boundary", "error"}:
                problems.append(Problem(tid, f"{case['name']}: unknown kind {case['kind']}"))
    return problems


def stats(tasks):
    out = {}
    for language in ("python", "javascript"):
        subset = [t for t in tasks if t["language"] == language]
        cases = [c for t in subset for c in t["cases"]]
        out[language] = {
            "tasks": len(subset),
            "cases": len(cases),
            "normal": sum(1 for c in cases if c["kind"] == "normal"),
            "boundary": sum(1 for c in cases if c["kind"] == "boundary"),
            "error": sum(1 for c in cases if c["kind"] == "error"),
        }
    out["total"] = {"tasks": len(tasks), "cases": sum(v["cases"] for v in out.values() if isinstance(v, dict) and "cases" in v)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    py_tasks = build("python", PY_SPECS, PY_DOC, PY_STUB)
    js_tasks = build("javascript", JS_SPECS, JS_DOC, JS_STUB)
    all_tasks = py_tasks + js_tasks

    problems = validate(all_tasks)
    for p in problems:
        print(f"[PROBLEM] {p.task_id}: {p.message}")
    if problems:
        return 1

    s = stats(all_tasks)
    print(f"validated {s['total']['tasks']} tasks, {s['total']['cases']} cases")
    print(f"  python: {s['python']}")
    print(f"  javascript: {s['javascript']}")

    if args.check:
        return 0

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps({"benchmark": "independent_v1", "tasks": all_tasks}, indent=2) + "\n",
                            encoding="utf-8", newline="\n")

    # spec report
    lines = ["# Phase 12B — Independent Functional Benchmark Spec", "",
             f"Tasks: {s['total']['tasks']} ({s['python']['tasks']} python, {s['javascript']['tasks']} javascript)",
             f"Cases: {s['total']['cases']}", ""]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"wrote {MANIFEST_PATH} and {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
