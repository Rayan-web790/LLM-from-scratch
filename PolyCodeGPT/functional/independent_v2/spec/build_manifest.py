"""Build and validate the independent v2 (100-task) benchmark manifest."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from functional.independent_v2.spec.javascript_specs import DOC as JS_DOC, STUB as JS_STUB, JS_SPECS
from functional.independent_v2.spec.python_specs import DOC as PY_DOC, STUB as PY_STUB, PY_SPECS

MANIFEST_PATH = ROOT / "functional" / "independent_v2" / "manifest.json"
REPORT_PATH = ROOT / "reports" / "phase12b_independent_v2_spec.md"

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
        task_id = f"{'py' if language == 'python' else 'js'}_v2_{index:02d}"
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
            "benchmark": "independent_v2",
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

    print(f"validated {len(all_tasks)} tasks, {sum(len(t['cases']) for t in all_tasks)} cases")

    if args.check:
        return 0

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps({"benchmark": "independent_v2", "tasks": all_tasks}, indent=2) + "\n",
                            encoding="utf-8", newline="\n")

    lines = ["# Phase 12B — Independent v2 Functional Benchmark Spec", "",
             f"Tasks: {len(all_tasks)} (50 python, 50 javascript)", ""]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"wrote {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
