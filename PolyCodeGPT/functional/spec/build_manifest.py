"""Build and validate the Phase 11 100-task manifest.

    python functional/spec/build_manifest.py            # write manifest + spec report
    python functional/spec/build_manifest.py --check    # validate only, no writes

Validation catches spec drift: every spec's entry point must match the function
name in the Phase 10 code stub, every case must be callable with the declared
signature, and each task must have at least two normal cases plus a boundary or
error case.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from functional.spec.javascript_specs import DOC as JS_DOC, STUB as JS_STUB, JS_SPECS  # noqa: E402
from functional.spec.python_specs import DOC as PY_DOC, STUB as PY_STUB, PY_SPECS  # noqa: E402

MANIFEST_PATH = ROOT / "functional" / "manifest.json"
REPORT_PATH = ROOT / "reports" / "phase11_functional_spec.md"

SIGNATURE_RE = {"python": re.compile(r"def\s+(\w+)\s*\(([^)]*)\)"),
                "javascript": re.compile(r"function\s+(\w+)\s*\(([^)]*)\)")}


@dataclass
class Problem:
    task_id: str
    message: str


def parse_signature(language: str, stub: str) -> tuple[str, int, int]:
    """Return (name, positional params, params with defaults)."""
    match = SIGNATURE_RE[language].search(stub)
    if not match:
        raise ValueError(f"cannot parse stub: {stub!r}")
    name, params = match.group(1), match.group(2).strip()
    if not params:
        return name, 0, 0
    parts = [p.strip() for p in params.split(",") if p.strip()]
    defaults = sum(1 for p in parts if "=" in p)
    return name, len(parts), defaults


def build(language: str, specs: list[dict], docs: tuple[str, ...], stubs: tuple[str, ...]) -> list[dict]:
    tasks: list[dict] = []
    for index, (spec, docstring, stub) in enumerate(zip(specs, docs, stubs), start=1):
        task_id = f"{'py' if language == 'python' else 'js'}_{index:02d}"
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
        })
    return tasks


def validate(tasks: list[dict]) -> list[Problem]:
    problems: list[Problem] = []
    seen: set[tuple[str, str]] = set()
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
                problems.append(
                    Problem(tid, f"{case['name']}: {len(args)} args for signature {task['signature']}")
                )
            has_expected = "expected" in case
            has_error = "expect_error" in case
            if has_expected == has_error:
                problems.append(Problem(tid, f"{case['name']}: exactly one of expected/expect_error"))
            if has_error and case["kind"] != "error":
                problems.append(Problem(tid, f"{case['name']}: expect_error case must have kind=error"))
            if case["kind"] not in {"normal", "boundary", "error"}:
                problems.append(Problem(tid, f"{case['name']}: unknown kind {case['kind']}"))
    return problems


def stats(tasks: list[dict]) -> dict:
    out: dict[str, dict] = {}
    for language in ("python", "javascript"):
        subset = [t for t in tasks if t["language"] == language]
        cases = [c for t in subset for c in t["cases"]]
        out[language] = {
            "tasks": len(subset),
            "cases": len(cases),
            "normal": sum(c["kind"] == "normal" for c in cases),
            "boundary": sum(c["kind"] == "boundary" for c in cases),
            "error": sum(c["kind"] == "error" for c in cases),
            "convention_cases": sum(bool(c.get("convention")) for c in cases),
            "documented_ambiguities": sum(bool(t["resolution"]) for t in subset),
        }
    return out


def write_report(tasks: list[dict], summary: dict) -> None:
    lines = [
        "# Phase 11 — Functional Benchmark Specification",
        "",
        "100 tasks (50 Python, 50 JavaScript) taken from the frozen Phase 10 inventory. "
        "Each task defines an entry point, required behaviour, normal and boundary cases, "
        "and error behaviour where the requirements justify one.",
        "",
        "Specs are positional against `scripts/phase10_tasks.py` — the docstrings the model "
        "saw in Phase 10 are imported, not re-typed, so the benchmark cannot silently drift.",
        "",
        "## Case counts",
        "",
        "| language | tasks | cases | normal | boundary | error | convention | documented ambiguities |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for language, row in summary.items():
        lines.append(
            f"| {language} | {row['tasks']} | {row['cases']} | {row['normal']} | "
            f"{row['boundary']} | {row['error']} | {row['convention_cases']} | "
            f"{row['documented_ambiguities']} |"
        )
    lines += [
        "",
        "A task is functionally successful only when **every** case passes. Cases marked "
        "*convention* encode a return convention the docstring does not state; they are also "
        "reported separately as a sensitivity check so the ambiguity is visible in the results.",
        "",
        "## Ambiguity policy",
        "",
        "Ambiguities are resolved once, up front, and recorded per task in `resolution`. The "
        "resolutions used are:",
        "",
        "1. **Miss value for searches** (`binary_search`, `linear_search`, JS `binarySearch`, "
        "`linearSearch`) → `-1`, flagged as convention cases.",
        "2. **Indexing** of ordinal tasks (`fibonacci`, `nth_prime`, `nth_even`) → the n-th item "
        "is 1-based except Fibonacci, which is 0-based with F(0)=0.",
        "3. **Case sensitivity** of text predicates → exact comparison unless the docstring says "
        "otherwise (`same_word`/`sameWord` are explicitly case-insensitive).",
        "4. **Empty-input semantics** → documented per task; untested where the docstring gives "
        "no basis (empty `find_max`, negative `factorial`).",
        "5. **Return shape** for `countdown` → a list/array, not printed output.",
        "6. **Error contracts** → `ValueError` for Python `safe_factorial`; any `Error` instance "
        "for JS `safeFactorial`.",
        "7. **Negative operands** for `my_mod`/`myMod` → untested, because Python and JavaScript "
        "define remainder signs differently and the docstring specifies neither.",
        "",
        "Style-only constraints that cannot be observed behaviourally (`without len()`, "
        "`without abs()`, `without %`, `without Math.abs`, `without s.length`) are recorded but "
        "not enforced.",
        "",
    ]
    for language in ("python", "javascript"):
        lines += [f"## {language.capitalize()} tasks", ""]
        for task in (t for t in tasks if t["language"] == language):
            lines.append(f"### {task['task_id']} — `{task['signature']}`")
            lines.append("")
            lines.append(f"- **Docstring:** {task['docstring']}")
            lines.append(f"- **Behaviour:** {task['behavior']}")
            if task["resolution"]:
                lines.append(f"- **Resolution:** {task['resolution']}")
            if task["convention_note"]:
                lines.append(f"- **Convention:** {task['convention_note']}")
            lines.append("")
            lines.append("| case | kind | args | expected |")
            lines.append("|---|---|---|---|")
            for case in task["cases"]:
                expected = (
                    f"raises {case['expect_error']}" if "expect_error" in case
                    else json.dumps(case["expected"])
                )
                kind = case["kind"] + ("*" if case.get("convention") else "")
                lines.append(
                    f"| {case['name']} | {kind} | `{json.dumps(case['args'])}` | `{expected}` |"
                )
            lines.append("")
    lines += ["* = convention case (excluded from the sensitivity metric).", ""]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate without writing")
    args = parser.parse_args(argv)

    tasks = build("python", PY_SPECS, PY_DOC, PY_STUB) + build(
        "javascript", JS_SPECS, JS_DOC, JS_STUB
    )
    problems = validate(tasks)
    summary = stats(tasks)

    if problems:
        print(f"{len(problems)} problem(s):")
        for p in problems:
            print(f"  {p.task_id}: {p.message}")
        return 1

    print(f"validated {len(tasks)} tasks, {sum(s['cases'] for s in summary.values())} cases")
    for language, row in summary.items():
        print(f"  {language}: {row}")
    if args.check:
        return 0

    manifest = {
        "phase": 11,
        "source": "scripts/phase10_tasks.py",
        "candidate_source": "reports/phase10_generation_evaluation_20261117_153043.json",
        "success_rule": "all cases pass",
        "convention_rule": "cases with convention=true are reported but excluded from the sensitivity metric",
        "summary": summary,
        "tasks": tasks,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    write_report(tasks, summary)
    print(f"wrote {MANIFEST_PATH.relative_to(ROOT)} and {REPORT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
