import sys, json, copy, subprocess
from pathlib import Path
ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT))
from functional.independent_v3.spec.javascript_specs import JS_SPECS as JS0
from functional.independent_v3.spec.python_specs import PY_SPECS as PY0
REFS = (ROOT/"functional/independent_v3/reference/javascript_refs.js").read_text(encoding="utf-8")

def transform_args(args):
    if not isinstance(args, list) or len(args) == 0:
        return copy.deepcopy(args)
    if len(args) == 1:
        a = args[0]
        if isinstance(a, list):
            if a and all(isinstance(x, int) and not isinstance(x, bool) for x in a):
                return [a + [a[-1] + 2]]
            if a and all(isinstance(x, str) for x in a):
                return [a + [a[-1] + "?!"]]
            if a and isinstance(a[0], list) and a[0] and all(isinstance(x, int) and not isinstance(x, bool) for x in a[0]):
                return [[a[0] + [a[0][-1] + 2]]]
            return [list(a) + [0]] if isinstance(a, list) else [a]
        if isinstance(a, int) and not isinstance(a, bool):
            return [a + (2 if a >= 0 else -2)]
        if isinstance(a, str):
            return [a + "?!"]
        if isinstance(a, float):
            return [a + 2.0]
        return [a]
    new = copy.deepcopy(args)
    first = new[0]
    if isinstance(first, list) and first and all(isinstance(x, int) and not isinstance(x, bool) for x in first):
        new[0] = first + [first[-1] + 2]
    elif isinstance(first, int) and not isinstance(first, bool):
        new[0] = first + 2
    elif isinstance(first, str):
        new[0] = first + "?!"
    return new

def js_eval(entry, args):
    code = REFS + "\n" + f"const __r = {entry}(...{json.dumps(args)});\nconsole.log(JSON.stringify(__r));\n"
    r = subprocess.run(["node", "-e", code], capture_output=True, text=True, timeout=10)
    if r.returncode != 0:
        return None, r.stderr.strip()
    out = r.stdout.strip()
    try:
        return json.loads(out), None
    except Exception as e:
        return None, f"parse error: {e} ({out})"

new_specs = []
for i, sp in enumerate(JS0):
    entry = sp["entry"]
    py_sp = PY0[i]
    cases = []
    seen_names = set()
    for c in py_sp["cases"]:
        new_args = transform_args(c["args"])
        exp, err = js_eval(entry, new_args)
        if err:
            exp2, err2 = js_eval(entry, c["args"])
            if err2:
                print(f"WARN {entry} {c['name']}: {err} / {err2}", file=sys.stderr)
                continue
            new_args = copy.deepcopy(c["args"])
            exp = exp2
        name = c["name"] + "V3"
        if name in seen_names:
            name = name + str(len(cases))
        seen_names.add(name)
        cases.append({"name": name, "kind": c["kind"], "args": new_args, "expected": exp})
    normals = [c for c in cases if c["kind"] == "normal"]
    bounds = [c for c in cases if c["kind"] in ("boundary", "error")]
    if len(normals) < 2 and cases:
        na = transform_args(cases[0]["args"])
        exp, err = js_eval(entry, na)
        if not err:
            cases.append({"name": "normalExtra", "kind": "normal", "args": na, "expected": exp})
    if not bounds and cases:
        first = cases[0]["args"]
        if first and isinstance(first[0], list):
            ba = [[]]
        elif first and isinstance(first[0], str):
            ba = [""]
        elif first and isinstance(first[0], int) and not isinstance(first[0], bool):
            ba = [0]
        elif first and isinstance(first[0], float):
            ba = [0.0]
        else:
            ba = [[]]
        if len(first) > 1:
            ba = []
            for x in first:
                if isinstance(x, list): ba.append([])
                elif isinstance(x, str): ba.append("")
                elif isinstance(x, int) and not isinstance(x, bool): ba.append(0)
                elif isinstance(x, float): ba.append(0.0)
                else: ba.append(0)
        exp, err = js_eval(entry, ba)
        if not err:
            cases.append({"name": "boundaryExtra", "kind": "boundary", "args": ba, "expected": exp})
    new_sp = copy.deepcopy(sp)
    new_sp["cases"] = cases
    new_specs.append(new_sp)

ns = {}
exec(open(ROOT/"functional/independent_v3/spec/javascript_specs.py").read(), ns)
stubs = ns["STUB"]
content = '''"""Independent v3 functional specifications (JavaScript) — 50 tasks."""

from __future__ import annotations

def c(name, args, expected, kind="normal", convention=False):
    case = {"name": name, "kind": kind, "args": args, "expected": expected}
    if convention:
        case["convention"] = True
    return case

def T(entry, behavior, cases, resolution=None, convention=None):
    return {"entry": entry, "behavior": behavior, "resolution": resolution,
            "convention_note": convention, "cases": cases}

JS_SPECS: list[dict] = ''' + repr(new_specs) + '''

DOC: tuple[str, ...] = tuple(s["behavior"] for s in JS_SPECS)

STUB: tuple[str, ...] = (
''' + "".join(f" {s!r},\n" for s in stubs) + ''')
'''
(ROOT/"functional/independent_v3/spec/javascript_specs.py").write_text(content, encoding="utf-8")
print("written", len(new_specs), "specs")
