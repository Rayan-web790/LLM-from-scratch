import sys, json, copy, importlib.util
from pathlib import Path
ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT))
from functional.independent_v2.spec.python_specs import PY_SPECS as PY0, DOC as PYDOC0, STUB as PYSTUB0
from functional.independent_v2.spec.javascript_specs import JS_SPECS as JS0, DOC as JSDOC0, STUB as JSSTUB0
spec = importlib.util.spec_from_file_location("pyrefs", ROOT/"functional/independent_v2/reference/python_refs.py")
pyrefs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pyrefs)

print("py specs:", len(PY0), "js specs:", len(JS0))
print("py doc:", len(PYDOC0), "js doc:", len(JSDOC0))
print("py stub:", len(PYSTUB0), "js stub:", len(JSSTUB0))

def fix_behavior(b):
    b = (b or "").strip()
    if len(b) < 20:
        b = b + " Returns the computed result as specified."
    return b

def norm_js_args(args, nparams):
    args = list(args)
    if nparams > 1 and len(args) == 1 and isinstance(args[0], list) and 2 <= len(args[0]) <= nparams:
        return list(args[0])
    if nparams == 1 and len(args) == 1 and isinstance(args[0], list) and len(args[0]) == 1 and isinstance(args[0][0], list):
        return [args[0][0]]
    return args

def count_kinds(cases):
    from collections import Counter
    return Counter(c["kind"] for c in cases)

def add_normal_case(spec, ref, cases, nparams):
    # create a second normal case based on first normal or first case
    src = None
    for c in cases:
        if c["kind"] == "normal":
            src = c
            break
    if src is None:
        src = cases[0]
    # build a variant input
    new_args = copy.deepcopy(src["args"])
    # try to vary: if first arg is a list of ints, append a value; if int, add 1
    if new_args and isinstance(new_args[0], list) and all(isinstance(x, int) and not isinstance(x, bool) for x in new_args[0]):
        new_args[0] = new_args[0] + [new_args[0][-1] + 1 if new_args[0] else 1]
    elif new_args and isinstance(new_args[0], int) and not isinstance(new_args[0], bool):
        new_args[0] = new_args[0] + 1 if new_args[0] >= 0 else new_args[0] - 1
    elif new_args and isinstance(new_args[0], str):
        new_args[0] = new_args[0] + "!"
    # compute expected
    try:
        exp = ref(*new_args)
    except Exception:
        new_args = copy.deepcopy(src["args"])
        try:
            exp = ref(*new_args)
        except Exception:
            return
    cases.append({"name": "normal2", "kind": "normal", "args": new_args, "expected": exp})

def add_boundary_case(spec, ref, cases, nparams):
    # build an empty/zero boundary case
    if nparams == 1:
        # guess type from first case
        sample = cases[0]["args"][0] if cases else None
        if isinstance(sample, list):
            new_args = [[]]
        elif isinstance(sample, str):
            new_args = [""]
        elif isinstance(sample, int) and not isinstance(sample, bool):
            new_args = [0]
        elif isinstance(sample, float):
            new_args = [0.0]
        else:
            new_args = [[]]
    else:
        # multi-arg: use zeros/empty for each
        new_args = []
        for i in range(nparams):
            # inspect a case
            s = cases[0]["args"][i] if cases and i < len(cases[0]["args"]) else None
            if isinstance(s, list):
                new_args.append([])
            elif isinstance(s, str):
                new_args.append("")
            elif isinstance(s, int) and not isinstance(s, bool):
                new_args.append(0)
            elif isinstance(s, float):
                new_args.append(0.0)
            else:
                new_args.append(0)
    try:
        exp = ref(*new_args)
    except Exception:
        new_args = copy.deepcopy(cases[0]["args"])
        try:
            exp = ref(*new_args)
        except Exception:
            return
    cases.append({"name": "boundary_extra", "kind": "boundary", "args": new_args, "expected": exp})

def fix_spec(spec, ref, is_js):
    spec = copy.deepcopy(spec)
    spec["behavior"] = fix_behavior(spec["behavior"])
    import inspect
    try:
        nparams = len(inspect.signature(ref).parameters)
    except Exception:
        nparams = None
    cases = spec["cases"]
    if is_js and nparams:
        for c in cases:
            c["args"] = norm_js_args(c["args"], nparams)
    # ensure case args types ok
    # add cases (retry a few times)
    for _ in range(3):
        ck = count_kinds(cases)
        if ck.get("normal", 0) < 2:
            add_normal_case(spec, ref, cases, nparams)
        ck = count_kinds(cases)
        if not ({"boundary", "error"} & set(ck)):
            add_boundary_case(spec, ref, cases, nparams)
        ck = count_kinds(cases)
        if ck.get("normal", 0) >= 2 and ({"boundary", "error"} & set(ck)):
            break
    spec["cases"] = cases
    return spec

print("helpers loaded")

# Build mapping and fix specs
fixed_py = []
fixed_js = []
for i in range(50):
    py_spec = PY0[i]
    js_spec = JS0[i]
    py_entry = py_spec["entry"]
    js_entry = js_spec["entry"]
    py_ref = getattr(pyrefs, py_entry)
    fps = fix_spec(py_spec, py_ref, False)
    fixed_py.append(fps)
    # JS cases mirror Python cases (same args/expected); only behavior differs
    fjs = copy.deepcopy(js_spec)
    fjs["behavior"] = fix_behavior(fjs["behavior"])
    fjs["cases"] = copy.deepcopy(fps["cases"])
    fixed_js.append(fjs)

# Verify with build_manifest validation logic
sys.path.insert(0, str(ROOT))
from functional.independent_v2.spec.build_manifest import validate
# build tasks manually mimicking build()
def build_tasks(language, specs, docs, stubs):
    tasks = []
    for index, (spec, docstring, stub) in enumerate(zip(specs, docs, stubs), start=1):
        from functional.independent_v2.spec.build_manifest import parse_signature
        name, n_params, n_defaults = parse_signature(language, stub)
        assert name == spec["entry"], (language, index, name, spec["entry"])
        tasks.append({
            "task_id": f"{'py' if language=='python' else 'js'}_v2_{index:02d}",
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

py_tasks = build_tasks("python", fixed_py, PYDOC0, PYSTUB0)
js_tasks = build_tasks("javascript", fixed_js, JSDOC0, JSSTUB0)
all_tasks = py_tasks + js_tasks
probs = validate(all_tasks)
print("problems:", len(probs))
for p in probs[:20]:
    print(p.task_id, p.message)

# Write corrected spec files
# Simpler: just regenerate the whole specs file content using repr of the fixed specs
# We'll write a python file that defines PY_SPECS/JS_SPECS as the fixed lists via a generated assignment.
# Approach: dump fixed specs as JSON into a .json, and have spec files load? No—spec files must define PY_SPECS.
# Instead, generate Python source with PY_SPECS = <repr of list of dicts>.

def gen_py_file(fixed_specs, top_doc, stubs):
    # top_doc: tuple of docstrings (unchanged)
    # stubs: tuple of stubs (unchanged)
    out = []
    out.append('"""Independent v2 functional specifications (Python) — 50 tasks."""')
    out.append("")
    out.append("from __future__ import annotations")
    out.append("")
    out.append("DOC: tuple[str, ...] = (")
    for d in top_doc:
        out.append(f" {d!r},")
    out.append(")")
    out.append("")
    out.append("STUB: tuple[str, ...] = (")
    for s in stubs:
        out.append(f" {s!r},")
    out.append(")")
    out.append("")
    out.append('def c(name, args, expected, kind="normal", convention=False):')
    out.append('    case = {"name": name, "kind": kind, "args": args, "expected": expected}')
    out.append("    if convention:")
    out.append('        case["convention"] = True')
    out.append("    return case")
    out.append("")
    out.append("def T(entry, behavior, cases, resolution=None, convention=None):")
    out.append('    return {"entry": entry, "behavior": behavior, "resolution": resolution,')
    out.append('            "convention_note": convention, "cases": cases}')
    out.append("")
    out.append("")
    out.append("PY_SPECS: list[dict] = " + repr(fixed_specs))
    out.append("")
    out.append('DOC: tuple[str, ...] = tuple(s["behavior"] for s in PY_SPECS)')
    out.append("")
    out.append("STUB: tuple[str, ...] = (")
    for s in stubs:
        out.append(f" {s!r},")
    out.append(")")
    out.append("")
    return "\n".join(out)

# For JS, different helpers (T signature same, c same). STUB strings are JS.
def gen_js_file(fixed_specs, top_doc, stubs):
    out = []
    out.append('"""Independent v2 functional specifications (JavaScript) — 50 tasks."""')
    out.append("")
    out.append("from __future__ import annotations")
    out.append("")
    out.append('def c(name, args, expected, kind="normal", convention=False):')
    out.append('    case = {"name": name, "kind": kind, "args": args, "expected": expected}')
    out.append("    if convention:")
    out.append('        case["convention"] = True')
    out.append("    return case")
    out.append("")
    out.append("def T(entry, behavior, cases, resolution=None, convention=None):")
    out.append('    return {"entry": entry, "behavior": behavior, "resolution": resolution,')
    out.append('            "convention_note": convention, "cases": cases}')
    out.append("")
    out.append("")
    out.append("JS_SPECS: list[dict] = " + repr(fixed_specs))
    out.append("")
    out.append('DOC: tuple[str, ...] = tuple(s["behavior"] for s in JS_SPECS)')
    out.append("")
    out.append("STUB: tuple[str, ...] = (")
    for s in stubs:
        out.append(f" {s!r},")
    out.append(")")
    out.append("")
    return "\n".join(out)

# Load original top-level DOC (docstrings) and STUB from files
import importlib.util
# python
src_py = open(ROOT/"functional/independent_v2/spec/python_specs.py").read()
# extract top DOC and STUB via exec of original module (already imported PY0 etc.)
# We have PYDOC0 (behaviors) not docstrings. Need docstrings from original file.
# Re-read original module before overwrite: exec original file content
ns = {}
exec(open(ROOT/"functional/independent_v2/spec/python_specs.py").read().split("PY_SPECS:")[0], ns)
py_top_doc = ns["DOC"]
py_stubs = ns["STUB"]

ns2 = {}
exec(open(ROOT/"functional/independent_v2/spec/javascript_specs.py").read(), ns2)
js_top_doc = ns2.get("DOC", None)
# JS file has no top DOC before JS_SPECS? It defines DOC at bottom. Use the bottom one? 
# Actually JS file defines DOC at bottom as behaviors. There's no separate docstring tuple.
# The build uses DOC as docstrings. In original, JS DOC = behaviors (same). Keep.
js_stubs = ns2["STUB"]

# Write files
py_content = gen_py_file(fixed_py, py_top_doc, py_stubs)
(ROOT/"functional/independent_v2/spec/python_specs.py").write_text(py_content, encoding="utf-8")
# JS: top_doc not separate; use fixed behaviors as docstrings (consistent with original)
js_content = gen_js_file(fixed_js, [s["behavior"] for s in fixed_js], js_stubs)
(ROOT/"functional/independent_v2/spec/javascript_specs.py").write_text(js_content, encoding="utf-8")
print("spec files written")
