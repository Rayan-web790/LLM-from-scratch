"""Static language analysis and syntax validation.

No corpus code is executed. Python is parsed with ``ast`` and JavaScript with
tree-sitter; both are syntax-only checks.
"""

from __future__ import annotations

import ast
import re
from functools import lru_cache
from pathlib import Path

from .constants import BAD_PATH_PARTS, BAD_SUFFIXES

_JS_IMPORT_RE = re.compile(
    r"(?m)^\s*(?:import\s+(?:[^;\n]+?\s+from\s+)?['\"][^'\"]+['\"]\s*;?|"
    r"(?:const|let|var)\s+[^;\n]+?=\s*require\s*\(\s*['\"][^'\"]+['\"]\s*\)\s*;?|"
    r"export\s+(?:\*|\{[^}]*\})\s+from\s+['\"][^'\"]+['\"]\s*;?)"
)
_PY_SIGNATURE_RE = re.compile(r"(?m)^\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_]\w*)")
_JS_SIGNATURE_RE = re.compile(
    r"(?m)^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:function|class)\s+([A-Za-z_$]\w*)"
)


@lru_cache(maxsize=1)
def _js_parser():
    from tree_sitter import Language, Parser
    import tree_sitter_javascript as tsjs

    return Parser(Language(tsjs.language()))


def path_parts(path: str) -> tuple[str, ...]:
    return tuple(p.lower() for p in (path or "").replace("\\", "/").split("/") if p)


def path_rejection(path: str) -> str | None:
    """Return a stable rejection code for obviously unsuitable paths."""
    p = (path or "").replace("\\", "/").lower()
    parts = path_parts(p)
    if not p:
        return "missing_path"
    if any(part in BAD_PATH_PARTS for part in parts):
        return "vendored_or_generated_path"
    if p.endswith(BAD_SUFFIXES):
        return "generated_or_minified_suffix"
    return None


def is_test_path(path: str) -> bool:
    p = (path or "").replace("\\", "/").lower()
    parts = path_parts(p)
    name = parts[-1] if parts else ""
    return (
        any(part in {"test", "tests", "spec", "specs", "__tests__"} for part in parts[:-1])
        or name.endswith((".test.js", ".spec.js", ".test.py", ".spec.py"))
        or name.startswith("test_")
        or name == "conftest.py"
    )


def syntax_python(code: str) -> tuple[bool, str]:
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as exc:
        return False, f"SyntaxError:{exc.msg}"
    except (ValueError, TypeError) as exc:
        return False, type(exc).__name__


def syntax_javascript(code: str) -> tuple[bool, str]:
    try:
        tree = _js_parser().parse(code.encode("utf-8"))
        root = tree.root_node
        if root.has_error:
            return False, "tree_sitter_has_error"
        return True, ""
    except Exception as exc:  # parser dependency failure is a hard reject
        return False, f"parser_error:{type(exc).__name__}"


def syntax_ok(language: str, code: str) -> tuple[bool, str]:
    if language == "python":
        return syntax_python(code)
    if language == "javascript":
        return syntax_javascript(code)
    raise ValueError(f"unsupported language: {language}")


def python_imports(code: str) -> str:
    """Return normalized top-level import lines, or empty string."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ""
    lines: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            try:
                lines.append(ast.unparse(node))
            except Exception:
                continue
    return "\n".join(lines)


def javascript_imports(code: str) -> str:
    return "\n".join(m.group(0).strip() for m in _JS_IMPORT_RE.finditer(code))


def imports_for(language: str, code: str) -> str:
    if language == "python":
        return python_imports(code)
    if language == "javascript":
        return javascript_imports(code)
    return ""


def signature_for(language: str, code: str) -> str:
    """Best-effort first declaration line; never a fabricated signature."""
    pattern = _PY_SIGNATURE_RE if language == "python" else _JS_SIGNATURE_RE
    m = pattern.search(code)
    if not m:
        return ""
    line_start = code.rfind("\n", 0, m.start()) + 1
    line_end = code.find("\n", m.start())
    if line_end == -1:
        line_end = len(code)
    return code[line_start:line_end].strip()


def python_docstring(code: str) -> str:
    try:
        return (ast.get_docstring(ast.parse(code), clean=False) or "").strip()
    except SyntaxError:
        return ""


def js_function_name(code: str) -> str:
    m = _JS_SIGNATURE_RE.search(code)
    return m.group(1) if m else ""


def classify_code(language: str, code: str, path: str) -> str:
    """Assign a coverage category without discarding valid unusual constructs."""
    if is_test_path(path):
        return "tests"
    low = code.lower()
    if language == "python":
        if re.search(r"(?m)^\s*class\s+\w+", code):
            return "class"
        if re.search(r"(?m)^\s*async\s+def\s+\w+", code):
            return "async"
        if re.search(r"(?m)^\s*def\s+\w+", code):
            return "function"
        return "module"
    if re.search(r"(?m)^\s*(?:export\s+)?(?:default\s+)?class\s+\w+", code):
        return "class"
    if "async " in low or "await " in low or "promise" in low:
        return "async"
    if "module.exports" in low or re.search(r"(?m)^\s*(?:import|export)\s", code):
        return "module"
    if "this." in low or "constructor(" in low:
        return "stateful"
    if "=>" in code or re.search(r"(?m)^\s*(?:export\s+)?function\s+\w+", code):
        return "function"
    return "module"


def alignment_check(docstring: str, code: str, language: str) -> tuple[bool, str]:
    """Check that a function-level prompt/docstring is usable with its code.

    This is a structural alignment test, not a semantic judgment. It rejects
    empty/boilerplate docstrings and docstrings whose named declaration does not
    occur in the code.
    """
    doc = (docstring or "").strip()
    if len(doc) < 20:
        return False, "docstring_too_short_or_missing"
    if not (code or "").strip():
        return False, "empty_code"
    if doc.lower() in {"todo", "fixme", "none", "n/a"}:
        return False, "boilerplate_docstring"
    if language == "python":
        m = _PY_SIGNATURE_RE.search(code)
    else:
        m = _JS_SIGNATURE_RE.search(code)
    if m:
        name = m.group(1)
        if name not in code:
            return False, "declaration_name_missing"
    return True, ""


def normalize_repo(repo: str) -> str:
    repo = (repo or "").strip().lower()
    repo = repo.removeprefix("https://github.com/").removeprefix("http://github.com/")
    repo = repo.removesuffix(".git").strip("/")
    return repo


def repo_group(source_id: str, repo: str) -> str:
    """Repository identity used for split isolation, source-independent."""
    del source_id
    return normalize_repo(repo) or "unknown"


def repo_from_url(url: str) -> str:
    m = re.search(r"github\.com/([^/]+/[^/#?]+)", url or "")
    return normalize_repo(m.group(1) if m else "")


def rel_posix(path: str | Path) -> str:
    return str(path).replace("\\", "/")
