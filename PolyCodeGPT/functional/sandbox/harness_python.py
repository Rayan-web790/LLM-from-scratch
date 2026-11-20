"""In-container Python harness (runs as an unprivileged user, no network).

Reads one JSON payload from stdin:

    {"token": "<random>", "entry": "factorial",
     "code": "<full candidate source>",
     "cases": [{"name": "n=5", "kind": "normal", "args": [5], "expected": 120}, ...]}

Result lines go to **stderr**, prefixed with the per-run token, as JSON. stdout is
swallowed so candidate prints cannot corrupt the protocol, and the token prevents
a candidate from forging results.

Error cases use ``"expect_error": "ValueError"`` (or ``"any"``) instead of
``"expected"``.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import traceback


def strict_equal(actual, expected) -> bool:
    """Equality that does not let True == 1 or 1 == 1.0 slip through."""
    if isinstance(actual, bool) or isinstance(expected, bool):
        return isinstance(actual, bool) and isinstance(expected, bool) and actual == expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        if isinstance(actual, float) or isinstance(expected, float):
            try:
                return abs(float(actual) - float(expected)) <= 1e-9 * max(1.0, abs(float(expected)))
            except (TypeError, ValueError, OverflowError):
                return False
        return actual == expected
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            strict_equal(a, e) for a, e in zip(actual, expected)
        )
    if isinstance(actual, tuple) and isinstance(expected, list):
        return strict_equal(list(actual), expected)
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            strict_equal(actual[k], expected[k]) for k in actual
        )
    if type(actual) is not type(expected):
        return False
    return actual == expected


def show(value, limit: int = 300) -> str:
    try:
        text = repr(value)
    except Exception:  # pragma: no cover - hostile __repr__
        text = f"<unrepr {type(value).__name__}>"
    return text if len(text) <= limit else text[:limit] + "..."


def main() -> int:
    payload = json.load(sys.stdin)
    token = payload["token"]
    sink = io.StringIO()

    def emit(obj: dict) -> None:
        sys.stderr.write(token + json.dumps(obj, default=str) + "\n")
        sys.stderr.flush()

    namespace = {"__name__": "candidate", "__builtins__": __builtins__}
    try:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            exec(compile(payload["code"], "<candidate>", "exec"), namespace)
    except BaseException as exc:  # noqa: BLE001 - candidate failures are data
        emit({
            "type": "exec_error",
            "error": type(exc).__name__,
            "message": str(exc)[:500],
            "traceback": traceback.format_exc(limit=2)[-500:],
        })
        return 0

    fn = namespace.get(payload["entry"])
    if not callable(fn):
        emit({"type": "entry_missing", "entry": payload["entry"]})
        return 0

    emit({"type": "entry_found"})

    for case in payload["cases"]:
        want_error = case.get("expect_error")
        try:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                value = fn(*case["args"])
        except BaseException as exc:  # noqa: BLE001
            if want_error:
                if isinstance(want_error, list):
                    ok = type(exc).__name__ in want_error
                else:
                    ok = want_error == "any" or type(exc).__name__ == want_error
                emit({
                    "type": "case",
                    "name": case["name"],
                    "kind": case.get("kind", "normal"),
                    "status": "pass" if ok else "fail",
                    "got": f"raised {type(exc).__name__}: {str(exc)[:200]}",
                    "expected": f"raises {want_error}",
                })
            else:
                emit({
                    "type": "case",
                    "name": case["name"],
                    "kind": case.get("kind", "normal"),
                    "status": "exception",
                    "error": type(exc).__name__,
                    "message": str(exc)[:300],
                })
            continue

        if want_error:
            emit({
                "type": "case",
                "name": case["name"],
                "kind": case.get("kind", "normal"),
                "status": "fail",
                "got": show(value),
                "expected": f"raises {want_error}",
            })
            continue

        ok = strict_equal(value, case["expected"])
        emit({
            "type": "case",
            "name": case["name"],
            "kind": case.get("kind", "normal"),
            "status": "pass" if ok else "fail",
            "got": show(value),
            "expected": show(case["expected"]),
        })

    emit({"type": "done"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
