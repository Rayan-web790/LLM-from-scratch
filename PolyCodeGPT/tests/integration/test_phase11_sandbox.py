"""Integration tests: real container execution of the sandbox contract.

These run only when Docker Desktop's Linux engine and both sandbox images are
present; otherwise they skip. Nothing here is model-generated code — the sources
are small fixtures written for the test, and they execute exclusively inside the
sandbox (never on the host or through the project virtual environment).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from functional.evaluate_functional import classify  # noqa: E402
from functional.sandbox import runner  # noqa: E402

CASES = [
    {"name": "normal", "kind": "normal", "args": [4], "expected": 16},
    {"name": "boundary", "kind": "boundary", "args": [0], "expected": 0},
]

PY_SQUARE = "def square(x):\n    return x * x\n"
PY_WRONG = "def square(x):\n    return x + x\n"
PY_THROWS = "def square(x):\n    raise RuntimeError('boom')\n"
PY_MISSING = "def cube(x):\n    return x * x * x\n"
JS_SQUARE = "function square(x) { return x * x; }\n"
JS_WRONG = "function square(x) { return x + x; }\n"


def docker_ready() -> bool:
    ok, _ = runner.docker_available()
    return ok and all(runner.image_exists(image) for image in runner.IMAGES.values())


pytestmark = pytest.mark.skipif(
    not docker_ready(), reason="Docker Linux engine or Phase 11 sandbox images unavailable"
)


@pytest.mark.parametrize(
    "language,source",
    [("python", PY_SQUARE), ("javascript", JS_SQUARE)],
)
def test_correct_candidate_succeeds(language: str, source: str) -> None:
    res = runner.run_code(language, source, "square", CASES)
    outcome, _, cases = classify(res, CASES)
    assert outcome == "success"
    assert [c["status"] for c in cases] == ["pass", "pass"]


@pytest.mark.parametrize(
    "language,source",
    [("python", PY_WRONG), ("javascript", JS_WRONG)],
)
def test_wrong_candidate_is_incorrect_output(language: str, source: str) -> None:
    res = runner.run_code(language, source, "square", CASES)
    assert classify(res, CASES)[0] == "incorrect_output"


@pytest.mark.parametrize("language,source", [("python", PY_THROWS), ("javascript", "function square(x) { throw new Error('boom'); }")])
def test_raising_candidate_is_runtime_error(language: str, source: str) -> None:
    res = runner.run_code(language, source, "square", CASES)
    assert classify(res, CASES)[0] == "runtime_error"


@pytest.mark.parametrize("language,source", [("python", PY_MISSING), ("javascript", "function cube(x) { return x * x * x; }")])
def test_missing_function_is_reported(language: str, source: str) -> None:
    res = runner.run_code(language, source, "square", CASES)
    assert classify(res, CASES)[0] == "missing_entry"


@pytest.mark.parametrize("language,source", [("python", "while True:\n    pass\n"), ("javascript", "while (true) {}\n")])
def test_infinite_loop_is_killed(language: str, source: str) -> None:
    res = runner.run_code(language, source, "square", CASES, timeout_s=4.0)
    # Python is stopped by the host-side kill; JavaScript is stopped earlier by
    # the in-VM budget. Either way the task must be a timeout, not an error.
    assert res.timed_out or res.by_type("timeout")
    assert classify(res, CASES)[0] == "timeout"


@pytest.mark.parametrize(
    "language,source",
    [
        ("python", "buf = bytearray(600 * 1024 * 1024)\ndef square(x):\n    return x * x\n"),
        ("javascript", "const a = new Array(60 * 1024 * 1024).fill(1);\nfunction square(x) { return x * x; }\n"),
    ],
)
def test_memory_hog_hits_resource_limit(language: str, source: str) -> None:
    # Either the runtime raises/aborts (V8 heap limit) or the container is
    # OOM-killed; both must classify as a resource-limit violation.
    res = runner.run_code(language, source, "square", CASES, timeout_s=25.0)
    assert classify(res, CASES)[0] == "resource_limit"


def test_container_cannot_write_outside_tmp() -> None:
    code = (
        "for p in ('/opt/harness/probe.txt', '/etc/probe.txt'):\n"
        "    try:\n"
        "        open(p, 'w').write('x')\n"
        "        print('WROTE', p)\n"
        "    except OSError as exc:\n"
        "        print('BLOCKED', type(exc).__name__)\n"
    )
    res = runner.run_command("python", code, timeout_s=20.0)
    assert "WROTE" not in res.stdout_tail
    assert "BLOCKED" in res.stdout_tail


def test_container_has_no_network() -> None:
    code = (
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('8.8.8.8', 53), timeout=3)\n"
        "    print('CONNECTED')\n"
        "except OSError as exc:\n"
        "    print('BLOCKED', type(exc).__name__)\n"
    )
    res = runner.run_command("python", code, timeout_s=20.0)
    assert "CONNECTED" not in res.stdout_tail
    assert "BLOCKED" in res.stdout_tail


def test_container_runs_as_non_root() -> None:
    res = runner.run_command("python", "import os\nprint('UID', os.getuid())\n", timeout_s=20.0)
    assert "UID 10001" in res.stdout_tail


def test_reference_oracle_passes_every_case_for_one_task() -> None:
    import json

    manifest = json.loads((PROJECT_ROOT / "functional" / "manifest.json").read_text(encoding="utf-8"))
    task = next(t for t in manifest["tasks"] if t["task_id"] == "py_49")
    source = (PROJECT_ROOT / "functional" / "reference" / "python_refs.py").read_text(encoding="utf-8")
    res = runner.run_code("python", source, task["entry"], task["cases"])
    assert classify(res, task["cases"])[0] == "success"
