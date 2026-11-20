"""Host-side orchestrator for the Phase 11 Docker sandbox.

Every candidate runs in a disposable container with:

* ``--network none``                     no network namespace connectivity
* ``--read-only`` + ``--tmpfs /tmp``     immutable root filesystem
* ``--user 10001:10001``                 unprivileged user baked into the image
* ``--cap-drop ALL``                     no Linux capabilities
* ``--security-opt no-new-privileges``   no setuid escalation
* ``--memory/--memory-swap``             hard RAM ceiling
* ``--cpus``, ``--pids-limit``, ``--ulimit``   CPU, process and file-size ceilings
* ``--rm``-equivalent cleanup            container removed after every run

No host directories are mounted; the payload travels over stdin and results come
back over stderr, so the container cannot read or write project files.

Usage::

    python functional/sandbox/runner.py --build          # build both images
    python functional/sandbox/runner.py --self-test      # smoke-test the harness
"""

from __future__ import annotations

import argparse
import json
import secrets
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent

IMAGES = {
    "python": "polycodegpt-sandbox-python:v1",
    "javascript": "polycodegpt-sandbox-node:v1",
}

DOCKERFILES = {
    "python": HERE / "Dockerfile.python",
    "javascript": HERE / "Dockerfile.node",
}

#: Resource ceilings applied to every container.
LIMITS = {
    "memory": "128m",
    "memory_swap": "128m",
    "cpus": "0.50",
    "pids": "32",
    "nofile": "64",
    "fsize": "1048576",
}

DEFAULT_TIMEOUT_S = 10.0
RUNNER_UID = "10001:10001"


@dataclass
class SandboxResult:
    """Everything observed about one sandboxed execution."""

    events: list[dict] = field(default_factory=list)
    exit_code: int | None = None
    timed_out: bool = False
    oom_killed: bool = False
    duration_s: float = 0.0
    stdout_tail: str = ""
    stderr_tail: str = ""
    docker_error: str | None = None

    @property
    def completed(self) -> bool:
        return any(e.get("type") == "done" for e in self.events)

    def by_type(self, kind: str) -> list[dict]:
        return [e for e in self.events if e.get("type") == kind]


def docker_available() -> tuple[bool, str]:
    """Return ``(ok, detail)`` describing whether the Linux engine is usable."""
    try:
        proc = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}|{{.OSType}}"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:  # pragma: no cover - env dependent
        return False, f"docker CLI unusable: {exc}"
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout).strip()[:400]
    version, _, ostype = proc.stdout.strip().partition("|")
    if ostype != "linux":
        return False, f"docker server OS is {ostype!r}, expected linux"
    return True, version


def image_exists(image: str) -> bool:
    proc = subprocess.run(
        ["docker", "image", "inspect", image], capture_output=True, text=True, timeout=60
    )
    return proc.returncode == 0


def build_images(force: bool = False) -> dict[str, str]:
    """Build both sandbox images; returns image name -> status text."""
    status: dict[str, str] = {}
    for language, image in IMAGES.items():
        if not force and image_exists(image):
            status[language] = "already present"
            continue
        proc = subprocess.run(
            ["docker", "build", "-f", str(DOCKERFILES[language]), "-t", image, str(HERE)],
            capture_output=True, text=True, timeout=900,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"build failed for {language}: {proc.stderr[-800:]}")
        status[language] = "built"
    return status


def _container_args(image: str, name: str) -> list[str]:
    return [
        "docker", "run", "--name", name,
        "--network", "none",
        "--read-only",
        "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=16m",
        "--user", RUNNER_UID,
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges:true",
        "--memory", LIMITS["memory"],
        "--memory-swap", LIMITS["memory_swap"],
        "--cpus", LIMITS["cpus"],
        "--pids-limit", LIMITS["pids"],
        "--ulimit", f"nofile={LIMITS['nofile']}:{LIMITS['nofile']}",
        "--ulimit", f"fsize={LIMITS['fsize']}:{LIMITS['fsize']}",
        "-i",
        image,
    ]


def _remove(name: str) -> None:
    subprocess.run(["docker", "rm", "-f", name], capture_output=True, text=True, timeout=60)


def _inspect(name: str) -> dict:
    proc = subprocess.run(
        ["docker", "inspect", "--format", "{{json .State}}", name],
        capture_output=True, text=True, timeout=60,
    )
    if proc.returncode != 0:
        return {}
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {}


def run_payload(
    language: str,
    payload: dict,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> SandboxResult:
    """Execute one payload inside a fresh container and parse its result lines."""
    image = IMAGES[language]
    token = secrets.token_hex(8)
    payload = {**payload, "token": token}
    name = f"ph11-{uuid.uuid4().hex[:12]}"
    result = SandboxResult()
    started = time.perf_counter()

    try:
        proc = subprocess.Popen(
            _container_args(image, name),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
    except OSError as exc:
        result.docker_error = f"failed to launch docker: {exc}"
        return result

    try:
        out, err = proc.communicate(json.dumps(payload), timeout=timeout_s)
        result.exit_code = proc.returncode
    except subprocess.TimeoutExpired:
        result.timed_out = True
        subprocess.run(["docker", "kill", name], capture_output=True, text=True, timeout=60)
        try:
            out, err = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:  # pragma: no cover - docker wedged
            proc.kill()
            out, err = "", ""
        result.exit_code = proc.returncode
    finally:
        state = _inspect(name)
        result.oom_killed = bool(state.get("OOMKilled"))
        _remove(name)
        result.duration_s = round(time.perf_counter() - started, 3)

    for line in (err or "").splitlines():
        if line.startswith(token):
            try:
                result.events.append(json.loads(line[len(token):]))
            except json.JSONDecodeError:
                continue
    result.stdout_tail = (out or "")[-500:]
    result.stderr_tail = "\n".join(
        line for line in (err or "").splitlines() if not line.startswith(token)
    )[-800:]
    if result.exit_code not in (0, None) and not result.timed_out and not result.events:
        result.docker_error = f"container exited {result.exit_code}: {result.stderr_tail[-300:]}"
    return result


def run_code(
    language: str,
    code: str,
    entry: str,
    cases: list[dict],
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> SandboxResult:
    """Convenience wrapper: run one candidate against its cases."""
    return run_payload(
        language,
        {"entry": entry, "code": code, "cases": cases},
        timeout_s=timeout_s,
    )


def run_raw(language: str, code: str, entry: str = "__probe__", timeout_s: float = DEFAULT_TIMEOUT_S) -> SandboxResult:
    """Run arbitrary probe code with no test cases (used by the security probes)."""
    return run_payload(language, {"entry": entry, "code": code, "cases": []}, timeout_s=timeout_s)


PROBE_ENTRYPOINTS = {"python": ("python3", ["-I", "-c"]), "javascript": ("node", ["-e"])}


def run_command(language: str, code: str, timeout_s: float = DEFAULT_TIMEOUT_S) -> SandboxResult:
    """Bypass the harness and run ``code`` with the interpreter directly.

    Only the security probes use this; it keeps the same container flags so the
    measured confinement is the confinement candidates get.
    """
    image = IMAGES[language]
    name = f"ph11-probe-{uuid.uuid4().hex[:12]}"
    binary, prefix = PROBE_ENTRYPOINTS[language]
    args = _container_args(image, name)
    args = args[:2] + ["--entrypoint", binary] + args[2:] + prefix + [code]
    result = SandboxResult()
    started = time.perf_counter()
    try:
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except OSError as exc:
        result.docker_error = f"failed to launch docker: {exc}"
        return result
    try:
        out, err = proc.communicate(timeout=timeout_s)
        result.exit_code = proc.returncode
    except subprocess.TimeoutExpired:
        result.timed_out = True
        subprocess.run(["docker", "kill", name], capture_output=True, text=True, timeout=60)
        try:
            out, err = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:  # pragma: no cover
            proc.kill()
            out, err = "", ""
        result.exit_code = proc.returncode
    finally:
        state = _inspect(name)
        result.oom_killed = bool(state.get("OOMKilled"))
        _remove(name)
        result.duration_s = round(time.perf_counter() - started, 3)
    result.stdout_tail = (out or "")[-4000:]
    result.stderr_tail = (err or "")[-2000:]
    return result


SELF_TEST_CASES = [
    {"name": "normal 5", "kind": "normal", "args": [5], "expected": 120},
    {"name": "boundary 0", "kind": "boundary", "args": [0], "expected": 1},
    {"name": "error negative", "kind": "error", "args": [-1], "expect_error": "ValueError"},
]


def self_test() -> int:
    """Prove the harness works end to end inside the sandbox."""
    samples = {
        "python": (
            "def factorial(n):\n"
            "    if n < 0:\n        raise ValueError('negative')\n"
            "    out = 1\n    for i in range(2, n + 1):\n        out *= i\n    return out\n"
        ),
        "javascript": (
            "function factorial(n) {\n"
            "  if (n < 0) throw new Error('negative');\n"
            "  let out = 1;\n  for (let i = 2; i <= n; i++) out *= i;\n  return out;\n}\n"
        ),
    }
    ok = True
    for language, code in samples.items():
        cases = [dict(c) for c in SELF_TEST_CASES]
        if language == "javascript":
            for case in cases:
                if case.get("expect_error"):
                    case["expect_error"] = "Error"
        res = run_code(language, code, "factorial", cases)
        statuses = [e.get("status") for e in res.by_type("case")]
        print(f"{language}: exit={res.exit_code} completed={res.completed} statuses={statuses}")
        if statuses != ["pass", "pass", "pass"] or not res.completed:
            print(f"  stderr: {res.stderr_tail[-400:]}")
            ok = False
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="build sandbox images")
    parser.add_argument("--force", action="store_true", help="rebuild even if present")
    parser.add_argument("--self-test", action="store_true", help="run the harness smoke test")
    args = parser.parse_args(argv)

    ok, detail = docker_available()
    print(f"docker: {'ok' if ok else 'UNAVAILABLE'} ({detail})")
    if not ok:
        return 2
    if args.build:
        for language, status in build_images(force=args.force).items():
            print(f"  {language}: {status}")
    if args.self_test:
        return self_test()
    return 0


if __name__ == "__main__":
    sys.exit(main())
