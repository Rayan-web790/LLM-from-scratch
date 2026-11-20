"""Part F containment probes for the Phase 11 sandbox.

Each probe is a harmless, self-contained program that *tries* to break a specific
containment property and reports what happened. Nothing is executed on the host:
every probe runs in the same disposable container configuration candidates get.

    python functional/sandbox/probes.py

Writes ``reports/phase11_sandbox_security.json`` and
``reports/phase11_sandbox_security.md``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from functional.sandbox import runner  # noqa: E402

REPORTS = ROOT / "reports"


@dataclass
class Probe:
    id: str
    language: str
    property: str
    description: str
    code: str
    expect: str          # "blocked" | "allowed" | "killed"
    timeout_s: float = runner.DEFAULT_TIMEOUT_S


@dataclass
class ProbeResult:
    id: str
    language: str
    property: str
    description: str
    expect: str
    observed: str
    passed: bool
    exit_code: int | None
    timed_out: bool
    oom_killed: bool
    duration_s: float


PY = [
    Probe(
        id="py_net_tcp",
        language="python",
        property="no network (TCP connect)",
        description="connect to 8.8.8.8:53",
        code=(
            "import socket\n"
            "try:\n"
            "    socket.create_connection(('8.8.8.8', 53), timeout=3)\n"
            "    print('CONNECTED')\n"
            "except OSError as e:\n"
            "    print('BLOCKED', type(e).__name__, e)\n"
        ),
        expect="blocked",
    ),
    Probe(
        id="py_net_dns",
        language="python",
        property="no network (DNS)",
        description="resolve example.com",
        code=(
            "import socket\n"
            "try:\n"
            "    print('RESOLVED', socket.getaddrinfo('example.com', 80)[0][4])\n"
            "except OSError as e:\n"
            "    print('BLOCKED', type(e).__name__, e)\n"
        ),
        expect="blocked",
    ),
    Probe(
        id="py_fs_root_write",
        language="python",
        property="read-only root filesystem",
        description="write /opt/harness/probe.txt and /etc/probe.txt",
        code=(
            "paths = ['/opt/harness/probe.txt', '/etc/probe.txt', '/harness_probe.txt']\n"
            "for p in paths:\n"
            "    try:\n"
            "        open(p, 'w').write('x')\n"
            "        print('WROTE', p)\n"
            "    except OSError as e:\n"
            "        print('BLOCKED', p, type(e).__name__, e.errno)\n"
        ),
        expect="blocked",
    ),
    Probe(
        id="py_host_paths",
        language="python",
        property="no host project files visible",
        description="look for the Windows project path inside the container",
        code=(
            "import os\n"
            "candidates = ['/c/Users', 'C:/Users', '/mnt/c/Users', '/workspace', '/project']\n"
            "found = [p for p in candidates if os.path.exists(p)]\n"
            "print('FOUND', found)\n"
            "print('ROOT_ENTRIES', sorted(os.listdir('/'))[:20])\n"
        ),
        expect="blocked",
    ),
    Probe(
        id="py_mounts",
        language="python",
        property="no host bind mounts",
        description="inspect /proc/self/mountinfo for host paths",
        code=(
            "info = open('/proc/self/mountinfo').read()\n"
            "hosts = [l for l in info.splitlines() if 'PolyCodeGPT' in l or 'LLM FROM SCRATCH' in l or '/c/Users' in l]\n"
            "root = [l for l in info.splitlines() if ' / ' in l]\n"
            "print('HOST_MOUNTS', len(hosts))\n"
            "print('ROOT_LINE', root[0] if root else 'none')\n"
        ),
        expect="blocked",
    ),
    Probe(
        id="py_identity",
        language="python",
        property="non-root user",
        description="print uid/gid and effective capabilities",
        code=(
            "import os\n"
            "print('UID', os.getuid())\n"
            "print('EUID', os.geteuid())\n"
            "status = open('/proc/self/status').read()\n"
            "for key in ('CapEff', 'NoNewPrivs', 'Seccomp'):\n"
            "    for line in status.splitlines():\n"
            "        if line.startswith(key + ':'):\n"
            "            print(key.upper(), line.split(':', 1)[1].strip())\n"
        ),
        expect="allowed",
    ),
    Probe(
        id="py_timeout",
        language="python",
        property="execution timeout",
        description="infinite loop must be killed by the host",
        code="while True:\n    pass\n",
        expect="killed",
        timeout_s=5.0,
    ),
    Probe(
        id="py_memory",
        language="python",
        property="memory ceiling",
        description="allocate 512 MiB under a 128 MiB cap",
        code=(
            "try:\n"
            "    b = bytearray(512 * 1024 * 1024)\n"
            "    print('ALLOCATED', len(b))\n"
            "except MemoryError as e:\n"
            "    print('BLOCKED MemoryError')\n"
        ),
        expect="blocked",
    ),
    Probe(
        id="py_pids",
        language="python",
        property="process ceiling",
        description="fork until the pids limit stops us",
        code=(
            "import os\n"
            "n = 0\n"
            "try:\n"
            "    while n < 200:\n"
            "        os.fork()\n"
            "        n += 1\n"
            "    print('FORKED', n)\n"
            "except OSError as e:\n"
            "    print('BLOCKED fork after', n, type(e).__name__)\n"
        ),
        expect="blocked",
    ),
    Probe(
        id="py_fsize",
        language="python",
        property="file-size ceiling",
        description="write 8 MiB to /tmp under a 1 MiB fsize limit",
        code=(
            "try:\n"
            "    with open('/tmp/big.bin', 'wb') as f:\n"
            "        f.write(b'x' * (8 * 1024 * 1024))\n"
            "    print('WROTE')\n"
            "except OSError as e:\n"
            "    print('BLOCKED', type(e).__name__, e.errno)\n"
        ),
        expect="blocked",
    ),
]

JS = [
    Probe(
        id="js_net_fetch",
        language="javascript",
        property="no network (fetch)",
        description="fetch http://1.1.1.1",
        code=(
            "fetch('http://1.1.1.1', {signal: AbortSignal.timeout(3000)})\n"
            "  .then(() => console.log('CONNECTED'))\n"
            "  .catch(e => console.log('BLOCKED', e.name, e.cause && e.cause.code));\n"
        ),
        expect="blocked",
        timeout_s=8.0,
    ),
    Probe(
        id="js_net_tcp",
        language="javascript",
        property="no network (TCP)",
        description="net.connect to 8.8.8.8:53",
        code=(
            "const net = require('node:net');\n"
            "const s = net.connect(53, '8.8.8.8');\n"
            "s.on('connect', () => { console.log('CONNECTED'); s.destroy(); });\n"
            "s.on('error', e => console.log('BLOCKED', e.code));\n"
            "setTimeout(() => { console.log('BLOCKED timeout'); s.destroy(); }, 3000);\n"
        ),
        expect="blocked",
        timeout_s=8.0,
    ),
    Probe(
        id="js_fs_root_write",
        language="javascript",
        property="read-only root filesystem",
        description="write /opt/harness/probe.txt",
        code=(
            "const fs = require('node:fs');\n"
            "for (const p of ['/opt/harness/probe.txt', '/etc/probe.txt']) {\n"
            "  try { fs.writeFileSync(p, 'x'); console.log('WROTE', p); }\n"
            "  catch (e) { console.log('BLOCKED', p, e.code); }\n"
            "}\n"
        ),
        expect="blocked",
    ),
    Probe(
        id="js_identity",
        language="javascript",
        property="non-root user",
        description="print uid/gid",
        code="console.log('UID', process.getuid(), 'GID', process.getgid());\n",
        expect="allowed",
    ),
    Probe(
        id="js_timeout",
        language="javascript",
        property="execution timeout",
        description="infinite loop must be killed by the host",
        code="while (true) {}\n",
        expect="killed",
        timeout_s=5.0,
    ),
    Probe(
        id="js_memory",
        language="javascript",
        property="memory ceiling",
        description="allocate 512 MiB under a 128 MiB cap",
        code=(
            "try {\n"
            "  const a = Buffer.alloc(512 * 1024 * 1024, 1);\n"
            "  console.log('ALLOCATED', a.length);\n"
            "} catch (e) { console.log('BLOCKED', e.name); }\n"
        ),
        expect="blocked",
    ),
]


def classify(probe: Probe, res: runner.SandboxResult) -> tuple[str, bool]:
    out = (res.stdout_tail or "").strip().replace("\r", "")
    last = out.splitlines()[-1] if out else ""

    if probe.expect == "killed":
        observed = "killed by host timeout" if res.timed_out else f"exit={res.exit_code} out={last[:120]}"
        return observed, res.timed_out

    if res.timed_out:
        return "timed out (unexpected)", False

    if probe.id == "py_identity":
        fields = dict(line.split(" ", 1) for line in out.splitlines() if " " in line)
        ok = (
            fields.get("UID") == "10001"
            and fields.get("EUID") == "10001"
            and fields.get("CAPEFF") == "0000000000000000"
            and fields.get("NONEWPRIVS") == "1"
        )
        return out.replace(chr(10), " | ")[:300], ok
    if probe.id == "js_identity":
        ok = "UID 10001" in out
        return out[:300], ok

    if probe.id == "py_mounts":
        ok = "HOST_MOUNTS 0" in out and " ro," in out + ","
        return out.replace("\n", " | ")[:300], ok

    if probe.id == "py_host_paths":
        ok = "FOUND []" in out
        return out.replace("\n", " | ")[:300], ok

    if probe.id in ("py_memory", "js_memory"):
        # Either the allocation raises inside the runtime, or the container is
        # OOM-killed (exit 137). Both mean the ceiling held.
        ok = (
            "BLOCKED" in out
            or res.oom_killed
            or res.exit_code == 137
        ) and "ALLOCATED" not in out
        return (last or f"oom_killed={res.oom_killed} exit={res.exit_code}")[:300], ok

    if probe.expect == "blocked":
        blocked = "BLOCKED" in out and "WROTE" not in out and "CONNECTED" not in out and "RESOLVED" not in out and "FORKED" not in out
        return (out.replace("\n", " | ")[:300] or f"exit={res.exit_code}"), blocked

    return last[:300], False


def main() -> int:
    ok, detail = runner.docker_available()
    if not ok:
        print(f"docker unavailable: {detail}")
        return 2

    for language, image in runner.IMAGES.items():
        if not runner.image_exists(image):
            print(f"missing image {image}; run runner.py --build first")
            return 2

    docker_server = subprocess.run(
        ["docker", "version", "--format", "{{.Server.Version}}"],
        capture_output=True, text=True, timeout=60,
    ).stdout.strip()

    results: list[ProbeResult] = []
    for probe in PY + JS:
        res = runner.run_command(probe.language, probe.code, timeout_s=probe.timeout_s)
        observed, passed = classify(probe, res)
        results.append(
            ProbeResult(
                id=probe.id, language=probe.language, property=probe.property,
                description=probe.description, expect=probe.expect, observed=observed,
                passed=passed, exit_code=res.exit_code, timed_out=res.timed_out,
                oom_killed=res.oom_killed, duration_s=res.duration_s,
            )
        )
        print(f"[{'PASS' if passed else 'FAIL'}] {probe.id}: {observed[:120]}")

    payload = {
        "docker_server_version": docker_server,
        "images": runner.IMAGES,
        "limits": runner.LIMITS,
        "container_flags": [
            "--network none", "--read-only",
            "--tmpfs /tmp:rw,noexec,nosuid,nodev,size=16m",
            "--user 10001:10001", "--cap-drop ALL",
            "--security-opt no-new-privileges:true",
            f"--memory {runner.LIMITS['memory']} --memory-swap {runner.LIMITS['memory_swap']}",
            f"--cpus {runner.LIMITS['cpus']}", f"--pids-limit {runner.LIMITS['pids']}",
            f"--ulimit nofile={runner.LIMITS['nofile']}",
            f"--ulimit fsize={runner.LIMITS['fsize']}",
            "no bind mounts; payload over stdin, results over stderr",
            "container removed after every run",
        ],
        "probes": [asdict(r) for r in results],
        "passed": sum(r.passed for r in results),
        "total": len(results),
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "phase11_sandbox_security.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Phase 11 — Sandbox Containment Verification",
        "",
        f"Docker server: `{docker_server}` (linux containers). "
        f"Probes: {payload['passed']}/{payload['total']} passed.",
        "",
        "## Container configuration",
        "",
        "```",
    ]
    lines += payload["container_flags"]
    lines += [
        "```",
        "",
        "## Probe results",
        "",
        "| probe | property | expected | observed | verdict |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        obs = r.observed.replace("|", "\\|")[:180]
        lines.append(
            f"| `{r.id}` | {r.property} | {r.expect} | {obs} | {'PASS' if r.passed else 'FAIL'} |"
        )
    lines += [
        "",
        "## Fixed limits",
        "",
        f"- memory: {runner.LIMITS['memory']} (swap disabled at the same value)",
        f"- cpus: {runner.LIMITS['cpus']}",
        f"- pids: {runner.LIMITS['pids']}",
        f"- open files: {runner.LIMITS['nofile']}",
        f"- max file size: {runner.LIMITS['fsize']} bytes",
        f"- wall clock: {runner.DEFAULT_TIMEOUT_S:.0f}s per execution, host-enforced by kill",
        "",
        "## Residual risk",
        "",
        "- The container shares the Docker Desktop Linux VM kernel; a kernel exploit is theoretically exploitable. No seccomp profile beyond Docker's default and no user namespace remapping is configured.",
        "- Docker Desktop itself runs with the host user's privileges; container escape flaws in Docker are out of scope of the app-level sandbox.",
        "- `--tmpfs /tmp` is writable (16 MiB, noexec) by design; probes confirm root and `/opt` are not writable.",
        "- Probes ran before any model-generated code was executed.",
        "",
    ]
    (REPORTS / "phase11_sandbox_security.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"saved reports/phase11_sandbox_security.md ({payload['passed']}/{payload['total']})")
    return 0 if payload["passed"] == payload["total"] else 1


if __name__ == "__main__":
    sys.exit(main())
