"""Unit tests for the sandbox container configuration (no container is started).

These assert the *shape* of the docker invocation so a future edit cannot quietly
drop `--network none`, mount a host directory, or run as root.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from functional.sandbox import runner  # noqa: E402


def args_for(language: str = "python") -> list[str]:
    return runner._container_args(runner.IMAGES[language], "ph11-test")


def flag_value(args: list[str], flag: str) -> str | None:
    return args[args.index(flag) + 1] if flag in args else None


def test_two_language_images_defined() -> None:
    assert set(runner.IMAGES) == {"python", "javascript"}


def test_network_is_disabled() -> None:
    assert flag_value(args_for(), "--network") == "none"


def test_root_filesystem_is_read_only_with_ephemeral_tmp() -> None:
    args = args_for()
    assert "--read-only" in args
    tmpfs = flag_value(args, "--tmpfs")
    assert tmpfs and tmpfs.startswith("/tmp:") and "noexec" in tmpfs


def test_container_runs_as_unprivileged_user() -> None:
    assert flag_value(args_for(), "--user") == "10001:10001"
    assert flag_value(args_for(), "--user") != "0:0"


def test_no_capabilities_and_no_privilege_escalation() -> None:
    args = args_for()
    assert flag_value(args, "--cap-drop") == "ALL"
    assert flag_value(args, "--security-opt") == "no-new-privileges:true"


def test_resource_limits_present() -> None:
    args = args_for()
    assert flag_value(args, "--memory") == runner.LIMITS["memory"]
    assert flag_value(args, "--memory-swap") == runner.LIMITS["memory_swap"]
    assert flag_value(args, "--cpus") == runner.LIMITS["cpus"]
    assert flag_value(args, "--pids-limit") == runner.LIMITS["pids"]
    ulimits = [args[i + 1] for i, a in enumerate(args) if a == "--ulimit"]
    assert any(u.startswith("nofile=") for u in ulimits)
    assert any(u.startswith("fsize=") for u in ulimits)


def test_no_host_mounts_and_not_privileged() -> None:
    args = args_for()
    for forbidden in ("-v", "--volume", "--mount", "--privileged", "--pid=host", "--net=host"):
        assert forbidden not in args
    host_markers = (str(PROJECT_ROOT), "C:\\", "/c/Users", "/mnt/c", "PolyCodeGPT")
    assert not [a for a in args if any(marker in a for marker in host_markers)]


def test_image_is_last_argument() -> None:
    for language, image in runner.IMAGES.items():
        args = args_for(language)
        assert args[0] == "docker" and args[1] == "run"
        assert args[-1] == image


def test_memory_ceiling_is_bounded() -> None:
    assert runner.LIMITS["memory"].endswith("m")
    assert int(runner.LIMITS["memory"].rstrip("m")) <= 512
    assert float(runner.LIMITS["cpus"]) <= 1.0


def test_wall_clock_timeout_is_strict() -> None:
    assert 0 < runner.DEFAULT_TIMEOUT_S <= 30


def test_probe_entrypoints_are_direct_interpreters() -> None:
    assert runner.PROBE_ENTRYPOINTS["python"][0] == "python3"
    assert runner.PROBE_ENTRYPOINTS["javascript"][0] == "node"
