"""Phase 14 final tooling-correction tests.

No model generation or candidate execution is performed. File-writing tests use
temporary directories only.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.evaluate_phase10_benchmark as generation  # noqa: E402
from scripts.evaluate_phase10_benchmark import (  # noqa: E402
    _phase10_generation_tasks,
    load_task_manifest,
    main,
    prepare_output_paths,
)

INDEPENDENT_MANIFEST = PROJECT_ROOT / "functional" / "independent_v3" / "manifest.json"
SCRIPT = PROJECT_ROOT / "scripts" / "evaluate_phase10_benchmark.py"


def _fake_results() -> dict:
    py_id, js_id = "py_v2_01", "js_v2_01"
    return {
        "task_suite": "independent_v3",
        "task_manifest": str(INDEPENDENT_MANIFEST),
        "task_manifest_path": str(INDEPENDENT_MANIFEST),
        "task_manifest_sha256": "a" * 64,
        "task_manifest_hash": "a" * 64,
        "task_ids": [py_id, js_id],
        "checkpoint": "experiments/phase12b/runs/exp_b/step_016260.pt",
        "checkpoint_sha256": "b" * 64,
        "checkpoint_step": 16260,
        "device": "cpu",
        "temperature": 0.0,
        "max_new_tokens": 64,
        "prompts_generated": 2,
        "python_tasks": 1,
        "javascript_tasks": 1,
        "per_task_results": [
            {
                "task_id": py_id,
                "language": "python",
                "docstring": "doc",
                "code_stub": "def f():",
                "generated_text": "    return 1",
                "syntax_candidate": "def f():\n    return 1",
                "syntax_valid": True,
                "eos_hit": False,
                "unique_token_ratio": 1.0,
                "repeated_line_ratio": 0.0,
                "repeated_4gram_ratio": 0.0,
                "generated_token_count": 3,
            },
            {
                "task_id": js_id,
                "language": "javascript",
                "docstring": "doc",
                "code_stub": "function f() {",
                "generated_text": "  return 1;\n}",
                "syntax_candidate": "function f() {\n  return 1;\n}",
                "syntax_valid": True,
                "eos_hit": False,
                "unique_token_ratio": 1.0,
                "repeated_line_ratio": 0.0,
                "repeated_4gram_ratio": 0.0,
                "generated_token_count": 5,
            },
        ],
        "syntax_valid": 2,
        "syntax_valid_rate": 1.0,
        "empty_generations": 0,
        "unique_token_ratio_mean": 1.0,
        "repeated_line_ratio_mean": 0.0,
        "repeated_4gram_ratio_mean": 0.0,
        "eos_hits": 0,
        "eos_hit_rate": 0.0,
    }


def test_generation_help_lists_output_options() -> None:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--json-out" in proc.stdout
    assert "--md-out" in proc.stdout


def test_independent_mode_does_not_import_phase10_tasks() -> None:
    script = f"""
import sys
from pathlib import Path
ROOT = Path(r"{PROJECT_ROOT}")
sys.path.insert(0, str(ROOT))

class BlockPhase10Tasks:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "scripts.phase10_tasks" or fullname.startswith("scripts.phase10_tasks."):
            raise AssertionError("scripts.phase10_tasks was imported")
        return None

sys.meta_path.insert(0, BlockPhase10Tasks())
from scripts.evaluate_phase10_benchmark import load_task_manifest
tasks, suite, manifest_hash = load_task_manifest(
    ROOT / "functional" / "independent_v3" / "manifest.json"
)
assert suite == "independent_v3"
assert len(tasks) == 100
assert sum(t["language"] == "python" for t in tasks) == 50
assert sum(t["language"] == "javascript" for t in tasks) == 50
assert all(t["task_id"] for t in tasks)
assert "scripts.phase10_tasks" not in sys.modules
print("independent mode OK without phase10 import")
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr


def test_default_phase10_path_still_loads_100_tasks() -> None:
    tasks = _phase10_generation_tasks()
    assert len(tasks) == 100
    assert sum(t["language"] == "python" for t in tasks) == 50
    assert sum(t["language"] == "javascript" for t in tasks) == 50
    assert all(t["task_id"] is None for t in tasks)


def test_independent_manifest_retains_all_ids_and_languages() -> None:
    tasks, suite, manifest_hash = load_task_manifest(INDEPENDENT_MANIFEST)
    assert suite == "independent_v3"
    expected = [f"py_v2_{i:02d}" for i in range(1, 51)] + [
        f"js_v2_{i:02d}" for i in range(1, 51)
    ]
    assert [t["task_id"] for t in tasks] == expected
    assert sum(t["language"] == "python" for t in tasks) == 50
    assert sum(t["language"] == "javascript" for t in tasks) == 50
    assert len(manifest_hash) == 64


def test_explicit_paths_are_resolved_and_parents_created(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(generation, "project_root", tmp_path)
    json_path, md_path, explicit = prepare_output_paths(
        Path("nested") / "out.json", Path("nested") / "out.md", "TS"
    )
    assert explicit is True
    assert json_path == tmp_path / "nested" / "out.json"
    assert md_path == tmp_path / "nested" / "out.md"
    assert json_path.parent.is_dir()
    assert md_path.parent.is_dir()


def test_one_explicit_path_derives_sibling(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(generation, "project_root", tmp_path)
    json_path, md_path, explicit = prepare_output_paths(
        Path("nested") / "out.json", None, "TS"
    )
    assert explicit is True
    assert json_path == tmp_path / "nested" / "out.json"
    assert md_path == tmp_path / "nested" / "out.md"

    json_path2, md_path2, _ = prepare_output_paths(
        None, Path("other") / "summary.md", "TS"
    )
    assert json_path2 == tmp_path / "other" / "summary.json"
    assert md_path2 == tmp_path / "other" / "summary.md"


def test_existing_json_is_rejected(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(generation, "project_root", tmp_path)
    json_path = tmp_path / "out.json"
    json_path.write_text("sentinel", encoding="utf-8")
    with pytest.raises(SystemExit, match="refusing to overwrite existing JSON output"):
        prepare_output_paths(json_path, tmp_path / "out.md", "TS")
    assert json_path.read_text(encoding="utf-8") == "sentinel"
    assert not (tmp_path / "out.md").exists()


def test_existing_markdown_is_rejected(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(generation, "project_root", tmp_path)
    md_path = tmp_path / "out.md"
    md_path.write_text("sentinel", encoding="utf-8")
    with pytest.raises(SystemExit, match="refusing to overwrite existing Markdown output"):
        prepare_output_paths(tmp_path / "out.json", md_path, "TS")
    assert md_path.read_text(encoding="utf-8") == "sentinel"
    assert not (tmp_path / "out.json").exists()


def test_default_paths_preserve_phase10_names_and_overwrite_behavior(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(generation, "project_root", tmp_path)
    json_path, md_path, explicit = prepare_output_paths(
        None, None, "20260101_000000"
    )
    assert explicit is False
    assert json_path == tmp_path / "reports" / "phase10_generation_evaluation_20260101_000000.json"
    assert md_path == tmp_path / "reports" / "phase10_evaluation_summary.md"

    # Default mode intentionally retains the historical overwrite behavior.
    md_path.write_text("existing", encoding="utf-8")
    json_path2, md_path2, explicit2 = prepare_output_paths(
        None, None, "20260101_000000"
    )
    assert explicit2 is False
    assert json_path2 == json_path
    assert md_path2 == md_path


def test_main_honors_explicit_paths_and_provenance(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(generation, "project_root", tmp_path)
    calls = []

    def fake_run_benchmark(**kwargs):
        calls.append(kwargs)
        return _fake_results()

    monkeypatch.setattr(generation, "run_benchmark", fake_run_benchmark)
    json_path = tmp_path / "new" / "candidate.json"
    md_path = tmp_path / "new" / "candidate.md"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_phase10_benchmark.py",
            "--json-out",
            str(json_path),
            "--md-out",
            str(md_path),
            "--task-manifest",
            str(INDEPENDENT_MANIFEST),
        ],
    )

    main()

    assert calls and calls[0]["task_manifest"] == INDEPENDENT_MANIFEST
    assert json_path.is_file()
    assert md_path.is_file()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["task_ids"] == ["py_v2_01", "js_v2_01"]
    assert payload["task_manifest_sha256"] == "a" * 64
    assert payload["checkpoint_sha256"] == "b" * 64
    assert payload["task_suite"] == "independent_v3"
    assert payload["seed"] == 42
    assert payload["temperature"] == 0.0
    assert payload["max_new_tokens"] == 64
    assert payload["python_tasks"] == 1
    assert payload["javascript_tasks"] == 1
    assert "Phase 10 Evaluation Summary" in md_path.read_text(encoding="utf-8")


def test_main_refuses_to_overwrite_before_generation(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(generation, "project_root", tmp_path)
    calls = []

    def fake_run_benchmark(**kwargs):
        calls.append(kwargs)
        return _fake_results()

    monkeypatch.setattr(generation, "run_benchmark", fake_run_benchmark)
    json_path = tmp_path / "candidate.json"
    json_path.write_text("sentinel", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_phase10_benchmark.py",
            "--json-out",
            str(json_path),
            "--md-out",
            str(tmp_path / "candidate.md"),
        ],
    )

    with pytest.raises(SystemExit, match="refusing to overwrite existing JSON output"):
        main()
    assert calls == []
    assert json_path.read_text(encoding="utf-8") == "sentinel"
    assert not (tmp_path / "candidate.md").exists()
