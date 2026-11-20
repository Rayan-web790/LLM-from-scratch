"""Integration test for Phase 10 benchmark orchestration.

Exercises the real run_benchmark() orchestration with a mock model
to verify 100 generation calls occur (50 Python + 50 JavaScript),
each receiving a correctly encoded generation prefix without EOS.
No model inference is performed; generated code is never executed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import torch

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import benchmark module
from scripts.evaluate_phase10_benchmark import (
    run_benchmark,
    build_generation_prefix,
)
from scripts.phase10_tasks import python_tasks, javascript_tasks, task_counts
from src.polycodegpt.data.tokenizer_io import load_tokenizer


class MockModel:
    """Test double that returns a fixed generation and records calls."""

    def __init__(self):
        self.call_count = 0
        self.received_prompt_ids = []

    def generate(self, input_ids, max_new_tokens=256, temperature=0.8, eos_token_id=2):
        self.call_count += 1
        self.received_prompt_ids.append(input_ids[0].tolist())
        out_ids = input_ids[0].tolist() + [10, 11, 12]
        return torch.tensor([out_ids], dtype=torch.long, device=input_ids.device)

    def eval(self):
        return self

    def to(self, device):
        return self

    def parameters(self):
        return []

    @property
    def config(self):
        class Config:
            vocab_size = 1000
        return Config()


@pytest.fixture
def real_tokenizer():
    return load_tokenizer("tokenizer/artifacts/polycodegpt_tokenizer_v1_1.json")


def test_run_benchmark_integration_100_calls(real_tokenizer):
    """Verify run_benchmark makes exactly 100 generation calls with correct prefixes."""
    mock_model = MockModel()

    with patch("scripts.evaluate_phase10_benchmark.load_model_and_tokenizer") as mock_load, \
         patch("scripts.evaluate_phase10_benchmark.generate_continuation") as mock_gen:

        mock_load.return_value = (mock_model, real_tokenizer, 8130)

        def mock_generate(model, device, prompt_ids, max_new_tokens=256, temperature=0.8, eos_id=2):
            mock_model.call_count += 1
            mock_model.received_prompt_ids.append(list(prompt_ids))
            # Return EOS for every 3rd call to test EOS hit counting
            if mock_model.call_count % 3 == 0:
                return [10, 11, 2], True  # EOS hit
            return [10, 11, 12], False  # No EOS

        mock_gen.side_effect = mock_generate

        results = run_benchmark(device="cpu", max_new_tokens=4, temperature=0.0)

    assert mock_gen.call_count == 100, f"Expected 100 generation calls, got {mock_gen.call_count}"
    assert results["python_tasks"] == 50, f"Expected 50 Python tasks, got {results['python_tasks']}"
    assert results["javascript_tasks"] == 50, f"Expected 50 JavaScript tasks, got {results['javascript_tasks']}"
    assert results["prompts_generated"] == 100, f"Expected 100 prompts, got {results['prompts_generated']}"

    assert len(mock_model.received_prompt_ids) == 100
    for i, prompt_ids in enumerate(mock_model.received_prompt_ids):
        assert 2 not in prompt_ids, f"Prompt {i} contains EOS token"
        assert 3 in prompt_ids, f"Prompt {i} missing code marker"
        assert 4 in prompt_ids, f"Prompt {i} missing docstring marker"

    # Verify EOS hits counted correctly (every 3rd call = ~33 EOS hits)
    expected_eos = 100 // 3
    assert results["eos_hits"] == expected_eos, f"Expected {expected_eos} EOS hits, got {results['eos_hits']}"
    assert results["eos_hit_rate"] == expected_eos / 100.0

    # Verify per-task results exist
    assert "per_task_results" in results
    assert len(results["per_task_results"]) == 100
    for r in results["per_task_results"]:
        assert "syntax_candidate" in r
        assert "eos_hit" in r
        assert "repeated_4gram_ratio" in r

    print(f"Integration test passed: {mock_gen.call_count} generation calls, "
          f"{results['python_tasks']} Python + {results['javascript_tasks']} JavaScript")


def test_task_inventory_persistent():
    counts = task_counts()
    assert counts["python"] == 50
    assert counts["javascript"] == 50
    assert counts["total"] == 100

    py = python_tasks()
    js = javascript_tasks()
    assert len(py) == 50
    assert len(js) == 50


def test_no_held_out_data_access():
    """Verify no held-out test data is accessed."""
    import inspect
    source = inspect.getsource(run_benchmark)
    lines = source.split("\n")
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        assert "valid_tasks.jsonl" not in stripped, f"Held-out data reference in code: {stripped}"
        assert "test.bin" not in stripped, f"test.bin reference in code: {stripped}"
        assert "valid.bin" not in stripped, f"valid.bin reference in code: {stripped}"


def test_no_code_execution():
    """Verify generated code is never executed."""
    import inspect
    source = inspect.getsource(run_benchmark)
    assert "exec(" not in source, "run_benchmark uses exec()"
    assert "subprocess" not in source, "run_benchmark uses subprocess"
    assert "eval(" not in source, "run_benchmark uses eval()"
    assert "os.system" not in source, "run_benchmark uses os.system"


def test_generation_prefix_no_eos():
    """Verify all 100 generation prefixes have no EOS."""
    for docstring, code_stub in python_tasks():
        prefix = build_generation_prefix("python", docstring, code_stub)
        assert "<|eos|>" not in prefix, f"Python prefix contains EOS: {prefix[:60]}"

    for docstring, code_stub in javascript_tasks():
        prefix = build_generation_prefix("javascript", docstring, code_stub)
        assert "<|eos|>" not in prefix, f"JS prefix contains EOS: {prefix[:60]}"


def test_syntax_validation_receives_combined_candidate(real_tokenizer):
    """Production validates code_stub + generated continuation, not continuation alone."""
    mock_model = MockModel()
    captured = []

    def fake_syntax_validity(gen_text, language):
        captured.append((gen_text, language))
        return True

    with patch("scripts.evaluate_phase10_benchmark.load_model_and_tokenizer") as mock_load, \
         patch("scripts.evaluate_phase10_benchmark.generate_continuation") as mock_gen, \
         patch("scripts.evaluate_phase10_benchmark.compute_syntax_validity", side_effect=fake_syntax_validity):

        mock_load.return_value = (mock_model, real_tokenizer, 8130)

        def mock_generate(model, device, prompt_ids, max_new_tokens=256, temperature=0.8, eos_id=2):
            return [10, 11, 12], False

        mock_gen.side_effect = mock_generate

        results = run_benchmark(device="cpu", max_new_tokens=4, temperature=0.0)

    assert len(captured) == 100
    # First Python task: combined candidate must equal stub + decoded continuation
    first_py_stub = python_tasks()[0][1]
    generated_text = real_tokenizer.decode([10, 11, 12], skip_special_tokens=False)
    expected_first = first_py_stub + generated_text
    assert captured[0][0] == expected_first, (
        f"Expected combined candidate {expected_first!r}, got {captured[0][0]!r}"
    )
    assert captured[0][1] == "python"
    # A later JavaScript task is also combined
    js_index = 50
    js_stub = javascript_tasks()[0][1]
    assert captured[js_index][0] == js_stub + generated_text
    assert captured[js_index][1] == "javascript"