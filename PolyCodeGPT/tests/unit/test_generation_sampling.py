"""Lightweight tests for the Phase 6 sampling comparison script.

No GPU generation is performed; only argument handling and seeding are checked.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

SCRIPT_PATH = PROJECT_ROOT / "scripts" / "generate_samples_sampling.py"


def load_module():
    spec = importlib.util.spec_from_file_location("generate_samples_sampling", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sampling_script_args_defaults():
    mod = load_module()
    args = mod.parse_args([])
    assert args.temperature == 0.8
    assert args.top_k == 50
    assert args.max_new_tokens == 128
    assert args.seed == 42
    assert args.device == "cuda"
    assert len(mod.PROMPTS) == 8


def test_sampling_script_seeding_deterministic():
    mod = load_module()
    mod.set_seed(42)
    a = torch.randn(4)
    mod.set_seed(42)
    b = torch.randn(4)
    assert torch.equal(a, b)


def test_greedy_script_unchanged_decoding():
    greedy_path = PROJECT_ROOT / "scripts" / "generate_samples.py"
    text = greedy_path.read_text(encoding="utf-8")
    # Greedy script must still use temperature=0 and remain separate.
    assert "temperature=0" in text
    assert "phase6_generation_samples.md" in text
    assert "phase6_generation_samples_sampling.md" not in text
