"""PolyCodeGPT Phase 6 sampling comparison (read-only, stochastic).

Uses the final checkpoint to generate completions for the same eight prompts as
scripts/generate_samples.py, but with temperature/top-k sampling instead of greedy
decoding. These are QUALITATIVE examples, not formal correctness scores; results
are stochastic and depend on the random seed.

The user runs this directly in their terminal; Freebuff does NOT execute it.

Usage:
    C:/Users/UltraPc/Desktop/LLM FROM SCRATCH/.venv/Scripts/python.exe scripts/generate_samples_sampling.py
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch

from polycodegpt.data.tokenizer_io import EOS_ID, load_tokenizer
from polycodegpt.model import GPT, ModelConfig
from polycodegpt.model.checkpoint import load_checkpoint as load_model_checkpoint
from polycodegpt.training.checkpoint import load_checkpoint as load_training_checkpoint
from polycodegpt.training import rebuild_model, require_cuda, weights_are_tied

DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "polycodegpt_first_pass" / "latest.pt"
TOKENIZER_PATH = PROJECT_ROOT / "tokenizer" / "artifacts" / "polycodegpt_tokenizer_v1_1.json"
MODEL_CONFIG_PATH = PROJECT_ROOT / "configs" / "model.yaml"
REPORT_PATH = PROJECT_ROOT / "reports" / "phase6_generation_samples_sampling.md"

MAX_NEW_TOKENS = 128
TEMPERATURE = 0.8
TOP_K = 50
SEED = 42

# Same eight prompts as scripts/generate_samples.py.
PROMPTS = [
    ("python", "Compute the factorial of n recursively.", "def factorial(n):\n    "),
    ("python", "Return the nth Fibonacci number.", "def fibonacci(n):\n    "),
    ("python", "Check whether a string is a palindrome.", "def is_palindrome(s):\n    "),
    ("python", "Merge two sorted lists into one sorted list.", "def merge_sorted(a, b):\n    "),
    ("javascript", "Compute the factorial of n recursively.", "function factorial(n) {\n  "),
    ("javascript", "Return the nth Fibonacci number.", "function fibonacci(n) {\n  "),
    ("javascript", "Check whether a string is a palindrome.", "function isPalindrome(s) {\n  "),
    ("javascript", "Merge two sorted arrays into one sorted array.", "function mergeSorted(a, b) {\n  "),
]


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def lang_token(lang: str) -> str:
    return "<|lang:python|>" if lang == "python" else "<|lang:javascript|>"


def build_prompt(lang: str, docstring: str, code_prefix: str) -> str:
    return f"{lang_token(lang)}<|docstring|>{docstring}<|code|>{code_prefix}"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="PolyCodeGPT stochastic sampling comparison")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)
    device = require_cuda(args.device)

    model_config = ModelConfig.from_yaml(MODEL_CONFIG_PATH)
    tokenizer = load_tokenizer(TOKENIZER_PATH, vocab_size=model_config.vocab_size)

    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint_path}")

    try:
        payload = load_training_checkpoint(checkpoint_path, map_location="cpu")
        model = rebuild_model(payload, device="cpu")
        step = payload["step"]
    except Exception:
        state_dict, config, meta = load_model_checkpoint(checkpoint_path, device="cpu")
        model = GPT(config)
        model.load_state_dict(state_dict)
        step = meta.get("step", "unknown")

    if not weights_are_tied(model):
        model.lm_head.weight = model.token_embedding.weight
    model = model.to(device).eval()

    if not all(torch.isfinite(p).all().item() for p in model.parameters()):
        raise SystemExit("checkpoint contains non-finite parameters")

    lines = []
    lines.append("# Phase 6 — Qualitative Generation Samples (Stochastic Sampling)\n")
    lines.append("**Status:** Qualitative examples (not formal correctness scores)\n")
    lines.append("**Decoding mode:** stochastic sampling (NOT a correctness benchmark)\n")
    lines.append(f"**Checkpoint:** `{checkpoint_path}` (step {step})\n")
    lines.append(f"**Decoding:** temperature={args.temperature}, top_k={args.top_k}, max_new_tokens={args.max_new_tokens}, seed={args.seed}\n")
    lines.append("**Note:** French and Arabic competence is NOT claimed; those languages were\n")
    lines.append("reserved in the tokenizer but not represented in the training corpus.\n")
    lines.append("---\n")

    for i, (lang, docstring, code_prefix) in enumerate(PROMPTS, start=1):
        set_seed(args.seed)
        prompt_text = build_prompt(lang, docstring, code_prefix)
        ids = tokenizer.encode(prompt_text, add_special_tokens=False).ids
        if not isinstance(ids, (list, tuple)) or not all(isinstance(x, int) for x in ids):
            raise TypeError(f"tokenizer.encode() returned {type(ids).__name__}, not a list of ints")
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)

        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                eos_token_id=EOS_ID,
            )

        generated_ids = out[0, len(ids):].tolist()
        if EOS_ID in generated_ids:
            generated_ids = generated_ids[:generated_ids.index(EOS_ID)]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

        lines.append(f"## Sample {i} — {lang.capitalize()}\n")
        lines.append(f"**Prompt docstring:** {docstring}\n")
        lines.append("**Prompt (serializer format):**\n")
        lines.append("```\n" + prompt_text + "\n```\n")
        lines.append("**Generated output:**\n")
        lines.append("```\n" + code_prefix + generated_text + "\n```\n")
        lines.append("---\n")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Samples written to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
