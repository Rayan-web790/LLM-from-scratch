from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polycodegpt.training.checkpoint import (
    load_checkpoint,
    rebuild_model,
    weights_are_tied,
)

CHECKPOINT = ROOT / "checkpoints/polycodegpt_phase20_scenario_b/best.pt"
TOKENIZER_PATH = ROOT / "tokenizer/artifacts/v2/polycodegpt_tokenizer_v2.json"
TOKENIZER_METADATA = ROOT / "tokenizer/artifacts/v2/tokenizer_metadata_v2.json"
MANIFEST = ROOT / "data/processed/polycodegpt_v2/scenario_b/manifest.json"

OUTPUT_DIR = ROOT / "logs/phase20_scenario_b"
OUTPUT_JSON = OUTPUT_DIR / "generation_samples.json"
OUTPUT_MD = ROOT / "reports/phase20_generation_samples.md"

EXPECTED_TOKENIZER_SHA = (
    "0e7f466ee516fe34becc638a3451eba29bced5efa5317647e486ab735cc7f120"
)
EXPECTED_MANIFEST_SHA = (
    "f0b84da44a2fc02f5a03f32867515eaeb3daf931d77bdbe48d1fe83036e1cbf8"
)
EXPECTED_PARAMETERS = 20_060_160

SPECIAL_IDS = {
    "<|eos|>": 3,
    "<|lang:python|>": 4,
    "<|lang:javascript|>": 5,
    "<|task:function|>": 7,
    "<|docstring|>": 18,
    "<|signature|>": 19,
    "<|code|>": 24,
}


SAMPLES = [
    {
        "name": "python_factorial",
        "language": "python",
        "prompt": (
            "<|bos|><|lang:python|><|task:function|>"
            "<|docstring|>Calculate the factorial of a non-negative integer n."
            "<|signature|>def factorial(n):"
            "<|code|>"
        ),
    },
    {
        "name": "python_is_palindrome",
        "language": "python",
        "prompt": (
            "<|bos|><|lang:python|><|task:function|>"
            "<|docstring|>Return True if the given string is a palindrome, ignoring case."
            "<|signature|>def is_palindrome(text):"
            "<|code|>"
        ),
    },
    {
        "name": "javascript_fibonacci",
        "language": "javascript",
        "prompt": (
            "<|bos|><|lang:javascript|><|task:function|>"
            "<|docstring|>Return the nth Fibonacci number using an iterative approach."
            "<|signature|>function fibonacci(n) {"
            "<|code|>"
        ),
    },
    {
        "name": "javascript_array_sum",
        "language": "javascript",
        "prompt": (
            "<|bos|><|lang:javascript|><|task:function|>"
            "<|docstring|>Return the sum of all numbers in an array."
            "<|signature|>function sumArray(numbers) {"
            "<|code|>"
        ),
    },
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_files():
    for path in (CHECKPOINT, TOKENIZER_PATH, TOKENIZER_METADATA, MANIFEST):
        if not path.is_file():
            raise FileNotFoundError(f"Required file missing: {path}")

    tokenizer_sha = sha256(TOKENIZER_PATH)
    manifest_sha = sha256(MANIFEST)

    if tokenizer_sha != EXPECTED_TOKENIZER_SHA:
        raise RuntimeError("Tokenizer SHA-256 mismatch.")
    if manifest_sha != EXPECTED_MANIFEST_SHA:
        raise RuntimeError("Scenario B manifest SHA-256 mismatch.")

    metadata = json.loads(TOKENIZER_METADATA.read_text(encoding="utf-8"))

    if metadata.get("tokenizer_version") != "2.0":
        raise RuntimeError("Expected tokenizer version 2.0.")
    if metadata.get("artifact_sha256") != tokenizer_sha:
        raise RuntimeError("Tokenizer metadata SHA-256 mismatch.")
    if metadata.get("vocab_size_actual") != 24000:
        raise RuntimeError("Unexpected tokenizer vocabulary size.")

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    if tokenizer.get_vocab_size() != 24000:
        raise RuntimeError("Loaded tokenizer vocabulary is not 24,000.")

    recorded_ids = metadata.get("special_token_ids", {})
    vocab = tokenizer.get_vocab()

    for token, expected_id in recorded_ids.items():
        if vocab.get(token) != int(expected_id):
            raise RuntimeError(f"Special-token ID mismatch: {token}")
        if tokenizer.encode(token, add_special_tokens=False).ids != [int(expected_id)]:
            raise RuntimeError(f"Special token does not encode correctly: {token}")

    for token, expected_id in SPECIAL_IDS.items():
        if recorded_ids.get(token) != expected_id:
            raise RuntimeError(f"Unexpected required token ID: {token}")

    return tokenizer, tokenizer_sha, manifest_sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 1 <= args.max_new_tokens <= 300:
        raise ValueError("--max-new-tokens must be between 1 and 300.")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)

    print("Verifying tokenizer and dataset manifest...")
    tokenizer, tokenizer_sha, manifest_sha = verify_files()

    print("Loading Phase 20 best checkpoint...")
    checkpoint = load_checkpoint(CHECKPOINT, map_location="cpu")

    if checkpoint.get("tokenizer_sha256") != tokenizer_sha:
        raise RuntimeError("Checkpoint/tokenizer hash mismatch.")
    if checkpoint.get("data_manifest_sha256") != manifest_sha:
        raise RuntimeError("Checkpoint/manifest hash mismatch.")
    if int(checkpoint.get("step", -1)) != 30500:
        raise RuntimeError("Unexpected checkpoint step.")
    if checkpoint.get("precision") != "bf16":
        raise RuntimeError("Checkpoint precision metadata is not BF16.")

    device = torch.device(args.device)
    model = rebuild_model(checkpoint, device=device)
    model.eval()

    parameter_count = sum(p.numel() for p in model.parameters())

    if parameter_count != EXPECTED_PARAMETERS:
        raise RuntimeError(
            f"Parameter count mismatch: {parameter_count:,} "
            f"(expected {EXPECTED_PARAMETERS:,})"
        )

    if not weights_are_tied(model):
        raise RuntimeError("Model embedding weights are not tied.")

    if model.config.context_length != 512:
        raise RuntimeError("Unexpected model context length.")

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")

        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("GPU does not support BF16 autocast.")

    print(f"Parameters: {parameter_count:,}")
    print(f"Checkpoint step: {checkpoint['step']}")
    print("Checkpoint integrity: PASSED")
    print("\nStarting deterministic generation...\n")

    results = []

    for sample in SAMPLES:
        encoded = tokenizer.encode(
            sample["prompt"],
            add_special_tokens=False,
        )
        input_ids = encoded.ids

        if not input_ids:
            raise RuntimeError(f"Empty prompt: {sample['name']}")

        if len(input_ids) + args.max_new_tokens > model.config.context_length:
            raise RuntimeError(f"Prompt exceeds context budget: {sample['name']}")

        input_tensor = torch.tensor(
            [input_ids], dtype=torch.long, device=device
        )

        print(f"Generating: {sample['name']} ({len(input_ids)} prompt tokens)")

        if device.type == "cuda":
            with torch.inference_mode():
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                ):
                    generated = model.generate(
                        input_tensor,
                        max_new_tokens=args.max_new_tokens,
                        temperature=0.0,
                        eos_token_id=SPECIAL_IDS["<|eos|>"],
                    )
        else:
            with torch.inference_mode():
                generated = model.generate(
                    input_tensor,
                    max_new_tokens=args.max_new_tokens,
                    temperature=0.0,
                    eos_token_id=SPECIAL_IDS["<|eos|>"],
                )

        new_ids = generated[0, len(input_ids):].tolist()
        raw_output = tokenizer.decode(
            new_ids,
            skip_special_tokens=False,
        )

        decoded = tokenizer.decode(
            new_ids,
            skip_special_tokens=True,
        )

        results.append({
            "name": sample["name"],
            "language": sample["language"],
            "prompt": sample["prompt"],
            "prompt_tokens": len(input_ids),
            "generated_tokens": len(new_ids),
            "raw_output": raw_output,
            "decoded_output": decoded,
            "eos_generated": SPECIAL_IDS["<|eos|>"] in new_ids,
        })

        print(f"Generated {len(new_ids)} tokens.")
        print("-" * 60)
        print(decoded[:700])
        print("-" * 60)

    report = {
        "phase": 20,
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "checkpoint_step": int(checkpoint["step"]),
        "best_validation_loss": float(checkpoint["best_val_loss"]),
        "tokenizer_sha256": tokenizer_sha,
        "manifest_sha256": manifest_sha,
        "parameter_count": parameter_count,
        "weights_tied": weights_are_tied(model),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "precision": "bf16" if device.type == "cuda" else "fp32",
        "decoding": "greedy",
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "test_partition_opened": False,
        "samples": results,
    }

    OUTPUT_JSON.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "# PolyCodeGPT Phase 20 — Generation Samples",
        "",
        f"- Checkpoint step: {checkpoint['step']}",
        f"- Validation subset loss: {checkpoint['best_val_loss']}",
        f"- Parameters: {parameter_count:,}",
        f"- Device: {device}",
        f"- Precision: {report['precision']}",
        "- Decoding: Greedy (temperature 0)",
        "- Test partition opened: No",
        "",
        "> These are qualitative generation samples, not correctness results.",
        "",
    ]

    for item in results:
        lines.extend([
            f"## {item['name']}",
            "",
            f"Language: {item['language']}",
            "",
            "### Prompt",
            "",
            "```text",
            item["prompt"],
            "```",
            "",
            "### Generated output",
            "",
            f"```{item['language']}",
            item["decoded_output"],
            "```",
            "",
            f"Generated tokens: {item['generated_tokens']}",
            "",
        ])

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print("\n========== GENERATION COMPLETE ==========")
    print(f"JSON: {OUTPUT_JSON}")
    print(f"Markdown: {OUTPUT_MD}")
    print("Training artifacts were not modified.")
    print("Test partition was not accessed.")


if __name__ == "__main__":
    main()
