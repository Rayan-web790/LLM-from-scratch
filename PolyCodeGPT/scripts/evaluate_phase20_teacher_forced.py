from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
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
MANIFEST = ROOT / "data/processed/polycodegpt_v2/scenario_b/manifest.json"
VALIDATION = ROOT / "data/processed/polycodegpt_v2/scenario_b/valid.jsonl"

OUTPUT_DIR = ROOT / "logs/phase20_scenario_b"
OUTPUT_JSON = OUTPUT_DIR / "teacher_forced_validation.json"
OUTPUT_MD = ROOT / "reports/phase20_teacher_forced_validation.md"

EXPECTED_TOKENIZER_SHA = (
    "0e7f466ee516fe34becc638a3451eba29bced5efa5317647e486ab735cc7f120"
)
EXPECTED_MANIFEST_SHA = (
    "f0b84da44a2fc02f5a03f32867515eaeb3daf931d77bdbe48d1fe83036e1cbf8"
)

EXPECTED_PARAMETERS = 20_060_160
EXPECTED_STEP = 30500

EOS_ID = 3
SAMPLES_PER_LANGUAGE = 8
SEED = 20261119

MAX_PROMPT_TOKENS = 200
MIN_TARGET_TOKENS = 20
MAX_TARGET_TOKENS = 220
MAX_CONTEXT = 512


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def perplexity(mean_loss: float):
    if mean_loss > 80:
        return None
    return math.exp(mean_loss)


def collect_candidates(tokenizer):
    candidates = defaultdict(list)

    with VALIDATION.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)

            language = record.get("language")
            task = record.get("task")

            if language not in ("python", "javascript"):
                continue

            if task != "function":
                continue

            text = record.get("text", "")
            marker = "<|code|>"

            if marker not in text:
                continue

            pos = text.find(marker)

            prompt = text[:pos + len(marker)]
            target = text[pos + len(marker):]

            prompt_ids = tokenizer.encode(
                prompt,
                add_special_tokens=False,
            ).ids

            target_ids = tokenizer.encode(
                target,
                add_special_tokens=False,
            ).ids

            if not prompt_ids or not target_ids:
                continue

            if len(prompt_ids) > MAX_PROMPT_TOKENS:
                continue

            if not MIN_TARGET_TOKENS <= len(target_ids) <= MAX_TARGET_TOKENS:
                continue

            if len(prompt_ids) + len(target_ids) > MAX_CONTEXT:
                continue

            if target_ids[-1] != EOS_ID:
                continue

            candidates[language].append({
                "record_id": record.get("record_id"),
                "path": record.get("path"),
                "repo": record.get("repo"),
                "prompt_ids": prompt_ids,
                "target_ids": target_ids,
            })

    rng = random.Random(SEED)
    selected = {}

    for language in ("python", "javascript"):
        pool = candidates[language]

        if len(pool) < SAMPLES_PER_LANGUAGE:
            raise RuntimeError(
                f"Not enough eligible {language} examples: {len(pool)}"
            )

        selected[language] = rng.sample(
            pool,
            SAMPLES_PER_LANGUAGE,
        )

    return selected, {
        language: len(items)
        for language, items in candidates.items()
    }


def evaluate_sample(model, sample, device):
    prompt_ids = sample["prompt_ids"]
    target_ids = sample["target_ids"]

    prompt_length = len(prompt_ids)
    target_length = len(target_ids)

    # Complete teacher-forced sequence.
    full_ids = prompt_ids + target_ids

    # Next-token prediction:
    # logits at position p-1 predict target token at position p.
    input_ids = torch.tensor(
        [full_ids[:-1]],
        dtype=torch.long,
        device=device,
    )

    targets = torch.tensor(
        target_ids,
        dtype=torch.long,
        device=device,
    )

    with torch.inference_mode():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            logits, _ = model(input_ids)

        # The first target token is predicted at prompt_length - 1.
        completion_logits = logits[
            0,
            prompt_length - 1:,
            :,
        ].float()

        if completion_logits.shape[0] != target_length:
            raise RuntimeError(
                "Logit/target alignment error: "
                f"{completion_logits.shape[0]} != {target_length}"
            )

        # Cross-entropy for each reference completion token.
        token_losses = F.cross_entropy(
            completion_logits,
            targets,
            reduction="none",
        )

        predictions = completion_logits.argmax(dim=-1)

        correct_tokens = int(
            (predictions == targets).sum().item()
        )

        total_loss = float(token_losses.sum().item())
        mean_loss = total_loss / target_length

    return {
        "total_loss": total_loss,
        "mean_loss": mean_loss,
        "perplexity": perplexity(mean_loss),
        "tokens": target_length,
        "correct_tokens": correct_tokens,
        "token_accuracy": correct_tokens / target_length,
    }


def main():
    print("========== PHASE 20 TEACHER-FORCED EVALUATION ==========")

    for path in (CHECKPOINT, TOKENIZER_PATH, MANIFEST, VALIDATION):
        if not path.is_file():
            raise FileNotFoundError(path)

    if sha256(TOKENIZER_PATH) != EXPECTED_TOKENIZER_SHA:
        raise RuntimeError("Tokenizer SHA-256 mismatch.")

    if sha256(MANIFEST) != EXPECTED_MANIFEST_SHA:
        raise RuntimeError("Manifest SHA-256 mismatch.")

    print("Tokenizer integrity: PASSED")
    print("Manifest integrity: PASSED")

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    if tokenizer.get_vocab_size() != 24000:
        raise RuntimeError("Unexpected tokenizer vocabulary size.")

    selected, candidate_counts = collect_candidates(tokenizer)

    print("\nEligible validation candidates:")
    for language in ("python", "javascript"):
        print(
            f"{language}: {candidate_counts.get(language, 0):,}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 is unavailable.")

    device = torch.device("cuda")

    print("\nLoading Phase 20 best checkpoint...")

    checkpoint = load_checkpoint(
        CHECKPOINT,
        map_location="cpu",
    )

    if checkpoint.get("tokenizer_sha256") != EXPECTED_TOKENIZER_SHA:
        raise RuntimeError("Checkpoint tokenizer hash mismatch.")

    if checkpoint.get("data_manifest_sha256") != EXPECTED_MANIFEST_SHA:
        raise RuntimeError("Checkpoint manifest hash mismatch.")

    if checkpoint.get("step") != EXPECTED_STEP:
        raise RuntimeError("Unexpected checkpoint step.")

    model = rebuild_model(
        checkpoint,
        device=device,
    )

    model.eval()

    parameter_count = sum(
        p.numel() for p in model.parameters()
    )

    if parameter_count != EXPECTED_PARAMETERS:
        raise RuntimeError(
            f"Unexpected parameter count: {parameter_count}"
        )

    if not weights_are_tied(model):
        raise RuntimeError("Model weights are not tied.")

    print("Checkpoint integrity: PASSED")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Parameters: {parameter_count:,}")
    print(f"Checkpoint step: {checkpoint['step']}")
    print("Precision: BF16")

    results = []

    aggregate = {
        language: {
            "total_loss": 0.0,
            "total_tokens": 0,
            "correct_tokens": 0,
            "samples": 0,
        }
        for language in ("python", "javascript")
    }

    torch.cuda.reset_peak_memory_stats(device)

    for language in ("python", "javascript"):
        print(f"\n========== {language.upper()} ==========")

        for index, sample in enumerate(selected[language], 1):
            metrics = evaluate_sample(
                model,
                sample,
                device,
            )

            item = {
                "language": language,
                "sample_number": index,
                "record_id": sample["record_id"],
                "path": sample["path"],
                "repo": sample["repo"],
                "prompt_tokens": len(sample["prompt_ids"]),
                **metrics,
            }

            results.append(item)

            agg = aggregate[language]
            agg["total_loss"] += metrics["total_loss"]
            agg["total_tokens"] += metrics["tokens"]
            agg["correct_tokens"] += metrics["correct_tokens"]
            agg["samples"] += 1

            print(
                f"[{index}/{SAMPLES_PER_LANGUAGE}] "
                f"{sample['record_id']} | "
                f"loss={metrics['mean_loss']:.4f} | "
                f"ppl={metrics['perplexity']:.4f} | "
                f"accuracy={metrics['token_accuracy']:.3f} | "
                f"tokens={metrics['tokens']}"
            )

    summaries = {}

    for language, agg in aggregate.items():
        mean_loss = agg["total_loss"] / agg["total_tokens"]

        summaries[language] = {
            "samples": agg["samples"],
            "completion_tokens": agg["total_tokens"],
            "mean_completion_loss": mean_loss,
            "completion_perplexity": perplexity(mean_loss),
            "token_accuracy": (
                agg["correct_tokens"] / agg["total_tokens"]
            ),
            "correct_tokens": agg["correct_tokens"],
        }

    peak_vram_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    )

    report = {
        "phase": 20,
        "evaluation": "real_validation_teacher_forced_completion",
        "checkpoint_step": int(checkpoint["step"]),
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "tokenizer_sha256": EXPECTED_TOKENIZER_SHA,
        "manifest_sha256": EXPECTED_MANIFEST_SHA,
        "parameter_count": parameter_count,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "precision": "bf16",
        "seed": SEED,
        "samples_per_language": SAMPLES_PER_LANGUAGE,
        "candidate_counts": candidate_counts,
        "scoring": (
            "Completion-only next-token cross-entropy; "
            "prompt targets excluded; EOS included."
        ),
        "test_partition_opened": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "peak_allocated_vram_mb": peak_vram_mb,
        "summaries": summaries,
        "results": results,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_JSON.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "# PolyCodeGPT Phase 20 — Teacher-Forced Validation",
        "",
        f"- Checkpoint step: {checkpoint['step']}",
        f"- GPU: {torch.cuda.get_device_name(device)}",
        "- Precision: BF16",
        f"- Seed: {SEED}",
        f"- Samples per language: {SAMPLES_PER_LANGUAGE}",
        "- Scoring: completion-only next-token cross-entropy",
        "- Prompt tokens excluded from loss: Yes",
        "- EOS included: Yes",
        "- Test partition opened: No",
        "",
        "> Teacher-forced perplexity measures next-token prediction with the correct preceding context. It does not measure free-running generation quality.",
        "",
        "## Summary",
        "",
        "| Language | Samples | Tokens | Mean loss | Perplexity | Token accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for language, summary in summaries.items():
        lines.append(
            f"| {language} | "
            f"{summary['samples']} | "
            f"{summary['completion_tokens']:,} | "
            f"{summary['mean_completion_loss']:.5f} | "
            f"{summary['completion_perplexity']:.5f} | "
            f"{summary['token_accuracy']:.4f} |"
        )

    lines.extend([
        "",
        "## Individual results",
        "",
    ])

    for item in results:
        lines.extend([
            f"### {item['language']} — sample {item['sample_number']}",
            "",
            f"- Record: `{item['record_id']}`",
            f"- Source: `{item['path']}`",
            f"- Completion tokens: {item['tokens']}",
            f"- Mean loss: {item['mean_loss']:.5f}",
            f"- Perplexity: {item['perplexity']:.5f}",
            f"- Token accuracy: {item['token_accuracy']:.4f}",
            "",
        ])

    lines.extend([
        "## Integrity",
        "",
        f"- Tokenizer SHA-256: `{EXPECTED_TOKENIZER_SHA}`",
        f"- Manifest SHA-256: `{EXPECTED_MANIFEST_SHA}`",
        f"- Peak allocated VRAM: {peak_vram_mb:.2f} MB",
        "- Training data modified: No",
        "- Checkpoints modified: No",
        "- Test partition opened: No",
        "",
    ])

    OUTPUT_MD.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("\n========== TEACHER-FORCED SUMMARY ==========")
    print(json.dumps(summaries, indent=2))

    print(f"\nPeak allocated VRAM: {peak_vram_mb:.2f} MB")
    print(f"JSON: {OUTPUT_JSON}")
    print(f"Markdown: {OUTPUT_MD}")

    print("\nTraining data and checkpoints were not modified.")
    print("Test partition was not opened.")
    print("\nEVALUATION COMPLETE")


if __name__ == "__main__":
    main()
