from __future__ import annotations

import ast
import difflib
import hashlib
import json
import random
import sys
from collections import defaultdict, Counter
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
MANIFEST = ROOT / "data/processed/polycodegpt_v2/scenario_b/manifest.json"
VALIDATION = ROOT / "data/processed/polycodegpt_v2/scenario_b/valid.jsonl"

OUTPUT_DIR = ROOT / "logs/phase20_scenario_b"
OUTPUT_JSON = OUTPUT_DIR / "real_validation_generation.json"
OUTPUT_MD = ROOT / "reports/phase20_real_validation_generation.md"

EXPECTED_TOKENIZER_SHA = (
    "0e7f466ee516fe34becc638a3451eba29bced5efa5317647e486ab735cc7f120"
)
EXPECTED_MANIFEST_SHA = (
    "f0b84da44a2fc02f5a03f32867515eaeb3daf931d77bdbe48d1fe83036e1cbf8"
)

EXPECTED_PARAMETERS = 20_060_160
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


def collect_candidates(tokenizer):
    candidates = defaultdict(list)
    language_counts = Counter()
    task_counts = Counter()

    with VALIDATION.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)

            language = record.get("language")
            task = record.get("task")

            language_counts[language] += 1
            task_counts[(language, task)] += 1

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
                prompt, add_special_tokens=False
            ).ids

            target_ids = tokenizer.encode(
                target, add_special_tokens=False
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
                "prompt": prompt,
                "target": target,
                "prompt_ids": prompt_ids,
                "target_ids": target_ids,
                "prompt_tokens": len(prompt_ids),
                "target_tokens": len(target_ids),
            })

    print("\n========== DATASET DISTRIBUTION ==========")
    print("Languages:", dict(language_counts))

    print("\nFunction tasks by language:")
    for key, count in sorted(task_counts.items()):
        if key[1] == "function":
            print(f"{key[0]}: {count:,}")

    print("\nEligible candidates:")
    for language in ("python", "javascript"):
        print(f"{language}: {len(candidates[language]):,}")

    rng = random.Random(SEED)
    selected = {}

    for language in ("python", "javascript"):
        pool = candidates[language]

        if len(pool) < SAMPLES_PER_LANGUAGE:
            raise RuntimeError(
                f"Not enough eligible {language} examples: {len(pool)}"
            )

        selected[language] = rng.sample(pool, SAMPLES_PER_LANGUAGE)

    return selected


def main():
    if not VALIDATION.is_file():
        raise FileNotFoundError(VALIDATION)

    if sha256(TOKENIZER_PATH) != EXPECTED_TOKENIZER_SHA:
        raise RuntimeError("Tokenizer SHA-256 mismatch.")

    if sha256(MANIFEST) != EXPECTED_MANIFEST_SHA:
        raise RuntimeError("Manifest SHA-256 mismatch.")

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    if tokenizer.get_vocab_size() != 24000:
        raise RuntimeError("Unexpected tokenizer vocabulary size.")

    print("Tokenizer and manifest integrity: PASSED")

    selected = collect_candidates(tokenizer)

    print("\nLoading Phase 20 best checkpoint...")
    checkpoint = load_checkpoint(CHECKPOINT, map_location="cpu")

    if checkpoint.get("tokenizer_sha256") != EXPECTED_TOKENIZER_SHA:
        raise RuntimeError("Checkpoint tokenizer hash mismatch.")

    if checkpoint.get("data_manifest_sha256") != EXPECTED_MANIFEST_SHA:
        raise RuntimeError("Checkpoint manifest hash mismatch.")

    if checkpoint.get("step") != 30500:
        raise RuntimeError("Unexpected checkpoint step.")

    device = torch.device("cuda")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 is unavailable.")

    model = rebuild_model(checkpoint, device=device)
    model.eval()

    parameter_count = sum(p.numel() for p in model.parameters())

    if parameter_count != EXPECTED_PARAMETERS:
        raise RuntimeError(f"Unexpected parameter count: {parameter_count}")

    if not weights_are_tied(model):
        raise RuntimeError("Model weights are not tied.")

    print("Checkpoint integrity: PASSED")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Parameters: {parameter_count:,}")
    print(f"Checkpoint step: {checkpoint['step']}")

    results = []

    for language in ("python", "javascript"):
        print(f"\n========== {language.upper()} ==========")

        for index, sample in enumerate(selected[language], 1):
            prompt_ids = sample["prompt_ids"]
            target_ids = sample["target_ids"]

            input_tensor = torch.tensor(
                [prompt_ids],
                dtype=torch.long,
                device=device,
            )

            max_new_tokens = len(target_ids)

            with torch.inference_mode():
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                ):
                    output = model.generate(
                        input_tensor,
                        max_new_tokens=max_new_tokens,
                        temperature=0.0,
                        eos_token_id=EOS_ID,
                    )

            generated_ids = output[0, len(prompt_ids):].tolist()

            exact_match = generated_ids == target_ids

            similarity = difflib.SequenceMatcher(
                None,
                target_ids,
                generated_ids,
                autojunk=False,
            ).ratio()

            generated_text = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            )

            target_text = tokenizer.decode(
                target_ids,
                skip_special_tokens=True,
            )

            syntax_valid = None

            if language == "python":
                try:
                    ast.parse(generated_text)
                    syntax_valid = True
                except SyntaxError:
                    syntax_valid = False

            item = {
                "language": language,
                "sample_number": index,
                "record_id": sample["record_id"],
                "path": sample["path"],
                "repo": sample["repo"],
                "prompt": sample["prompt"],
                "reference": target_text,
                "generated": generated_text,
                "prompt_tokens": len(prompt_ids),
                "reference_tokens": len(target_ids),
                "generated_tokens": len(generated_ids),
                "exact_match": exact_match,
                "token_sequence_similarity": similarity,
                "eos_generated": EOS_ID in generated_ids,
                "python_syntax_valid": syntax_valid,
            }

            results.append(item)

            print(
                f"[{index}/{SAMPLES_PER_LANGUAGE}] "
                f"{sample['record_id']} | "
                f"exact={exact_match} | "
                f"similarity={similarity:.3f} | "
                f"generated={len(generated_ids)} | "
                f"reference={len(target_ids)}"
            )

            if syntax_valid is not None:
                print(f"    Python syntax valid: {syntax_valid}")

    summaries = {}

    for language in ("python", "javascript"):
        subset = [r for r in results if r["language"] == language]

        summaries[language] = {
            "samples": len(subset),
            "exact_matches": sum(r["exact_match"] for r in subset),
            "exact_match_rate": sum(r["exact_match"] for r in subset) / len(subset),
            "mean_token_sequence_similarity": sum(
                r["token_sequence_similarity"] for r in subset
            ) / len(subset),
            "eos_rate": sum(r["eos_generated"] for r in subset) / len(subset),
        }

        if language == "python":
            summaries[language]["syntax_valid_count"] = sum(
                r["python_syntax_valid"] for r in subset
            )
            summaries[language]["syntax_valid_rate"] = sum(
                r["python_syntax_valid"] for r in subset
            ) / len(subset)

    report = {
        "phase": 20,
        "evaluation": "real_validation_function_generation",
        "checkpoint_step": int(checkpoint["step"]),
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "tokenizer_sha256": EXPECTED_TOKENIZER_SHA,
        "manifest_sha256": EXPECTED_MANIFEST_SHA,
        "parameter_count": parameter_count,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "precision": "bf16",
        "decoding": "greedy",
        "seed": SEED,
        "samples_per_language": SAMPLES_PER_LANGUAGE,
        "test_partition_opened": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
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
        "# PolyCodeGPT Phase 20 — Real Validation Generation",
        "",
        f"- Checkpoint step: {checkpoint['step']}",
        f"- Device: {device}",
        f"- Precision: BF16",
        f"- Decoding: Greedy",
        f"- Seed: {SEED}",
        f"- Test partition opened: No",
        "",
        "> Exact match and token similarity measure reference reproduction, not semantic correctness.",
        "",
        "## Summary",
        "",
        "| Language | Samples | Exact matches | Mean token similarity | EOS rate |",
        "|---|---:|---:|---:|---:|",
    ]

    for language, summary in summaries.items():
        lines.append(
            f"| {language} | {summary['samples']} | "
            f"{summary['exact_matches']} | "
            f"{summary['mean_token_sequence_similarity']:.3f} | "
            f"{summary['eos_rate']:.3f} |"
        )

    lines.extend(["", "## Individual results", ""])

    for item in results:
        lines.extend([
            f"### {item['language']} — sample {item['sample_number']}",
            "",
            f"- Record: `{item['record_id']}`",
            f"- Source: `{item['path']}`",
            f"- Exact match: {item['exact_match']}",
            f"- Token similarity: {item['token_sequence_similarity']:.3f}",
            f"- Generated/reference tokens: "
            f"{item['generated_tokens']}/{item['reference_tokens']}",
            f"- EOS generated: {item['eos_generated']}",
        ])

        if item["python_syntax_valid"] is not None:
            lines.append(
                f"- Python syntax valid: {item['python_syntax_valid']}"
            )

        lines.extend([
            "",
            "**Generated:**",
            "",
            f"```{item['language']}",
            item["generated"],
            "```",
            "",
            "**Reference:**",
            "",
            f"```{item['language']}",
            item["reference"],
            "```",
            "",
        ])

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print("\n========== EVALUATION COMPLETE ==========")
    print(json.dumps(summaries, indent=2))
    print(f"\nJSON: {OUTPUT_JSON}")
    print(f"Markdown: {OUTPUT_MD}")
    print("Training data and checkpoints were not modified.")
    print("Test partition was not opened.")


if __name__ == "__main__":
    main()
