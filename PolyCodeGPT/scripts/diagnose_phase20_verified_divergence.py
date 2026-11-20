from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import defaultdict, Counter
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

OUTPUT = ROOT / "logs/phase20_scenario_b/verified_first_divergence.json"

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


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_candidates(tokenizer):
    candidates = defaultdict(list)

    with VALIDATION.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)

            language = record.get("language")
            if language not in ("python", "javascript"):
                continue

            if record.get("task") != "function":
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
                "prompt_ids": prompt_ids,
                "target_ids": target_ids,
            })

    rng = random.Random(SEED)
    selected = {}

    for language in ("python", "javascript"):
        pool = candidates[language]

        if len(pool) < SAMPLES_PER_LANGUAGE:
            raise RuntimeError(f"Insufficient {language} candidates")

        selected[language] = rng.sample(pool, SAMPLES_PER_LANGUAGE)

    return selected


def main():
    print("========== INTEGRITY CHECKS ==========")

    if sha256(TOKENIZER_PATH) != EXPECTED_TOKENIZER_SHA:
        raise RuntimeError("Tokenizer SHA mismatch")

    if sha256(MANIFEST) != EXPECTED_MANIFEST_SHA:
        raise RuntimeError("Manifest SHA mismatch")

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    if tokenizer.get_vocab_size() != 24000:
        raise RuntimeError("Unexpected vocabulary size")

    selected = collect_candidates(tokenizer)

    checkpoint = load_checkpoint(CHECKPOINT, map_location="cpu")

    if checkpoint.get("step") != 30500:
        raise RuntimeError("Unexpected checkpoint step")

    if checkpoint.get("tokenizer_sha256") != EXPECTED_TOKENIZER_SHA:
        raise RuntimeError("Checkpoint tokenizer hash mismatch")

    if checkpoint.get("data_manifest_sha256") != EXPECTED_MANIFEST_SHA:
        raise RuntimeError("Checkpoint manifest hash mismatch")

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 unavailable")

    device = torch.device("cuda")
    model = rebuild_model(checkpoint, device=device)
    model.eval()

    parameter_count = sum(p.numel() for p in model.parameters())

    if parameter_count != EXPECTED_PARAMETERS:
        raise RuntimeError("Parameter count mismatch")

    if not weights_are_tied(model):
        raise RuntimeError("Weights are not tied")

    print("Integrity checks: PASSED")
    print("Checkpoint step:", checkpoint["step"])
    print("GPU:", torch.cuda.get_device_name(device))
    print("Parameters:", parameter_count)

    results = []

    print("\n========== VERIFIED FIRST DIVERGENCE ==========")

    for language in ("python", "javascript"):
        for index, sample in enumerate(selected[language], 1):
            prompt_ids = sample["prompt_ids"]
            target_ids = sample["target_ids"]

            input_tensor = torch.tensor(
                [prompt_ids],
                dtype=torch.long,
                device=device,
            )

            with torch.inference_mode():
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                ):
                    output = model.generate(
                        input_tensor,
                        max_new_tokens=len(target_ids),
                        temperature=0.0,
                        eos_token_id=EOS_ID,
                    )

            # Preserve the ORIGINAL generated token IDs.
            generated_ids = output[0, len(prompt_ids):].tolist()

            divergence = next(
                (
                    i for i, (expected, actual) in enumerate(
                        zip(target_ids, generated_ids)
                    )
                    if expected != actual
                ),
                min(len(target_ids), len(generated_ids)),
            )

            exact_match = generated_ids == target_ids

            result = {
                "language": language,
                "sample_number": index,
                "record_id": sample["record_id"],
                "path": sample["path"],
                "reference_tokens": len(target_ids),
                "generated_tokens": len(generated_ids),
                "exact_match": exact_match,
                "first_divergence": divergence,
                "expected_token": None,
                "generated_token": None,
                "expected_rank": None,
                "expected_probability": None,
                "top_10": [],
            }

            if divergence < len(target_ids):
                expected_id = target_ids[divergence]

                prefix = prompt_ids + target_ids[:divergence]

                prefix_tensor = torch.tensor(
                    [prefix],
                    dtype=torch.long,
                    device=device,
                )

                with torch.inference_mode():
                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.bfloat16,
                    ):
                        logits, _ = model(prefix_tensor)

                next_logits = logits[0, -1].float()
                probabilities = F.softmax(next_logits, dim=-1)

                top_probs, top_ids = torch.topk(probabilities, k=10)

                expected_probability = probabilities[expected_id].item()

                expected_rank = int(
                    (probabilities > probabilities[expected_id]).sum().item()
                ) + 1

                generated_id = (
                    generated_ids[divergence]
                    if divergence < len(generated_ids)
                    else None
                )

                result.update({
                    "expected_token": tokenizer.id_to_token(expected_id),
                    "generated_token": (
                        tokenizer.id_to_token(generated_id)
                        if generated_id is not None else None
                    ),
                    "expected_token_id": expected_id,
                    "generated_token_id": generated_id,
                    "expected_rank": expected_rank,
                    "expected_probability": expected_probability,
                    "top_10": [
                        {
                            "rank": rank,
                            "token_id": token_id,
                            "token": tokenizer.id_to_token(token_id),
                            "probability": probability,
                            "is_expected": token_id == expected_id,
                        }
                        for rank, (token_id, probability) in enumerate(
                            zip(top_ids.tolist(), top_probs.tolist()), 1
                        )
                    ],
                })

            results.append(result)

            print(
                f"\n{language.upper()} #{index} | {sample['path']}"
            )
            print("Record:", sample["record_id"])
            print("Exact match:", exact_match)
            print("Reference/generated:", len(target_ids), "/", len(generated_ids))
            print("First divergence:", divergence)

            if result["expected_rank"] is not None:
                print("Expected token:", repr(result["expected_token"]))
                print("Generated token:", repr(result["generated_token"]))
                print("Expected rank:", result["expected_rank"])
                print(
                    "Expected probability:",
                    f"{result['expected_probability']:.6f}",
                )

                print("Top predictions:")
                for candidate in result["top_10"]:
                    marker = " <-- EXPECTED" if candidate["is_expected"] else ""
                    print(
                        f"  {candidate['rank']:2d}. "
                        f"{candidate['token']!r:35s} "
                        f"{candidate['probability']:.6f}{marker}"
                    )

    report = {
        "evaluation": "verified_first_divergence_original_token_ids",
        "checkpoint_step": checkpoint["step"],
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "tokenizer_sha256": EXPECTED_TOKENIZER_SHA,
        "manifest_sha256": EXPECTED_MANIFEST_SHA,
        "seed": SEED,
        "samples_per_language": SAMPLES_PER_LANGUAGE,
        "decoding": "greedy",
        "test_partition_opened": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n========== COMPLETE ==========")
    print("Report:", OUTPUT)
    print("No training or checkpoint changes.")
    print("Test partition was not opened.")


if __name__ == "__main__":
    main()
