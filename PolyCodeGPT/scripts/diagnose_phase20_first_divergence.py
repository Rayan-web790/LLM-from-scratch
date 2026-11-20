from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

from polycodegpt.training.checkpoint import load_checkpoint, rebuild_model

CHECKPOINT = ROOT / "checkpoints/polycodegpt_phase20_scenario_b/best.pt"
TOKENIZER_PATH = ROOT / "tokenizer/artifacts/v2/polycodegpt_tokenizer_v2.json"
VALIDATION = ROOT / "data/processed/polycodegpt_v2/scenario_b/valid.jsonl"
GENERATION = ROOT / "logs/phase20_scenario_b/real_validation_generation.json"

EOS_ID = 3

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

print("Loading validation examples and generation report...")

generation = json.loads(GENERATION.read_text(encoding="utf-8"))
tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

records = {}
with VALIDATION.open("r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)
        records[record.get("record_id")] = record

checkpoint = load_checkpoint(CHECKPOINT, map_location="cpu")

if checkpoint.get("step") != 30500:
    raise RuntimeError("Unexpected checkpoint step.")

device = torch.device("cuda")
model = rebuild_model(checkpoint, device=device)
model.eval()

print("Checkpoint:", checkpoint["step"])
print("GPU:", torch.cuda.get_device_name(device))

print("\n========== FIRST-DIVERGENCE LOGIT ANALYSIS ==========")

for item in generation["results"]:
    record = records.get(item["record_id"])

    if record is None:
        print("MISSING RECORD:", item["record_id"])
        continue

    text = record["text"]
    marker = "<|code|>"
    position = text.find(marker)

    if position < 0:
        continue

    prompt = text[:position + len(marker)]
    reference = text[position + len(marker):]

    prompt_ids = tokenizer.encode(
        prompt, add_special_tokens=False
    ).ids

    reference_ids = tokenizer.encode(
        reference, add_special_tokens=False
    ).ids

    generated_ids = tokenizer.encode(
        item["generated"], add_special_tokens=False
    ).ids

    divergence = next(
        (
            i for i, (a, b) in enumerate(
                zip(reference_ids, generated_ids)
            )
            if a != b
        ),
        min(len(reference_ids), len(generated_ids)),
    )

    if divergence >= len(reference_ids):
        print("\n", item["language"], item["path"])
        print("No reference-token divergence within reference length.")
        continue

    prefix = prompt_ids + reference_ids[:divergence]

    input_ids = torch.tensor(
        [prefix], dtype=torch.long, device=device
    )

    with torch.inference_mode():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            logits, _ = model(input_ids)

    next_logits = logits[0, -1].float()
    probabilities = F.softmax(next_logits, dim=-1)

    top_probs, top_ids = torch.topk(probabilities, k=10)

    expected_id = reference_ids[divergence]
    expected_probability = probabilities[expected_id].item()

    expected_rank = int(
        (probabilities > probabilities[expected_id]).sum().item()
    ) + 1

    generated_id = generated_ids[divergence] if divergence < len(generated_ids) else None

    print(f"\n{item['language'].upper()} | {item['path']}")
    print("First divergent token:", divergence)
    print("Expected token:", repr(tokenizer.id_to_token(expected_id)))
    print("Generated token:", repr(
        tokenizer.id_to_token(generated_id)
        if generated_id is not None else None
    ))
    print(f"Expected token rank: {expected_rank}")
    print(f"Expected token probability: {expected_probability:.6f}")

    print("Top 10 predictions:")
    for rank, (token_id, probability) in enumerate(
        zip(top_ids.tolist(), top_probs.tolist()), 1
    ):
        token = tokenizer.id_to_token(token_id)
        marker = " <-- EXPECTED" if token_id == expected_id else ""
        print(
            f"  {rank:2d}. {repr(token):35s} "
            f"{probability:.6f}{marker}"
        )

print("\nDIAGNOSTIC COMPLETE")
print("No training or checkpoint changes were made.")
print("Test partition was not opened.")
