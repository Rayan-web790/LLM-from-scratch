import json
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer

ROOT = Path.cwd()

BIN = ROOT / "data/processed/polycodegpt_v2/scenario_b/train.bin"
TOKENIZER_PATH = ROOT / "tokenizer/artifacts/v2/polycodegpt_tokenizer_v2.json"
DIVERGENCE_REPORT = ROOT / "logs/phase20_scenario_b/verified_first_divergence.json"

VOCAB_SIZE = 24000
CHUNK_SIZE = 5_000_000

tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
report = json.loads(DIVERGENCE_REPORT.read_text(encoding="utf-8"))

print("========== FILE VALIDATION ==========")

size = BIN.stat().st_size
print("Binary size:", size)
print("Expected uint16 tokens:", size // 2)

if size % 2 != 0:
    raise RuntimeError("Binary file is not aligned to uint16.")

counts = np.zeros(VOCAB_SIZE, dtype=np.int64)
tokens = np.memmap(BIN, dtype=np.uint16, mode="r")

for start in range(0, len(tokens), CHUNK_SIZE):
    chunk = tokens[start:start + CHUNK_SIZE]

    if int(chunk.max()) >= VOCAB_SIZE:
        raise RuntimeError("Token ID exceeds vocabulary size.")

    counts += np.bincount(
        chunk.astype(np.int64),
        minlength=VOCAB_SIZE,
    )

print("\n========== TOKEN COUNT VALIDATION ==========")
print("Counted tokens:", int(counts.sum()))
print("Nonzero token IDs:", int(np.count_nonzero(counts)))

print("\n========== TOP 40 TRAINING TOKENS ==========")

for rank, token_id in enumerate(
    np.argsort(counts)[::-1][:40], 1
):
    token = tokenizer.id_to_token(int(token_id))
    print(
        f"{rank:2d}. ID={token_id:5d} "
        f"count={counts[token_id]:12,d} "
        f"token={token!r}"
    )

print("\n========== WHITESPACE TOKEN VOCABULARY ==========")

vocab = tokenizer.get_vocab()
whitespace_tokens = []

for token, token_id in vocab.items():
    if "\u010a" in token or "\u0120" in token or "\u0109" in token:
        whitespace_tokens.append((token_id, token, counts[token_id]))

print("Whitespace-related vocabulary entries:", len(whitespace_tokens))

for token_id, token, frequency in sorted(
    whitespace_tokens,
    key=lambda x: x[2],
    reverse=True,
)[:50]:
    print(
        f"ID={token_id:5d} "
        f"count={frequency:12,d} "
        f"token={token!r}"
    )

print("\n========== VERIFIED DIVERGENCE TOKEN IDS ==========")

for item in report["results"]:
    token_id = item.get("expected_token_id")

    if token_id is None:
        continue

    token = tokenizer.id_to_token(token_id)

    print(
        f"{item['language']:10s} "
        f"{item['path']:48s} "
        f"ID={token_id:5d} "
        f"count={counts[token_id]:12,d} "
        f"token={token!r}"
    )

print("\n========== AUDIT COMPLETE ==========")
print("Training data was read only.")
print("No checkpoint or dataset files were modified.")
print("Validation and test partitions were not accessed.")
