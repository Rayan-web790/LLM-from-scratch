import json
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
from tokenizers import Tokenizer

ROOT = Path.cwd()
DATA = ROOT / "data/processed/polycodegpt_v2/scenario_b"
INPUT = DATA / "train.jsonl"
TOKENIZER_PATH = ROOT / "tokenizer/artifacts/v2/polycodegpt_tokenizer_v2.json"
OUTPUT = ROOT / "logs/phase20_scenario_b/language_token_frequency_audit.json"

VOCAB_SIZE = 24000
BATCH_SIZE = 128
PROGRESS_EVERY = 10000
CODE_MARKER = "<|code|>"

# Expected tokens from the verified first-divergence diagnostic.
EXPECTED_IDS = {
    301: "Python indentation: 7 spaces",
    300: "Indentation: 3 spaces",
    14079: "Union",
    20659: "blacklist",
    1565: "context",
    664: "el",
    332: "JavaScript indentation: 11 spaces",
    359: "Indentation: 1 space",
    394: "if",
    2507: "logger",
    346: "opening parenthesis",
}

tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
vocab = tokenizer.get_vocab()

def new_counter():
    return np.zeros(VOCAB_SIZE, dtype=np.int64)

def add_counts(destination, ids):
    if ids:
        destination += np.bincount(
            np.asarray(ids, dtype=np.int64),
            minlength=VOCAB_SIZE,
        )

def top_tokens(counts, limit=30):
    result = []
    for token_id in np.argsort(counts)[::-1][:limit]:
        frequency = int(counts[token_id])
        if frequency == 0:
            continue
        result.append({
            "id": int(token_id),
            "token": tokenizer.id_to_token(int(token_id)),
            "count": frequency,
        })
    return result

print("========== STARTING LANGUAGE AUDIT ==========")
print("Input:", INPUT)
print("Batch size:", BATCH_SIZE)

overall = new_counter()
by_language = defaultdict(new_counter)
by_section = defaultdict(new_counter)
by_category = defaultdict(new_counter)

record_counts = defaultdict(int)
metadata_tokens = defaultdict(int)
actual_tokens = defaultdict(int)
missing_code_marker = defaultdict(int)

expected_counts = defaultdict(lambda: defaultdict(int))
expected_section_counts = defaultdict(lambda: defaultdict(int))

records = []
started = time.time()
batch = []

def process_batch(items):
    if not items:
        return

    texts = []
    descriptors = []

    for record in items:
        text = record["text"]
        language = record["language"]
        category = record.get("category", "unknown")

        if CODE_MARKER in text:
            context, code = text.split(CODE_MARKER, 1)
            context += CODE_MARKER
        else:
            context, code = text, ""
            missing_code_marker[language] += 1

        texts.append(context)
        descriptors.append((language, category, "context"))

        texts.append(code)
        descriptors.append((language, category, "code"))

    encodings = tokenizer.encode_batch(
        texts,
        add_special_tokens=False,
    )

    for encoding, descriptor in zip(encodings, descriptors):
        language, category, section = descriptor
        ids = encoding.ids

        add_counts(by_language[language], ids)
        add_counts(by_section[(language, section)], ids)
        add_counts(by_category[(language, category)], ids)
        add_counts(overall, ids)

        actual_tokens[language] += len(ids)

        for token_id in EXPECTED_IDS:
            expected_counts[language][token_id] += ids.count(token_id)
            expected_section_counts[(language, section)][token_id] += ids.count(token_id)

    for record in items:
        language = record["language"]
        category = record.get("category", "unknown")

        record_counts[language] += 1
        metadata_tokens[language] += int(record.get("token_count", 0))
        records.append((language, category))

with INPUT.open("r", encoding="utf-8") as f:
    for line_number, line in enumerate(f, 1):
        record = json.loads(line)
        batch.append(record)

        if len(batch) >= BATCH_SIZE:
            process_batch(batch)
            batch.clear()

        if line_number % PROGRESS_EVERY == 0:
            elapsed = time.time() - started
            print(
                f"Processed {line_number:,} records | "
                f"{elapsed:.1f}s elapsed"
            )

if batch:
    process_batch(batch)

elapsed = time.time() - started

print("\n========== VALIDATION ==========")
print("Total records:", sum(record_counts.values()))
print("Total actual tokens:", int(overall.sum()))
print("Elapsed seconds:", round(elapsed, 2))

print("\n========== LANGUAGE TOTALS ==========")

for language in sorted(record_counts):
    print(
        f"{language:12s} "
        f"records={record_counts[language]:9,d} "
        f"metadata_tokens={metadata_tokens[language]:12,d} "
        f"actual_tokens={actual_tokens[language]:12,d} "
        f"missing_code_marker={missing_code_marker[language]:6,d}"
    )

print("\n========== TOP TOKENS BY LANGUAGE ==========")

for language in sorted(by_language):
    print(f"\n--- {language.upper()} ---")
    for item in top_tokens(by_language[language], 30):
        print(
            f"ID={item['id']:5d} "
            f"count={item['count']:12,d} "
            f"token={item['token']!r}"
        )

print("\n========== CODE VS CONTEXT ==========")

for language in sorted(record_counts):
    for section in ("context", "code"):
        counts = by_section[(language, section)]
        print(
            f"{language:12s} {section:8s} "
            f"tokens={int(counts.sum()):12,d}"
        )

        for item in top_tokens(counts, 15):
            print(
                f"    ID={item['id']:5d} "
                f"count={item['count']:12,d} "
                f"token={item['token']!r}"
            )

print("\n========== CATEGORY TOTALS ==========")

for (language, category), counts in sorted(by_category.items()):
    print(
        f"{language:12s} {category:15s} "
        f"tokens={int(counts.sum()):12,d}"
    )

print("\n========== EXPECTED DIVERGENCE TOKENS ==========")

for language in sorted(expected_counts):
    print(f"\n--- {language.upper()} ---")

    for token_id, description in EXPECTED_IDS.items():
        count = expected_counts[language][token_id]
        if count:
            print(
                f"ID={token_id:5d} "
                f"count={count:12,d} "
                f"{description}"
            )

print("\n========== DIVERGENCE TOKENS BY SECTION ==========")

for language in sorted(record_counts):
    for section in ("context", "code"):
        values = expected_section_counts[(language, section)]
        nonzero = [
            (token_id, count)
            for token_id, count in values.items()
            if count
        ]

        if nonzero:
            print(f"\n{language} / {section}")
            for token_id, count in sorted(nonzero):
                print(
                    f"ID={token_id:5d} "
                    f"count={count:12,d} "
                    f"{EXPECTED_IDS[token_id]}"
                )

print("\n========== SAVING REPORT ==========")

report = {
    "records": dict(record_counts),
    "metadata_tokens": dict(metadata_tokens),
    "actual_tokens": dict(actual_tokens),
    "missing_code_marker": dict(missing_code_marker),
    "total_records": sum(record_counts.values()),
    "total_actual_tokens": int(overall.sum()),
    "elapsed_seconds": elapsed,
    "top_tokens_by_language": {
        lang: top_tokens(counts, 100)
        for lang, counts in by_language.items()
    },
    "top_tokens_by_section": {
        f"{lang}|{section}": top_tokens(counts, 100)
        for (lang, section), counts in by_section.items()
    },
    "category_tokens": {
        f"{lang}|{category}": int(counts.sum())
        for (lang, category), counts in by_category.items()
    },
    "expected_token_counts": {
        lang: {
            str(token_id): count
            for token_id, count in values.items()
        }
        for lang, values in expected_counts.items()
    },
    "expected_token_section_counts": {
        f"{lang}|{section}": {
            str(token_id): count
            for token_id, count in values.items()
        }
        for (lang, section), values in expected_section_counts.items()
    },
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(
    json.dumps(report, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("Report:", OUTPUT)
print("Audit finished. No training/checkpoint files modified.")
