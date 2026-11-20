import json
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

p = Path("data/processed/polycodegpt_v2/scenario_b/train.jsonl")

alignment_counts = Counter()
doc_lengths = defaultdict(list)
failed_examples = defaultdict(list)

with p.open("r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)

        lang = r["language"]
        category = r.get("category", "unknown")
        task = r.get("task", "unknown")
        alignment = r.get("alignment", "missing")

        key = (lang, category, task, alignment)
        alignment_counts[key] += 1

        text = r["text"]
        marker = "<|docstring|>"

        if marker in text:
            doc = text.split(marker, 1)[1].split(
                "<|signature|>", 1
            )[0]
            doc_lengths[(lang, task)].append(len(doc.strip()))

        if alignment.startswith("failed_to_module:"):
            if len(failed_examples[alignment]) < 3:
                failed_examples[alignment].append({
                    "language": lang,
                    "category": category,
                    "path": r.get("path"),
                    "preview": text[:350],
                })

print("========== ALIGNMENT STATUS COUNTS ==========")

for key, count in sorted(alignment_counts.items()):
    print(f"{key}: {count:,}")

print("\n========== DOCSTRING LENGTHS ==========")

for key, values in sorted(doc_lengths.items()):
    a = np.asarray(values)
    print(
        key,
        "count:", len(a),
        "median:", int(np.median(a)),
        "p10:", int(np.percentile(a, 10)),
        "p90:", int(np.percentile(a, 90)),
        "min:", int(a.min()),
    )

print("\n========== FAILED ALIGNMENT EXAMPLES ==========")

for reason, examples in sorted(failed_examples.items()):
    print("\n", reason)
    for example in examples:
        print(example)

print("\nRead-only audit complete.")
