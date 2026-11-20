import json
from pathlib import Path
from collections import Counter, defaultdict

p = Path("data/processed/polycodegpt_v2/scenario_b/train.jsonl")

category_task = Counter()
language_task = Counter()
marker_checks = defaultdict(Counter)
examples = defaultdict(list)

total = 0

with p.open("r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        total += 1

        language = r["language"]
        category = r.get("category", "unknown")
        task = r.get("task", "unknown")
        text = r["text"]

        category_task[(language, category, task)] += 1
        language_task[(language, task)] += 1

        checks = {
            "docstring": (
                bool(r.get("docstring_present")),
                "<|docstring|>" in text
            ),
            "imports": (
                bool(r.get("imports_present")),
                "<|imports|>" in text
            ),
            "signature": (
                bool(r.get("signature_present")),
                "<|signature|>" in text
            ),
        }

        for name, (metadata, marker) in checks.items():
            marker_checks[(language, name)]["metadata_true"] += metadata
            marker_checks[(language, name)]["marker_present"] += marker
            marker_checks[(language, name)]["both_true"] += (
                metadata and marker
            )
            marker_checks[(language, name)]["disagree"] += (
                metadata != marker
            )

            if metadata != marker and len(examples[(language, name)]) < 3:
                examples[(language, name)].append({
                    "path": r.get("path"),
                    "category": category,
                    "task": task,
                    "metadata": metadata,
                    "marker": marker,
                    "preview": text[:250],
                })

print("========== CATEGORY -> TASK CROSS-TAB ==========")

for (language, category, task), count in sorted(category_task.items()):
    print(
        f"{language:12s} "
        f"{category:15s} -> {task:12s} "
        f"{count:9,d}"
    )

print("\n========== TASK TOTALS BY LANGUAGE ==========")

for (language, task), count in sorted(language_task.items()):
    print(f"{language:12s} {task:15s} {count:10,d}")

print("\n========== METADATA / SERIALIZATION CONSISTENCY ==========")

for (language, name), counts in sorted(marker_checks.items()):
    print(f"\n{language.upper()} / {name}")
    for key, value in counts.items():
        print(f"  {key}: {value:,}")

    if examples[(language, name)]:
        print("  Example disagreements:")
        for item in examples[(language, name)]:
            print("   ", item)

print("\nTotal records:", total)
print("Audit complete. Read-only; no training files modified.")
