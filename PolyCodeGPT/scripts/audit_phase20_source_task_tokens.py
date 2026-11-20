import json
from collections import defaultdict
from pathlib import Path

p = Path("data/processed/polycodegpt_v2/scenario_b/train.jsonl")

stats = defaultdict(lambda: {"records": 0, "tokens": 0})

with p.open("r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        key = (r["source_id"], r["language"], r["task"])

        stats[key]["records"] += 1
        stats[key]["tokens"] += int(r["token_count"])

print("========== TOKEN EXPOSURE BY SOURCE / TASK ==========")

for (source, lang, task), values in sorted(stats.items()):
    print(
        f"{source:12s} {lang:10s} {task:8s} "
        f"records={values['records']:>9,} "
        f"tokens={values['tokens']:>12,}"
    )

print("\n========== LANGUAGE TOTALS ==========")

language_totals = defaultdict(int)

for (source, lang, task), values in stats.items():
    language_totals[lang] += values["tokens"]

for lang, tokens in sorted(language_totals.items()):
    print(f"{lang:10s} {tokens:>15,}")

print("\nRead-only audit complete.")
