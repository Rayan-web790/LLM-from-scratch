import json
from collections import Counter, defaultdict
from pathlib import Path

p = Path("data/processed/polycodegpt_v2/scenario_b/train.jsonl")

counts = Counter()
totals = Counter()
alignment = Counter()

with p.open("r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        source = r["source_id"]
        lang = r["language"]
        task = r["task"]

        counts[(source, lang, task)] += 1
        totals[(source, lang)] += 1
        alignment[(source, lang, r.get("alignment", "missing"))] += 1

print("========== SOURCE / LANGUAGE / TASK ==========")

for (source, lang), total in sorted(totals.items()):
    print(f"\n{source} | {lang} | total={total:,}")
    for task in ("function", "module", "text"):
        n = counts[(source, lang, task)]
        print(f"  {task:10s}: {n:>9,} ({100*n/total:5.1f}%)")

print("\n========== SOURCE / ALIGNMENT ==========")

for key, n in sorted(alignment.items()):
    print(f"{key}: {n:,}")

print("\nAudit complete. No files modified.")
