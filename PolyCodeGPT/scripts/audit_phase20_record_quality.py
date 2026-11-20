import json
from pathlib import Path
from collections import defaultdict
import numpy as np

p = Path("data/processed/polycodegpt_v2/scenario_b/train.jsonl")

stats = defaultdict(lambda: {
    "records": 0,
    "tokens": [],
    "docstrings": 0,
    "imports": 0,
    "signatures": 0,
    "task_mismatches": 0,
})

total = 0
record_ids = set()
duplicate_ids = 0

with p.open("r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        total += 1

        rid = r["record_id"]
        if rid in record_ids:
            duplicate_ids += 1
        record_ids.add(rid)

        key = (r["language"], r.get("category", "unknown"))
        s = stats[key]

        s["records"] += 1
        s["tokens"].append(int(r["token_count"]))
        s["docstrings"] += bool(r.get("docstring_present"))
        s["imports"] += bool(r.get("imports_present"))
        s["signatures"] += bool(r.get("signature_present"))
        s["task_mismatches"] += (
            r.get("task") != r.get("category")
        )

print("========== RECORD QUALITY AUDIT ==========")
print("Total records:", total)
print("Duplicate record IDs:", duplicate_ids)

for (language, category), s in sorted(stats.items()):
    lengths = np.asarray(s["tokens"], dtype=np.int64)

    print(f"\n--- {language.upper()} / {category.upper()} ---")
    print("Records:", s["records"])
    print("Total tokens:", int(lengths.sum()))
    print("Average tokens:", round(float(lengths.mean()), 2))
    print("Median tokens:", int(np.median(lengths)))
    print("P90 tokens:", int(np.percentile(lengths, 90)))
    print("Maximum tokens:", int(lengths.max()))
    print("Docstring records:", s["docstrings"])
    print("Import records:", s["imports"])
    print("Signature records:", s["signatures"])
    print("Task/category mismatches:", s["task_mismatches"])

print("\n========== AUDIT COMPLETE ==========")
