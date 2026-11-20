import json
from pathlib import Path
from collections import defaultdict

p = Path("data/processed/polycodegpt_v2/scenario_b/train.jsonl")

targets = {
    ("python", "function", "module"),
    ("javascript", "function", "module"),
    ("python", "function", "function"),
    ("javascript", "function", "function"),
}

examples = defaultdict(list)

with p.open("r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)

        key = (
            r["language"],
            r.get("category"),
            r.get("task"),
        )

        if key not in targets:
            continue

        if len(examples[key]) >= 5:
            continue

        text = r["text"]
        code_marker = "<|code|>"

        if code_marker in text:
            code = text.split(code_marker, 1)[1]
        else:
            code = text

        examples[key].append({
            "path": r.get("path"),
            "record_id": r.get("record_id"),
            "docstring_present": r.get("docstring_present"),
            "signature_present": r.get("signature_present"),
            "token_count": r.get("token_count"),
            "serialized_prefix": text[:500],
            "code_preview": code[:1200],
        })

for key in sorted(examples):
    print("\n" + "=" * 90)
    print("LANGUAGE / CATEGORY / TASK:", key)
    print("=" * 90)

    for i, item in enumerate(examples[key], 1):
        print(f"\n--- EXAMPLE {i} ---")
        for field, value in item.items():
            print(f"\n{field}:")
            print(value)

print("\nRead-only diagnostic complete.")
