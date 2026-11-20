import json
from collections import Counter, defaultdict
from pathlib import Path

p = Path("data/interim/scenario_c_audit/extracted_candidates.jsonl")

languages = defaultdict(lambda: {
    "records": 0,
    "tokens": 0,
    "over_512_records": 0,
    "over_1024_records": 0,
    "repositories": Counter(),
    "categories": Counter(),
    "dependencies": Counter(),
    "lengths": [],
})

with p.open(encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        d = languages[r["language"]]

        d["records"] += 1
        d["tokens"] += r["token_count"]
        d["lengths"].append(r["token_count"])
        d["repositories"][r["repo"]] += r["token_count"]
        d["categories"][r["category"]] += 1

        if r["token_count"] > 512:
            d["over_512_records"] += 1
        if r["token_count"] > 1024:
            d["over_1024_records"] += 1

        d["dependencies"].update(r.get("dependency_flags", []))

print("========== SCENARIO C DATASET IMPACT ==========")

for lang, d in sorted(languages.items()):
    lengths = sorted(d["lengths"])
    n = len(lengths)

    def percentile(q):
        return lengths[min(n - 1, int((n - 1) * q))]

    print(f"\n--- {lang.upper()} ---")
    print("Records:", d["records"])
    print("Total tokens:", f'{d["tokens"]:,}')
    print("Median tokens:", percentile(0.50))
    print("P90 tokens:", percentile(0.90))
    print("P95 tokens:", percentile(0.95))
    print("Over 512:", d["over_512_records"])
    print("Over 1024:", d["over_1024_records"])
    print("Unique repositories:", len(d["repositories"]))

    print("\nCategories:")
    for name, count in d["categories"].most_common():
        print(f"  {name}: {count:,}")

    print("\nDependency flags:")
    for name, count in d["dependencies"].most_common():
        print(f"  {name}: {count:,}")

    print("\nTop 10 repositories by extracted tokens:")
    for repo, tokens in d["repositories"].most_common(10):
        print(f"  {tokens:>9,} tokens | {repo}")

total_records = sum(d["records"] for d in languages.values())
total_tokens = sum(d["tokens"] for d in languages.values())

print("\n========== COMBINED ==========")
print("Records:", f"{total_records:,}")
print("Tokens:", f"{total_tokens:,}")

for lang, d in sorted(languages.items()):
    print(
        f"{lang}: {d['tokens'] / total_tokens:.1%} of extracted tokens"
    )

print("\nDATASET IMPACT AUDIT COMPLETE")
