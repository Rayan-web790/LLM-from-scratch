import json
from pathlib import Path
from collections import Counter

ROOT = Path("data/processed/polycodegpt_v2/scenario_b")
SAMPLE = Path("data/interim/scenario_c_audit/parent_sample.jsonl")

repo_partitions = {
    "train": set(),
    "valid": set(),
}

for partition in ("train", "valid"):
    path = ROOT / f"{partition}.jsonl"
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("repo"):
                repo_partitions[partition].add(r["repo"])

overlap = repo_partitions["train"] & repo_partitions["valid"]
print("Train repositories:", len(repo_partitions["train"]))
print("Validation repositories:", len(repo_partitions["valid"]))
print("Repository overlap:", len(overlap))

sample_counts = Counter()
unknown = Counter()

with SAMPLE.open(encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        repo = r["repo"]

        if repo in repo_partitions["train"]:
            sample_counts[r["language"] + "_train"] += 1
        elif repo in repo_partitions["valid"]:
            sample_counts[r["language"] + "_valid"] += 1
        else:
            unknown[r["language"]] += 1

print("\nSample parent partition mapping:")
for key, value in sorted(sample_counts.items()):
    print(f"  {key}: {value}")

print("\nRepositories missing from existing split:", dict(unknown))
print("\nPARTITION CHECK COMPLETE")
