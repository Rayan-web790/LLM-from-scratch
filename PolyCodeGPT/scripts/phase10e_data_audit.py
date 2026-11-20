from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from scripts.phase10f_labels import classify  # noqa: E402

random.seed(42)
N = 200
py = []
js = []
with open("data/processed/polycodegpt_corpus/train.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r.get("language") == "python":
            py.append(r)
        elif r.get("language") == "javascript":
            js.append(r)

py_s = random.sample(py, N)
js_s = random.sample(js, N)


for name, sample, lang in [("Python", py_s, "python"), ("JavaScript", js_s, "javascript")]:
    counts = Counter()
    for r in sample:
        for c in classify(lang, r):
            counts[c] += 1
    print(f"=== {name} sample (n={len(sample)}) ===")
    for k, v in sorted(counts.items()):
        print(f"{k}: {v} ({v/len(sample)*100:.1f}%)")
    print()
    # representatives
    print(f"Representative {name} examples:")
    for r in sample[:3]:
        print("DOC:", r.get("docstring", "")[:150].replace("\n", " "))
        print("CODE:", r.get("code", "")[:200].replace("\n", "\\n"))
        print("---")
    print()
