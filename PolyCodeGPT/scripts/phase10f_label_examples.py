"""Write ten complete labelled records per language as UTF-8 evidence.

The sample is the same one used by the Phase 10E audit: ``random.seed(42)`` over
the records of each language in ``train.jsonl``, preserving file order. Output is
written as UTF-8 JSON so Windows console code pages cannot truncate it.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.phase10f_labels import CATEGORY_NAMES, classify  # noqa: E402

TRAIN = ROOT / "data" / "processed" / "polycodegpt_corpus" / "train.jsonl"
OUT = ROOT / "reports" / "phase10f_label_examples.json"
N = 200
K = 10


def main() -> None:
    random.seed(42)
    by_lang: dict[str, list[dict]] = {"python": [], "javascript": []}
    with open(TRAIN, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            lang = rec.get("language")
            if lang in by_lang:
                by_lang[lang].append(rec)

    payload = {
        "note": (
            "Exploratory heuristic labels, not manual semantic annotation. "
            "Sample: random.seed(42), 200 records per language from train.jsonl, "
            "first 10 of each sample shown in full."
        ),
        "categories": list(CATEGORY_NAMES),
        "sample_seed": 42,
        "sample_size": N,
        "languages": {},
    }

    for lang, records in by_lang.items():
        sample = random.sample(records, N)
        payload["languages"][lang] = {
            "corpus_records": len(records),
            "examples": [
                {
                    "index": i + 1,
                    "record_id": rec.get("record_id"),
                    "repo": rec.get("repo"),
                    "path": rec.get("path"),
                    "description_chars": len(rec.get("docstring") or ""),
                    "code_lines": len((rec.get("code") or "").splitlines()),
                    "description": rec.get("docstring"),
                    "code": rec.get("code"),
                    "heuristic_labels": sorted(classify(lang, rec)),
                }
                for i, rec in enumerate(sample[:K])
            ],
        }

    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for lang, data in payload["languages"].items():
        labels = [", ".join(e["heuristic_labels"]) for e in data["examples"]]
        print(f"{lang}: {len(data['examples'])} examples written from {data['corpus_records']:,} records")
        for i, lab in enumerate(labels, 1):
            print(f"  {i}. {lab}")
    print(f"saved {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
