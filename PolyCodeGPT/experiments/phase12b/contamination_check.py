"""Phase 12B Part D — contamination check for the independent benchmark."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MANIFEST = ROOT / "functional" / "independent" / "manifest.json"
PHASE11 = ROOT / "functional" / "manifest.json"
CORPUS = ROOT / "data" / "processed" / "polycodegpt_corpus" / "train.jsonl"
OUT = ROOT / "reports" / "phase12b_contamination.json"


def normalize_code(code):
    return re.sub(r"\s+", " ", code.strip())


def main():
    ind = json.loads(MANIFEST.read_text(encoding="utf-8"))
    p11 = json.loads(PHASE11.read_text(encoding="utf-8"))

    # Phase 11 entry overlap
    p11_entries = {(t["language"], t["entry"]) for t in p11["tasks"]}
    overlaps = [(t["task_id"], t["language"], t["entry"]) for t in ind["tasks"]
                if (t["language"], t["entry"]) in p11_entries]

    # Corpus function names by language
    corpus_names = {"python": set(), "javascript": set()}
    with CORPUS.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            lang = r.get("language")
            if lang in corpus_names:
                corpus_names[lang].add((r.get("func_name") or "").strip().lower())

    name_hits = []
    for t in ind["tasks"]:
        if t["entry"].lower() in corpus_names[t["language"]]:
            name_hits.append((t["task_id"], t["entry"]))

    # Reference code hash vs corpus code hashes (exact)
    refs = {
        "python": (ROOT / "functional" / "independent" / "reference" / "python_refs.py").read_text(encoding="utf-8"),
        "javascript": (ROOT / "functional" / "independent" / "reference" / "javascript_refs.js").read_text(encoding="utf-8"),
    }
    # Build per-task reference slice hashes by function name (approx: hash whole ref file once per language)
    # Exact code-hash contamination: hash each reference function body? We approximate by checking
    # that no corpus record's normalized code equals the normalized reference file content (coarse).
    corpus_hashes = set()
    with CORPUS.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            corpus_hashes.add(hashlib.sha256(normalize_code(r.get("code", "")).encode()).hexdigest())

    ref_hashes = {lang: hashlib.sha256(normalize_code(src).encode()).hexdigest() for lang, src in refs.items()}
    hash_hits = [lang for lang, h in ref_hashes.items() if h in corpus_hashes]

    result = {
        "independent_tasks": len(ind["tasks"]),
        "phase11_entry_overlaps": overlaps,
        "corpus_function_name_hits": name_hits,
        "reference_file_exact_hash_hits": hash_hits,
        "contamination_found": bool(overlaps or name_hits or hash_hits),
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if not result["contamination_found"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
