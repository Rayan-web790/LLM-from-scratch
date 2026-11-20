"""Contamination check for independent benchmark v2."""
from __future__ import annotations
import hashlib, json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
MANIFEST = ROOT / "functional" / "independent_v2" / "manifest.json"
PHASE11 = ROOT / "functional" / "manifest.json"
V1 = ROOT / "functional" / "independent" / "manifest.json"
CORPUS = ROOT / "data" / "processed" / "polycodegpt_corpus" / "train.jsonl"
OUT = ROOT / "reports" / "phase12b_contamination_v2.json"

def normalize_code(code):
    return re.sub(r"\s+", " ", code.strip())

def main():
    v2 = json.loads(MANIFEST.read_text(encoding="utf-8"))
    p11 = json.loads(PHASE11.read_text(encoding="utf-8"))
    v1 = json.loads(V1.read_text(encoding="utf-8"))

    # entry overlap with Phase 11
    p11_entries = {(t["language"], t["entry"]) for t in p11["tasks"]}
    p11_overlaps = [(t["task_id"], t["language"], t["entry"]) for t in v2["tasks"]
                    if (t["language"], t["entry"]) in p11_entries]

    # entry overlap with v1
    v1_entries = {(t["language"], t["entry"]) for t in v1["tasks"]}
    v1_overlaps = [(t["task_id"], t["language"], t["entry"]) for t in v2["tasks"]
                   if (t["language"], t["entry"]) in v1_entries]

    # corpus function names
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
    for t in v2["tasks"]:
        if t["entry"].lower() in corpus_names[t["language"]]:
            name_hits.append((t["task_id"], t["entry"]))

    # reference code hash vs corpus (coarse: hash whole ref file per language)
    refs = {
        "python": (ROOT / "functional" / "independent_v2" / "reference" / "python_refs.py").read_text(encoding="utf-8"),
        "javascript": (ROOT / "functional" / "independent_v2" / "reference" / "javascript_refs.js").read_text(encoding="utf-8"),
    }
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

    # behavior near-duplicate: jaccard between v2 and p11 behaviors
    def words(s):
        return set(re.findall(r"[A-Za-z_]+", (s or "").lower()))
    near = []
    for a in v2["tasks"]:
        for b in p11["tasks"]:
            ja = words(a["behavior"]) & words(b["behavior"])
            union = words(a["behavior"]) | words(b["behavior"])
            jac = len(ja) / len(union) if union else 0
            if jac >= 0.6:
                near.append({"v2": a["task_id"], "p11": b["task_id"], "jaccard": round(jac, 3)})

    result = {
        "independent_tasks": len(v2["tasks"]),
        "phase11_entry_overlaps": p11_overlaps,
        "v1_entry_overlaps": v1_overlaps,
        "corpus_function_name_hits": name_hits,
        "reference_file_exact_hash_hits": hash_hits,
        "behavior_near_duplicate_pairs": near,
        "contamination_found": bool(p11_overlaps or v1_overlaps or name_hits or hash_hits or near),
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if not result["contamination_found"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
