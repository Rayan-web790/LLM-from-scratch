"""Phase 12B Part C — audit independent benchmark v1 for genuine independence."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

V1 = ROOT / "functional" / "independent" / "manifest.json"
P11 = ROOT / "functional" / "manifest.json"
OUT = ROOT / "reports" / "phase12b_benchmark_v1_audit.json"


def words(s):
    return set(re.findall(r"[A-Za-z_]+", (s or "").lower()))


def _t(x):
    if isinstance(x, list):
        return tuple(_t(i) for i in x)
    if isinstance(x, dict):
        return tuple(sorted((k, _t(v)) for k, v in x.items()))
    return x

def case_signature(task):
    sigs = []
    for c in task["cases"]:
        sigs.append((c["kind"], _t(c.get("args", [])),
               repr(c.get("expected", c.get("expect_error")))))
    return tuple(sigs)


def main():
    v1 = json.loads(V1.read_text(encoding="utf-8"))["tasks"]
    p11 = json.loads(P11.read_text(encoding="utf-8"))["tasks"]

    # behavior similarity
    behav_pairs = []
    for a in v1:
        for b in p11:
            ja = words(a["behavior"]) & words(b["behavior"])
            union = words(a["behavior"]) | words(b["behavior"])
            jac = len(ja) / len(union) if union else 0
            if jac >= 0.35:
                behav_pairs.append({"v1": a["task_id"], "p11": b["task_id"], "jaccard": round(jac, 3)})

    # case I/O exact overlap
    p11_sigs = {(t["language"], case_signature(t)) for t in p11}
    case_overlaps = [t["task_id"] for t in v1 if (t["language"], case_signature(t)) in p11_sigs]

    # entry name overlap (should be none after renaming)
    name_overlaps = [(t["task_id"], t["entry"]) for t in v1
                     if t["entry"] in {x["entry"] for x in p11 if x["language"] == t["language"]}]

    # reference code 5-gram jaccard (rough) between v1 and p11 refs? skip (different files)

    result = {
        "v1_tasks": len(v1),
        "v1_cases": sum(len(t["cases"]) for t in v1),
        "phase11_tasks": len(p11),
        "behavior_similar_pairs": behav_pairs,
        "case_io_exact_overlaps": case_overlaps,
        "entry_name_overlaps": name_overlaps,
        "assessment": (
            "v1 tasks are generic algorithmic primitives (sum, max, reverse, unique, "
            "anagram, clamp, etc.) whose problem shapes overlap substantially with "
            "Phase 11 tasks even after renaming. Classified as a PILOT benchmark, "
            "not a genuinely independent final benchmark. v2 (100 tasks) is required."
        ),
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2)[:2000])


if __name__ == "__main__":
    main()
