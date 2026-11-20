"""Phase 12B Part B — measure actual token sampling per experiment branch.

Streams the training corpus (read-only), serializes each record with the project
serializer, encodes with the project tokenizer, and reports per-language token
totals plus the derived sampling/revisit figures for each branch.

    python experiments/phase12b/verify_tokens.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from polycodegpt.data import load_tokenizer
from polycodegpt.data.tokenizer_io import TOKENIZER_PATH
from polycodegpt.tokenization.serializer import (
    serialize_python_record,
    serialize_javascript_record,
)

CORPUS = ROOT / "data" / "processed" / "polycodegpt_corpus" / "train.jsonl"
CURATED = ROOT / "data" / "curated" / "js_v1" / "train.jsonl"
OUT = ROOT / "reports" / "phase12b_token_sampling.json"


def encode_totals(records_iter, serializer, batch_size=512):
    tok = load_tokenizer(TOKENIZER_PATH)
    total = 0
    n = 0
    lengths = []
    batch_texts = []
    batch_n = 0
    for rec in records_iter:
        batch_texts.append(serializer(rec))
        batch_n += 1
        if batch_n >= batch_size:
            encs = tok.encode_batch(batch_texts, add_special_tokens=False)
            for enc in encs:
                l = len(enc.ids)
                total += l
                lengths.append(l)
            n += batch_n
            batch_texts = []
            batch_n = 0
            if n % 50000 == 0:
                print(f"  processed {n} records, tokens so far {total:,}", flush=True)
    if batch_texts:
        encs = tok.encode_batch(batch_texts, add_special_tokens=False)
        for enc in encs:
            l = len(enc.ids)
            total += l
            lengths.append(l)
        n += batch_n
    return n, total, lengths


def stream_lang(lang):
    with CORPUS.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("language") == lang:
                yield r


def stream_curated():
    with CURATED.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main():
    t0 = time.time()
    print("Python original...")
    pn, pt, plen = encode_totals(stream_lang("python"), serialize_python_record)
    print(f"  python: {pn} records, {pt:,} tokens")
    print("JavaScript original...")
    jn, jt, jlen = encode_totals(stream_lang("javascript"), serialize_javascript_record)
    print(f"  javascript: {jn} records, {jt:,} tokens")
    print("JavaScript curated...")
    cn, ct, clen = encode_totals(stream_curated(), serialize_javascript_record)
    print(f"  curated js: {cn} records, {ct:,} tokens")

    total_budget = pt + jt  # control total
    half = total_budget // 2

    # Branch sampling derived figures
    branches = {
        "control": {
            "python_records": pn, "python_tokens": pt,
            "javascript_records": jn, "javascript_tokens": jt,
            "python_share": pt / total_budget, "javascript_share": jt / total_budget,
            "js_revisits_per_record": 1.0,
            "js_unique_coverage": jn,
            "effective_js_epochs": 1.0,
            "sampling": "original distribution, no resampling",
        },
        "exp_a": {
            "python_records": pn, "python_tokens": pt,
            "javascript_records_available": cn, "javascript_tokens_available": ct,
            "javascript_target_tokens": jt,
            "js_revisits_per_record": jt / ct,
            "js_unique_coverage": cn,
            "effective_js_epochs": jt / ct,
            "python_share": pt / total_budget, "javascript_share": jt / total_budget,
            "sampling": "curated JS oversampled with replacement to original JS token volume",
        },
        "exp_b": {
            "python_target_tokens": half, "javascript_target_tokens": half,
            "python_records_available": pn, "python_tokens_available": pt,
            "javascript_records_available": jn, "javascript_tokens_available": jt,
            "python_subsample_ratio": half / pt,
            "js_oversample_ratio": half / jt,
            "python_share": 0.5, "javascript_share": 0.5,
            "sampling": "balanced 50/50 tokens: python subsampled, js oversampled with replacement",
        },
        "exp_c": {
            "python_target_tokens": half, "javascript_target_tokens": half,
            "python_records_available": pn, "python_tokens_available": pt,
            "javascript_records_available": cn, "javascript_tokens_available": ct,
            "python_subsample_ratio": half / pt,
            "js_oversample_ratio": half / ct,
            "python_share": 0.5, "javascript_share": 0.5,
            "sampling": "balanced 50/50 tokens: python subsampled, curated js oversampled with replacement",
        },
    }

    result = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tokenizer": "polycodegpt_tokenizer_v1_1",
        "original_python": {"records": pn, "tokens": pt, "median": None},
        "original_javascript": {"records": jn, "tokens": jt},
        "curated_javascript": {"records": cn, "tokens": ct},
        "control_total_tokens": total_budget,
        "half_budget": half,
        "ratio_python_to_javascript_original": pt / jt,
        "branches": branches,
        "elapsed_s": round(time.time() - t0, 1),
    }
    # medians
    import statistics
    result["original_python"]["median_tokens"] = statistics.median(plen)
    result["original_javascript"]["median_tokens"] = statistics.median(jlen)
    result["curated_javascript"]["median_tokens"] = statistics.median(clen)

    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2)[:3000])
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()