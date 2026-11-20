from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from polycodegpt.data.tokenizer_io import load_tokenizer
from polycodegpt.tokenization import serializer

tok = load_tokenizer("tokenizer/artifacts/polycodegpt_tokenizer_v1_1.json")

def stats_for(path):
    recs = [json.loads(l) for l in open(path, encoding="utf-8")]
    # We'll stream parse instead to avoid huge list? recs okay for valid; train 392k lines ~2GB? Use streaming.
    pass

def process(path):
    py_total=[]; js_total=[]; py_doc=[]; js_doc=[]; py_code=[]; js_code=[]
    py_n=js_n=0
    py_tokens=js_tokens=0
    py_doc_tokens=js_doc_tokens=0
    py_code_tokens=js_code_tokens=0
    batch_texts=[]; batch_langs=[]
    def flush():
        nonlocal py_tokens, js_tokens, py_doc_tokens, js_doc_tokens, py_code_tokens, js_code_tokens
        if not batch_texts: return
        encs = tok.encode_batch(batch_texts, add_special_tokens=False)
        for lang, enc, text in zip(batch_langs, encs, batch_texts):
            ids = enc.ids
            # find markers
            try:
                ci = ids.index(3)  # CODE_ID? check
            except ValueError:
                ci = None
            # Use serializer assign? Simpler: count tokens between markers by text? Use IDs.
            # We'll just total length and doc/code via marker positions.
            # CODE_ID=3, DOCSTRING_ID=4, EOS_ID=2 in this tokenizer? verify below.
            pass
        batch_texts.clear(); batch_langs.clear()

    # Instead of complex, use serializer helpers not available; compute marker positions.
    # Re-run with marker detection.
    batch_texts=[]; batch_langs=[]
    def flush2():
        nonlocal py_tokens, js_tokens, py_doc_tokens, js_doc_tokens, py_code_tokens, js_code_tokens
        if not batch_texts: return
        encs = tok.encode_batch(batch_texts, add_special_tokens=False)
        for lang, enc in zip(batch_langs, encs):
            ids = list(enc.ids)
            # markers
            code_pos = ids.index(3)
            doc_pos = ids.index(4)
            eos_pos = ids.index(2)
            total = len(ids)
            doc = code_pos - doc_pos - 1
            code = eos_pos - code_pos - 1
            if lang == "python":
                py_total.append(total); py_doc.append(doc); py_code.append(code)
                py_n_ref = None
                py_tokens += total; py_doc_tokens += doc; py_code_tokens += code
            else:
                js_total.append(total); js_doc.append(doc); js_code.append(code)
                js_tokens += total; js_doc_tokens += doc; js_code_tokens += code
        batch_texts.clear(); batch_langs.clear()

    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            lang = r.get("language")
            if lang not in ("python","javascript"): 
                continue
            text = serializer.serialize_code_record(r)
            batch_texts.append(text)
            batch_langs.append(lang)
            if len(batch_texts) >= 256:
                flush2()
    flush2()

    def summarize(arr):
        a = np.asarray(arr, dtype=np.int64)
        return {
            "n": int(a.size),
            "mean": float(a.mean()) if a.size else 0.0,
            "median": float(np.median(a)) if a.size else 0.0,
            "p90": float(np.percentile(a,90)) if a.size else 0.0,
            "p95": float(np.percentile(a,95)) if a.size else 0.0,
            "max": int(a.max()) if a.size else 0,
        }

    print("FILE", path)
    print("Python records", len(py_total), "tokens", py_tokens, "doc", py_doc_tokens, "code", py_code_tokens)
    print("  total len", summarize(py_total))
    print("  doc len", summarize(py_doc))
    print("  code len", summarize(py_code))
    print("JavaScript records", len(js_total), "tokens", js_tokens, "doc", js_doc_tokens, "code", js_code_tokens)
    print("  total len", summarize(js_total))
    print("  doc len", summarize(js_doc))
    print("  code len", summarize(js_code))
    print("PY:JS token ratio", py_tokens/js_tokens)

process("data/processed/polycodegpt_corpus/valid.jsonl")
process("data/processed/polycodegpt_corpus/train.jsonl")
