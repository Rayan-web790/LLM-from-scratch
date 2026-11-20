from __future__ import annotations

import json
import random
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts.evaluate_phase10_benchmark import (
    load_model_and_tokenizer,
    build_generation_prefix,
    tokenize_prompt,
    compute_syntax_validity,
)
from scripts.phase10_tasks import javascript_tasks

import subprocess, tempfile, os, shutil

def rep_metrics(text):
    lines = [l for l in text.splitlines() if l.strip()]
    from collections import Counter
    repeated_lines = sum(1 for c in Counter(lines).values() if c > 1) / len(lines) if lines else 0.0
    toks = text.split()
    grams = [tuple(toks[i:i+4]) for i in range(len(toks)-3)] if len(toks) >= 8 else []
    gc = Counter(grams)
    rep4 = (sum(c for c in gc.values() if c > 1) / len(grams)) if grams else 0.0
    ids = None
    return repeated_lines, rep4

def run_setting(model, tok, tasks, max_new_tokens, temperature, top_k):
    out = []
    for doc, stub in tasks:
        prefix = build_generation_prefix("javascript", doc, stub)
        ids = tokenize_prompt(prefix, tok)
        inp = torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device if list(model.parameters()) else torch.device("cpu"))
        # device from model maybe cpu; use cuda
        device = torch.device("cuda")
        inp = torch.tensor([ids], dtype=torch.long, device=device)
        if temperature == 0:
            gen_ids = model.generate(inp, max_new_tokens=max_new_tokens, temperature=0.0, eos_token_id=2)[0, len(ids):].tolist()
            eos = 2 in gen_ids
            if eos:
                gen_ids = gen_ids[:gen_ids.index(2)]
        else:
            torch.manual_seed(42)
            random.seed(42)
            full = model.generate(inp, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, eos_token_id=2)
            gen_ids = full[0, len(ids):].tolist()
            eos = 2 in gen_ids
            if eos:
                gen_ids = gen_ids[:gen_ids.index(2)]
        text = tok.decode(gen_ids, skip_special_tokens=False)
        cand = stub + text
        rl, r4 = rep_metrics(text)
        out.append({
            "docstring": doc,
            "generated_text": text,
            "candidate": cand,
            "eos_hit": eos,
            "generated_token_count": len(gen_ids),
            "syntax_valid": bool(compute_syntax_validity(cand, "javascript")),
            "repeated_line_ratio": rl,
            "repeated_4gram_ratio": r4,
            "unique_token_ratio": (len(set(gen_ids)) / len(gen_ids)) if gen_ids else 0.0,
        })
    return out

def main():
    model, tok, step = load_model_and_tokenizer("checkpoints/polycodegpt_first_pass/step_008130.pt", torch.device("cuda"))
    tasks = list(javascript_tasks())[:10]
    results = {}
    # 1 greedy 64
    results["greedy_64"] = run_setting(model, tok, tasks, 64, 0.0, None)
    # 2 greedy 128
    results["greedy_128"] = run_setting(model, tok, tasks, 128, 0.0, None)
    # 3 sampling 0.8 top_k 50
    results["sampling_t0.8_topk50_64"] = run_setting(model, tok, tasks, 64, 0.8, 50)
    # repetition penalty unsupported
    results["repetition_penalty"] = "unsupported: model.generate does not accept repetition_penalty"

    # summarize
    for name, data in results.items():
        if isinstance(data, str):
            print(name, data)
            continue
        eos = sum(d["eos_hit"] for d in data)
        sv = sum(d["syntax_valid"] for d in data)
        print(f"{name}: EOS {eos}/{len(data)}, syntax_valid {sv}/{len(data)}")

    out_path = Path("reports/phase10e_inference_diagnostics.json")
    out_path.write_text(json.dumps({"tasks": [t[0] for t in tasks], "results": results}, indent=2), encoding="utf-8")
    print("saved", out_path)

if __name__ == "__main__":
    main()
