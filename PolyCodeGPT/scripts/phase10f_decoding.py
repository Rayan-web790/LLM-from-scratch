from __future__ import annotations
import json, random, sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from scripts.evaluate_phase10_benchmark import load_model_and_tokenizer, build_generation_prefix, tokenize_prompt, compute_syntax_validity
from scripts.phase10_tasks import javascript_tasks

def gen_one(model, tok, doc, stub, max_new_tokens, temperature, top_k, seed):
    if temperature != 0:
        torch.manual_seed(seed)
        random.seed(seed)
    prefix = build_generation_prefix("javascript", doc, stub)
    ids = tokenize_prompt(prefix, tok)
    device = torch.device("cuda")
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    full = model.generate(inp, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, eos_token_id=2)
    gen_ids = full[0, len(ids):].tolist()
    eos = 2 in gen_ids
    if eos:
        gen_ids = gen_ids[:gen_ids.index(2)]
    text = tok.decode(gen_ids, skip_special_tokens=False)
    cand = stub + text
    from collections import Counter
    lines = [l for l in text.splitlines() if l.strip()]
    rl = (sum(1 for c in Counter(lines).values() if c > 1) / len(lines)) if lines else 0.0
    toks = text.split()
    grams = [tuple(toks[i:i+4]) for i in range(len(toks)-3)] if len(toks) >= 8 else []
    gc = Counter(grams)
    r4 = (sum(c for c in gc.values() if c > 1) / len(grams)) if grams else 0.0
    return {
        "seed": seed if temperature != 0 else None,
        "generated_text": text,
        "candidate": cand,
        "eos_hit": eos,
        "generated_token_count": len(gen_ids),
        "syntax_valid": bool(compute_syntax_validity(cand, "javascript")),
        "unique_token_ratio": (len(set(gen_ids)) / len(gen_ids) if gen_ids else 0.0),
        "repeated_line_ratio": rl,
        "repeated_4gram_ratio": r4,
    }

def main():
    model, tok, step = load_model_and_tokenizer("checkpoints/polycodegpt_first_pass/step_008130.pt", torch.device("cuda"))
    tasks = list(javascript_tasks())[:10]
    settings = [
        ("greedy_temp0", 0.0, None),
        ("sampling_t0.4_topkNone", 0.4, None),
        ("sampling_t0.8_topkNone", 0.8, None),
        ("sampling_t0.8_topk50", 0.8, 50),
        ("sampling_t0.4_topk50", 0.4, 50),
    ]
    results = {}
    for name, temp, topk in settings:
        runs = []
        seeds = [None] if temp == 0 else [42, 123, 999]
        for seed in seeds:
            run_results = []
            for doc, stub in tasks:
                run_results.append(gen_one(model, tok, doc, stub, 64, temp, topk, seed if seed is not None else 42))
            runs.append({"seed": seed, "results": run_results})
        results[name] = {"temperature": temp, "top_k": topk, "max_new_tokens": 64, "runs": runs}
        # summary
        for run in runs:
            eos = sum(r["eos_hit"] for r in run["results"])
            sv = sum(r["syntax_valid"] for r in run["results"])
            print(f"{name} seed={run['seed']}: EOS {eos}/10, syntax_valid {sv}/10")
    out = Path("reports/phase10f_decoding_diagnostics.json")
    out.write_text(json.dumps({"tasks": [t[0] for t in tasks], "results": results}, indent=2), encoding="utf-8")
    print("saved", out)

if __name__ == "__main__":
    main()
