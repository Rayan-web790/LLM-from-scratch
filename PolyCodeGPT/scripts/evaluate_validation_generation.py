"""PolyCodeGPT Phase 6B: validation-set generation diagnostic (read-only).

Isolated validation-only inference experiment. Uses ONLY ``valid.jsonl`` (never
test data), the existing step-8130 checkpoint, the existing tokenizer, and greedy
decoding (temperature=0). No weights are modified and no training runs.

Experiments:
  A. Docstring-to-code: prompt ends at ``<|code|>``, using the real validation
     docstring; the actual validation code is the reference.
  B. Code-prefix continuation: the model input ends after about half of the
     validation code tokens; the remaining code tokens are the reference.

The user runs this manually in their terminal; Freebuff does NOT execute it.

Usage (Windows CMD):
    cd /d "C:\\Users\\UltraPc\\Desktop\\LLM FROM SCRATCH\\PolyCodeGPT"
    "C:\\Users\\UltraPc\\Desktop\\LLM FROM SCRATCH\\.venv\\Scripts\\python.exe" scripts\\evaluate_validation_generation.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch

from polycodegpt.data.tokenizer_io import CODE_ID, EOS_ID, load_tokenizer
from polycodegpt.model import GPT, ModelConfig
from polycodegpt.model.checkpoint import load_checkpoint as load_model_checkpoint
from polycodegpt.tokenization.serializer import (
    CODE_TOKEN, DOCSTRING_TOKEN, escape_content, record_language, serialize_code_record,
)
from polycodegpt.training import rebuild_model, require_cuda, weights_are_tied
from polycodegpt.training.checkpoint import load_checkpoint as load_training_checkpoint

DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "polycodegpt_first_pass" / "step_008130.pt"
TOKENIZER_PATH = PROJECT_ROOT / "tokenizer" / "artifacts" / "polycodegpt_tokenizer_v1_1.json"
MODEL_CONFIG_PATH = PROJECT_ROOT / "configs" / "model.yaml"
VALID_PATH = PROJECT_ROOT / "data" / "processed" / "polycodegpt_corpus" / "valid.jsonl"
REPORT_PATH = PROJECT_ROOT / "reports" / "phase6_validation_generation_diagnostic.md"

MAX_NEW_TOKENS = 128
SEED = 42
N_PER_LANGUAGE = 10
MIN_CODE_TOKENS = 24
MIN_PREFIX_TOKENS = 8
MAX_PREFIX_TOKENS = 64
MIN_CONTINUATION_TOKENS = 8


class InsufficientEligibleRecordsError(RuntimeError):
    pass


def stable_record_id(record):
    rid = record.get("record_id")
    if rid:
        return str(rid)
    payload = json.dumps(record, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def record_hash(record):
    return hashlib.sha256(stable_record_id(record).encode("utf-8")).hexdigest()[:16]


def count_code_tokens(ids, code_id=CODE_ID, eos_id=EOS_ID):
    if code_id not in ids:
        return 0
    code_index = ids.index(code_id)
    tail = list(ids[code_index + 1:])
    if tail and tail[-1] == eos_id:
        tail = tail[:-1]
    return len(tail)


def tokenize_serialized(record, tokenizer):
    text = serialize_code_record(dict(record))
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    if not isinstance(ids, list) or not all(isinstance(x, int) for x in ids):
        raise TypeError("tokenizer.encode() must return a list of ints")
    return text, list(ids)


def eligibility_reason(record, tokenizer=None, min_code_tokens=MIN_CODE_TOKENS):
    docstring = record.get("docstring")
    code = record.get("code")
    if not isinstance(docstring, str) or not docstring.strip():
        return "empty docstring"
    if not isinstance(code, str) or not code.strip():
        return "empty code"
    try:
        lang = record_language(dict(record))
    except Exception as exc:
        return f"invalid language: {exc}"
    if lang not in ("python", "javascript"):
        return f"unsupported language: {lang}"
    if tokenizer is None:
        return None
    try:
        _, ids = tokenize_serialized(record, tokenizer)
    except Exception as exc:
        return f"tokenization failed: {exc}"
    n_code = count_code_tokens(ids)
    if n_code < min_code_tokens:
        return f"too few code tokens ({n_code} < {min_code_tokens})"
    if len(ids) > 0 and ids[-1] != EOS_ID:
        return "serialized sequence missing trailing EOS"
    return None


def is_eligible(record, tokenizer=None, min_code_tokens=MIN_CODE_TOKENS):
    return eligibility_reason(record, tokenizer, min_code_tokens) is None


def select_validation_samples(records, tokenizer=None, *, seed=SEED, n_per_language=N_PER_LANGUAGE, min_code_tokens=MIN_CODE_TOKENS):
    pools = {"python": [], "javascript": []}
    skipped = []
    for record in records:
        rec = dict(record)
        reason = eligibility_reason(rec, tokenizer, min_code_tokens)
        if reason is not None:
            skipped.append((record_hash(rec), reason))
            continue
        try:
            lang = record_language(rec)
        except Exception as exc:
            skipped.append((record_hash(rec), f"invalid language: {exc}"))
            continue
        if lang not in pools:
            skipped.append((record_hash(rec), f"unsupported language: {lang}"))
            continue
        pools[lang].append(rec)
    selected = []
    rng = random.Random(seed)
    for lang in ("python", "javascript"):
        pool = sorted(pools[lang], key=lambda r: stable_record_id(r))
        if len(pool) < n_per_language:
            raise InsufficientEligibleRecordsError(
                f"insufficient eligible {lang} records: need {n_per_language}, have {len(pool)}"
            )
        selected.extend(rng.sample(pool, n_per_language))
    return selected, skipped


select_samples = select_validation_samples
select_validation_sample = select_validation_samples
sample_records = select_validation_samples
filter_eligible_records = select_validation_samples


def ensure_validation_path(path=VALID_PATH):
    """Refuse any split whose filename is not ``valid.jsonl``."""
    path = Path(path)
    if path.name != "valid.jsonl":
        raise ValueError(f"refusing to read a non-validation split: {path}")
    return path


def load_validation_records(path=VALID_PATH):
    path = ensure_validation_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"validation file not found: {path}")
    records = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({exc})") from exc
    return records


def build_docstring_prompt(record):
    lang = record_language(dict(record))
    token = "<|lang:python|>" if lang == "python" else "<|lang:javascript|>"
    docstring = record.get("docstring") or ""
    return f"{token}{DOCSTRING_TOKEN}{escape_content(docstring)}{CODE_TOKEN}"


def slice_code_prefix(ids, *, code_id=CODE_ID, eos_id=EOS_ID, min_prefix=MIN_PREFIX_TOKENS, max_prefix=MAX_PREFIX_TOKENS, min_continuation=MIN_CONTINUATION_TOKENS):
    ids = list(ids)
    if code_id not in ids:
        raise ValueError("serialized sequence has no <|code|> marker")
    code_index = ids.index(code_id)
    code_ids = ids[code_index + 1:]
    if code_ids and code_ids[-1] == eos_id:
        code_ids = code_ids[:-1]
    if not code_ids:
        raise ValueError("empty code token sequence")
    n = len(code_ids)
    if n < min_prefix + min_continuation:
        raise ValueError(f"insufficient code tokens: need at least {min_prefix + min_continuation}, got {n}")
    prefix_len = min(max_prefix, max(min_prefix, n // 2))
    if n - prefix_len < min_continuation:
        prefix_len = n - min_continuation
    if prefix_len < min_prefix:
        prefix_len = min_prefix
    if prefix_len > max_prefix:
        prefix_len = max_prefix
    if n - prefix_len < min_continuation:
        raise ValueError("cannot satisfy both prefix and continuation constraints")
    prompt_ids = ids[: code_index + 1 + prefix_len]
    reference_ids = code_ids[prefix_len:]
    return prompt_ids, reference_ids, code_ids, prefix_len


compute_code_prefix = slice_code_prefix
code_prefix_split = slice_code_prefix
split_code_prefix = slice_code_prefix


def protect_context_length(prompt_ids, context_length):
    if len(prompt_ids) > context_length:
        raise ValueError(f"prompt length {len(prompt_ids)} exceeds context length {context_length}")


def ensure_context_fits(prompt_ids, context_length):
    protect_context_length(prompt_ids, context_length)



def load_model_and_tokenizer(checkpoint_path, device):
    model_config = ModelConfig.from_yaml(MODEL_CONFIG_PATH)
    tokenizer = load_tokenizer(TOKENIZER_PATH, vocab_size=model_config.vocab_size)
    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint_path}")
    try:
        payload = load_training_checkpoint(checkpoint_path, map_location="cpu")
        model = rebuild_model(payload, device="cpu")
        step = payload["step"]
    except Exception:
        state_dict, config, meta = load_model_checkpoint(checkpoint_path, device="cpu")
        model = GPT(config)
        model.load_state_dict(state_dict)
        step = meta.get("step", "unknown")
    if not weights_are_tied(model):
        model.lm_head.weight = model.token_embedding.weight
    model = model.to(device).eval()
    if not all(torch.isfinite(p).all().item() for p in model.parameters()):
        raise SystemExit("checkpoint contains non-finite parameters")
    return model, tokenizer, step


def generate_greedy(model, device, prompt_ids, max_new_tokens, eos_id):
    input_ids = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=0, eos_token_id=eos_id)
    generated = out[0, len(prompt_ids):].tolist()
    stopped_at_eos = eos_id in generated
    if stopped_at_eos:
        gen_no_eos = generated[:generated.index(eos_id)]
    else:
        gen_no_eos = generated
    reached_limit = not stopped_at_eos
    return gen_no_eos, stopped_at_eos, reached_limit


def is_repetitive(text):
    tokens = text.split()
    if len(tokens) < 8:
        return False
    from collections import Counter
    grams = [tuple(tokens[i:i + 4]) for i in range(len(tokens) - 3)]
    counts = Counter(grams)
    return any(c > 3 for c in counts.values())


def diagnostic_similarity(generated_ids, reference_ids):
    g = list(generated_ids)
    r = list(reference_ids)
    match = 0
    for a, b in zip(g, r):
        if a == b:
            match += 1
        else:
            break
    exact = (g == r) and len(g) > 0
    overlap = len(set(g) & set(r)) / max(1, len(set(g) | set(r)))
    return {"prefix_match_tokens": match, "exact_continuation_match": exact, "token_overlap_jaccard": round(overlap, 4)}


def md_code_block(text):
    return "```\n" + text + "\n```"

def run(args):
    device = require_cuda(args.device)
    model, tokenizer, step = load_model_and_tokenizer(args.checkpoint, device)
    context_length = model.config.context_length

    valid_path = ensure_validation_path(args.valid_path)
    records = load_validation_records(valid_path)
    selected, skipped = select_validation_samples(
        records, tokenizer, seed=args.seed, n_per_language=args.n_per_language, min_code_tokens=args.min_code_tokens,
    )

    examples = []
    for rec in selected:
        lang = record_language(rec)
        serialized, ids = tokenize_serialized(rec, tokenizer)
        code_tokens = count_code_tokens(ids)
        prompt_a = build_docstring_prompt(rec)
        prompt_a_ids = tokenizer.encode(prompt_a, add_special_tokens=False).ids
        ref_a_ids = ids[ids.index(CODE_ID) + 1:]
        if ref_a_ids and ref_a_ids[-1] == EOS_ID:
            ref_a_ids = ref_a_ids[:-1]
        try:
            prompt_b_ids, ref_b_ids, code_ids_b, prefix_len = slice_code_prefix(
                ids, min_prefix=args.min_prefix_tokens, max_prefix=args.max_prefix_tokens, min_continuation=args.min_continuation_tokens,
            )
            b_error = None
        except Exception as exc:
            prompt_b_ids, ref_b_ids, code_ids_b, prefix_len = [], [], [], 0
            b_error = str(exc)
        examples.append({
            "record": rec, "lang": lang, "hash": record_hash(rec), "serialized": serialized, "ids": ids,
            "code_tokens": code_tokens, "prompt_a": prompt_a, "prompt_a_ids": prompt_a_ids, "ref_a_ids": ref_a_ids,
            "prompt_b_ids": prompt_b_ids, "ref_b_ids": ref_b_ids, "code_ids_b": code_ids_b, "prefix_len": prefix_len, "b_error": b_error,
        })

    for ex in examples:
        ex["a_skipped_reason"] = None
        try:
            protect_context_length(ex["prompt_a_ids"], context_length)
        except ValueError as exc:
            ex["a_skipped_reason"] = str(exc)
            ex["a_gen_ids"] = []
            ex["a_stopped_eos"] = False
            ex["a_reached_limit"] = False
            ex["a_text"] = ""
        else:
            gen_ids, stopped, reached = generate_greedy(model, device, ex["prompt_a_ids"], args.max_new_tokens, EOS_ID)
            ex["a_gen_ids"] = gen_ids
            ex["a_stopped_eos"] = stopped
            ex["a_reached_limit"] = reached
            ex["a_text"] = tokenizer.decode(gen_ids, skip_special_tokens=False)
            ex["a_diag"] = diagnostic_similarity(gen_ids, ex["ref_a_ids"])

        ex["b_skipped_reason"] = ex.get("b_error")
        if ex["b_skipped_reason"] is None:
            try:
                protect_context_length(ex["prompt_b_ids"], context_length)
            except ValueError as exc:
                ex["b_skipped_reason"] = str(exc)
        if ex["b_skipped_reason"] is None:
            gen_ids, stopped, reached = generate_greedy(model, device, ex["prompt_b_ids"], args.max_new_tokens, EOS_ID)
            ex["b_gen_ids"] = gen_ids
            ex["b_stopped_eos"] = stopped
            ex["b_reached_limit"] = reached
            ex["b_text"] = tokenizer.decode(gen_ids, skip_special_tokens=False)
            ex["b_diag"] = diagnostic_similarity(gen_ids, ex["ref_b_ids"])
        else:
            ex["b_gen_ids"] = []
            ex["b_stopped_eos"] = False
            ex["b_reached_limit"] = False
            ex["b_text"] = ""
            ex["b_diag"] = {}

    from polycodegpt.data.tokenizer_io import TOKENIZER_VERSION, TOKENIZER_SHA256

    lines = []
    lines.append("# Phase 6B — Validation-Set Generation Diagnostic\n")
    lines.append("**Status:** Validation-only inference experiment (not a correctness score)\n")
    lines.append(f"**Checkpoint:** `{args.checkpoint}` (step {step})\n")
    lines.append(f"**Tokenizer:** v{TOKENIZER_VERSION} (sha256 `{TOKENIZER_SHA256}`)\n")
    lines.append(f"**Selection seed:** {args.seed}\n")
    lines.append(f"**Selected:** {len(selected)} ({sum(1 for e in examples if e['lang']=='python')} python, {sum(1 for e in examples if e['lang']=='javascript')} javascript)\n")
    lines.append(f"**Distinct record hashes:** {len({e['hash'] for e in examples})}\n")
    lines.append(f"**Skipped during selection:** {len(skipped)} records\n")
    lines.append(f"**Generation settings:** greedy (temperature=0), max_new_tokens={args.max_new_tokens}\n")
    lines.append("**Data access:** valid.jsonl only.\n")
    lines.append("**Diagnostic measures (definitions):**\n")
    lines.append("- `prefix_match_tokens`: number of leading generated tokens equal to the "
                 "reference tokens before the first mismatch.\n")
    lines.append("- `exact_continuation_match`: true only when the generated token sequence "
                 "equals the reference token sequence (both non-empty).\n")
    lines.append("- `token_overlap_jaccard`: |set(generated) ∩ set(reference)| / "
                 "|set(generated) ∪ set(reference)| over token IDs.\n")
    lines.append("- Empty/nearly empty: <=2 generated tokens before EOS/limit.\n")
    lines.append("- Repetitive: a whitespace 4-gram repeated more than 3 times.\n")
    lines.append("- All of the above are diagnostic token measures, NOT programming-correctness "
                 "scores. Correctness/quality requires human review.\n")
    lines.append("---\n")

    for i, ex in enumerate(examples, start=1):
        lines.append(f"## Example {i} — {ex['lang'].capitalize()} (id {ex['hash']})\n")
        lines.append(f"**Prompt tokens (A):** {len(ex['prompt_a_ids'])} | **Reference code tokens (A):** {len(ex['ref_a_ids'])}\n")
        lines.append("### Experiment A — Docstring-to-code\n")
        lines.append("**Prompt:**\n")
        lines.append(md_code_block(ex["prompt_a"]) + "\n")
        if ex["a_skipped_reason"]:
            lines.append(f"**Skipped:** {ex['a_skipped_reason']}\n")
        else:
            lines.append(f"**Generated ({len(ex['a_gen_ids'])} tokens, {'EOS' if ex['a_stopped_eos'] else 'TOKEN LIMIT'}):**\n")
            lines.append(md_code_block(ex["a_text"]) + "\n")
            lines.append("**Reference code:**\n")
            lines.append(md_code_block(ex["record"].get("code", "")) + "\n")
            d = ex.get("a_diag", {})
            lines.append(f"**Diagnostic:** prefix_match={d.get('prefix_match_tokens')} exact={d.get('exact_continuation_match')} jaccard={d.get('token_overlap_jaccard')}\n")
        lines.append("### Experiment B — Code-prefix continuation\n")
        if ex["b_skipped_reason"]:
            lines.append(f"**Skipped:** {ex['b_skipped_reason']}\n")
        else:
            lines.append(f"**Code tokens (EOS excluded):** {len(ex['code_ids_b'])} | "
                         f"**Prefix tokens:** {ex['prefix_len']} | "
                         f"**Reference continuation tokens:** {len(ex['ref_b_ids'])} | "
                         f"**Prompt tokens:** {len(ex['prompt_b_ids'])}\n")
            prompt_b_text = tokenizer.decode(ex["prompt_b_ids"], skip_special_tokens=False)
            lines.append("**Prompt:**\n")
            lines.append(md_code_block(prompt_b_text) + "\n")
            lines.append(f"**Generated ({len(ex['b_gen_ids'])} tokens, {'EOS' if ex['b_stopped_eos'] else 'TOKEN LIMIT'}):**\n")
            lines.append(md_code_block(ex["b_text"]) + "\n")
            ref_b_text = tokenizer.decode(ex["ref_b_ids"], skip_special_tokens=False)
            lines.append("**Reference continuation:**\n")
            lines.append(md_code_block(ref_b_text) + "\n")
            d = ex.get("b_diag", {})
            lines.append(f"**Diagnostic:** prefix_match={d.get('prefix_match_tokens')} exact={d.get('exact_continuation_match')} jaccard={d.get('token_overlap_jaccard')}\n")
        lines.append("---\n")

    lines.append("## Summary (automated diagnostics)\n")
    a_empty = sum(1 for e in examples if not e.get("a_skipped_reason") and len(e.get("a_gen_ids", [])) <= 2)
    b_empty = sum(1 for e in examples if not e.get("b_skipped_reason") and len(e.get("b_gen_ids", [])) <= 2)
    a_rep = sum(1 for e in examples if not e.get("a_skipped_reason") and is_repetitive(e.get("a_text", "")))
    b_rep = sum(1 for e in examples if not e.get("b_skipped_reason") and is_repetitive(e.get("b_text", "")))
    a_exact = sum(1 for e in examples if e.get("a_diag", {}).get("exact_continuation_match"))
    b_exact = sum(1 for e in examples if e.get("b_diag", {}).get("exact_continuation_match"))
    a_eos = sum(1 for e in examples if e.get("a_stopped_eos"))
    b_eos = sum(1 for e in examples if e.get("b_stopped_eos"))
    lines.append(f"- Examples: {len(examples)}\n")
    a_jac = [e.get("a_diag", {}).get("token_overlap_jaccard", 0.0) for e in examples if not e.get("a_skipped_reason")]
    b_jac = [e.get("b_diag", {}).get("token_overlap_jaccard", 0.0) for e in examples if not e.get("b_skipped_reason")]
    a_mean = round(sum(a_jac) / len(a_jac), 4) if a_jac else 0.0
    b_mean = round(sum(b_jac) / len(b_jac), 4) if b_jac else 0.0
    lines.append(f"- Repetitive generations: A={a_rep}, B={b_rep} (4-gram repeated >3 times; heuristic).\n")
    lines.append(f"- Empty or nearly empty continuations (<=2 tokens): A={a_empty}, B={b_empty}.\n")
    lines.append("- Structurally plausible but semantically incorrect code: not automatically scored; "
                 "manual review of the generated outputs above is required.\n")
    lines.append(f"- Generations resembling their validation references: mean token-overlap Jaccard "
                 f"A={a_mean}, B={b_mean}; exact continuation matches A={a_exact}, B={b_exact}. "
                 "These are diagnostic token measures, not correctness labels.\n")
    lines.append(f"- Experiment A vs B: A stopped at EOS {a_eos} times, B {b_eos} times; "
                 f"A empty={a_empty}/repetitive={a_rep}/exact={a_exact}, "
                 f"B empty={b_empty}/repetitive={b_rep}/exact={b_exact}. "
                 "Differences indicate whether docstring-to-code or code-continuation prompting "
                 "is the harder setting for this checkpoint.\n")
    lines.append("- No generation is labelled correct merely because its tokens resemble the reference.\n")

    if skipped:
        lines.append("\n## Skipped during selection\n")
        for h, reason in skipped[:50]:
            lines.append(f"- {h}: {reason}\n")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Diagnostic report written to {args.report}")
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="PolyCodeGPT validation-set generation diagnostic")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--valid-path", type=Path, default=VALID_PATH, dest="valid_path")
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS, dest="max_new_tokens")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-per-language", type=int, default=N_PER_LANGUAGE, dest="n_per_language")
    parser.add_argument("--min-code-tokens", type=int, default=MIN_CODE_TOKENS, dest="min_code_tokens")
    parser.add_argument("--min-prefix-tokens", type=int, default=MIN_PREFIX_TOKENS, dest="min_prefix_tokens")
    parser.add_argument("--max-prefix-tokens", type=int, default=MAX_PREFIX_TOKENS, dest="max_prefix_tokens")
    parser.add_argument("--min-continuation-tokens", type=int, default=MIN_CONTINUATION_TOKENS, dest="min_continuation_tokens")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except InsufficientEligibleRecordsError as exc:
        raise SystemExit(
            f"STOP: {exc}. The sample size will not be changed and test data will not be used."
        )
