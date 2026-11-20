# Phase 10: Baseline Generation Evaluation and Inference Optimization

## Project Context

This phase builds on the completed Phase 9 integrity and repetition audit. The PolyCodeGPT model is at checkpoint `step_008130.pt` with tokenizer v1.1.0. All binary artifacts (`valid.bin`, `train.bin`, `test.bin`) are verified immutable. No further training, checkpoint modification, or test data access will be performed.

The Phase 9 audit established that:
- `valid.bin` SHA-256: `ee55019be57e1b53a0a74b232ae6d37b5e2935c4b6af95f70221e714fa7e52df` (matches manifest and Phase 5A)
- Training token accounting: 130,345,937 code + 2,847,020 docs = 133,192,957 total
- BPE-based unique-token ratio < 0.30 flags 41,403 records (10.54%), not 82,230 (20.94% from CodeSearchNet-token analysis)
- Flagged records have median BPE code length 689 vs. 152 for unflagged (4.5x difference)
- 11.96x group-median length disparity and 11.8x pairwise-median disparity make length-controlled comparison infeasible
- Conclusions correctly distinguish correlation from causation

## 1. Preserve Immutable Baseline

- **Checkpoint**: `checkpoints/polycodegpt_first_pass/step_008130.pt` — treated as immutable baseline
- **Never** overwrite, re-weight, or fine-tune this checkpoint
- **Hash for provenance**: Record SHA-256 of step_008130.pt in the Phase 10 report
- All generation benchmarks reference this exact checkpoint; any alternative checkpoint is a separate experiment, not a Phase 10 variant

## 2. Fixed Python and JavaScript Generation Benchmarks

Two static benchmark suites, each with 50 prompts, fixed across all evaluation runs:

### Python Benchmark (50 prompts)
- Topics: factorial, Fibonacci, palindrome check, merge sorted lists, binary search, BFS/DFS, string reversal, anagram check, prime sieve, matrix multiplication
- Prompt format: `<|lang:python|><|docstring|>{docstring}<|code|>{code_stub}<|eos|>`
- Continuation length: max_new_tokens=64
- Deterministic: temperature=0, no top-k/top-p filtering

### JavaScript Benchmark (50 prompts)
- Topics: factorial, Fibonacci, palindrome check, merge sorted arrays, binary search, BFS/DFS, string reversal, anagram check, prime sieve, function composition
- Prompt format: `<|lang:javascript|><|docstring|>{docstring}<|code|>{code_stub}<|eos|>`
- Continuation length: max_new_tokens=64
- Deterministic: temperature=0

**Prompt examples** (the full 50+50 set is fixed and version-controlled):

| # | Language | Docstring | Code Stub |
|---|----------|-----------|-----------|
| 1 | Python | Compute the factorial of n recursively. | def factorial(n): |
| 2 | Python | Return the nth Fibonacci number. | def fibonacci(n): |
| 3 | Python | Check whether a string is a palindrome. | def is_palindrome(s): |
| 4 | Python | Merge two sorted lists into one sorted list. | def merge_sorted(a, b): |
| 5 | JavaScript | Compute the factorial of n recursively. | function factorial(n) { |
| 6 | JavaScript | Return the nth Fibonacci number. | function fibonacci(n) { |
| ... | ... | ... | ... |

## 3. Prompt Format and Decoding Parameters Evaluation

### Serialization Format (verified in Phase 9)
`<|lang:X|><|docstring|>{docstring}<|code|>{code}<|eos|>`
- Language marker: `<|lang:python|>` or `<|lang:javascript|>`
- Docstring marker: `<|docstring|>`
- Code marker: `<|code|>`
- EOS marker: `<|eos|>`
- Code IDs are measured between `<|code|>` and `<|eos|>`

### Decoding Parameters to Evaluate
| Parameter | Values | Purpose |
|---|---|---|
| temperature | 0 (greedy), 0.8, 1.0 | Greedy for benchmark; higher values for diversity analysis |
| max_new_tokens | 32, 64, 128 | Short, medium, long continuation |
| do_sample | True/False | Switch between sampling and greedy |

**Only report**: temperature=0, max_new_tokens=64 as the canonical benchmark. Other parameters are auxiliary.

## 4. Evaluation Metrics

For each generated continuation, compute the following metrics, applied to the **code portion** (between `<|code|>` and `<|eos|>`):

1. **Syntax validity**: Can the continuation be parsed as valid Python (using `ast.parse`) or valid JavaScript (using `eval` in a sandbox, or `acorn`/`esprima` parsing)? Report pass/fail ratio.

2. **Repetition metrics** (BPE token IDs, PolyCodeGPT v1.1.0):
   - **Unique-token ratio**: fraction of distinct tokens in the code sequence
   - **Repeated-line ratio**: fraction of lines that appear more than once
   - **Repeated 4-gram ratio**: fraction of overlapping 4-token sequences that appear multiple times
   - Threshold flag: UR < 0.30 → "repetitive"

3. **Completeness**: Does the generation finish with `<|eos|>`? Does it truncate abruptly? Report:
   - EOS hit ratio
   - Average tokens generated before first repetition
   - Completion ratio (generation produces a function/class with body, not just a signature)

4. **Functional correctness**: For a subset of 10 prompts per language, execute the generated code:
   - **Python**: `python -m py_compile` or `python -c "exec(gen)"` — does it compile/run without error?
   - **JavaScript**: Node.js `node -e "..."` — does it execute without error?
   - Record: passes/fails, error type (syntax error, runtime error, undefined variable)

**Important**: Functional correctness runs are limited to the 20 prompts (10 per language). The remaining 80 prompts are evaluated for syntax validity, repetition, and completeness only. Do not attempt GPU-heavy execution suites.

## 5. Fair Comparison Methodology for Future Checkpoints

Any future checkpoint comparison must adhere to these constraints:

- **Same tokenizer**: v1.1.0 only; do not compare across tokenizer versions
- **Same serialization format**: `<|lang:X|><|docstring|>{docstring}<|code|>{code}<|eos|>`
- **Same prompt set**: the fixed 100-prompt benchmark (50 Python + 50 JavaScript)
- **Same decoding**: temperature=0, max_new_tokens=64 (or explicitly state deviations)
- **Same metric calculations**: unique-token ratio, repeated 4-gram ratio, syntax validity, functional correctness on the 20-hard-coded prompts
- **No test data access**: use the validation partition (`valid.jsonl` / `valid.bin`) only
- **Report the baseline**: always reference step_008130.pt as the immutable baseline; any improvement claimed must be relative to this checkpoint's metrics

### Comparison Table Template

| Checkpoint | UR (flagged %) | R4 median | Syntax validity % | Functional correctness % | Notes |
|---|---|---|---|---|---|
| step_008130.pt (baseline) | 0.2668 / 0.4961 | 0.2342 / 0.0422 | TBD (Phase 10) | TBD (Phase 10) | Immutable; no GPU generation needed for report |
| step_XXXXX.pt | ... | ... | ... | ... |  |

## 6. Validation Data Only — No Test Data Access

- **Use only**: `data/processed/polycodegpt_corpus/valid.jsonl` and `data/processed/tokenized_v1_1/valid.bin`
- **Never**: open, reference, or modify `test.bin` or `data/manifests/codesearchnet_*`
- **Justification**: Phase 9 already confirmed test.bin was not accessed; Phase 10 maintains this discipline
- All metrics, ablations, and comparisons draw from the validation partition exclusively

## 7. Lightweight Evaluation Tooling and Unit Tests

### Evaluation Script (no GPU required when `--verify-only`)
- `scripts/evaluate_generation_benchmark.py` — reads the fixed 100 prompts, runs generation at temperature=0, computes all metrics
- `--verify-only` flag: performs reconstruction + distribution checks without model generation (lightweight)
- Output: JSON report written to `reports/phase10_generation_evaluation_<timestamp>.json`

### Unit Tests (existing test suite integration)
Add to `tests/unit/`:
- `test_phase10_generation_benchmark.py` — validates the benchmark script produces expected output structure
- `test_phase10_metrics.py` — validates metric calculations against known reference values from Phase 9
- `test_phase10_serialization.py` — validates the `<|lang:X|><|docstring|>{docstring}<|code|>{code}<|eos|>` format parsing

Run: `python -m pytest tests/unit/test_phase10_*.py -q`

### Artifacts produced (all read-only, no dataset modification):
- `reports/phase10_generation_evaluation_<timestamp>.json`
- `reports/phase10_plan.md` (this document)
- Updated `reports/phase10_generation_benchmark.md` (optional qualitative appendix)

## 8. No GPU-Intensive Experiments

- **Generation**: temperature=0, max_new_tokens=64 — on CPU, this typically completes within 2-5 minutes per prompt on a modern CPU. Total: ~5-8 hours for 100 prompts.
- **If CPU time is prohibitive**: reduce to 50 prompts (25 Python + 25 JavaScript) or report metrics on a deterministic subset.
- **Functional correctness**: only on the 20 hard-coded prompts (10 per language). Do not scale.
- **No distributed training, no LoRA fine-tuning, no checkpoint comparison beyond the fixed benchmark**.

## 9. Deliverables

### 9.1 `reports/phase10_plan.md`
- This document, saved permanently.

### 9.2 `reports/phase10_generation_evaluation_<timestamp>.json`
- Empirical results from the benchmark run.
- Structure (example):
```json
{
  "checkpoint": "step_008130.pt",
  "timestamp": "2026-11-17T22:30:00",
  "python": {
    "prompts": 50,
    "syntax_validity": 0.74,
    "ur_median_flagged": 0.2668,
    "ur_unflagged_median": 0.4961,
    "r4_median_flagged": 0.2342,
    "r4_median_unflagged": 0.0422,
    "eos_hit_ratio": 0.96,
    "functional_correctness": 0.68,
    "sample_outputs": [...]
  },
  "javascript": {
    "prompts": 50,
    "syntax_validity": 0.62,
    "ur_median_flagged": 0.2668,
    "ur_unflagged_median": 0.4961,
    "r4_median_flagged": 0.2342,
    "r4_median_unflagged": 0.0422,
    "eos_hit_ratio": 0.94,
    "functional_correctness": 0.55,
    "sample_outputs": [...]
  },
  "notes": "Functional correctness limited to 20 prompts (10 per language). Remaining 80 evaluated for syntax validity and repetition only."
}
```

### 9.3 `scripts/evaluate_generation_benchmark.py` (new, read-only)
- The evaluation driver script.

### 9.4 Unit tests in `tests/unit/test_phase10_*.py` (new)
- Lightweight tests validating script behavior and metric integrity.

## 10. What Phase 10 Does NOT Do

- ❌ Do not modify `step_008130.pt` or any checkpoint
- ❌ Do not retrain or fine-tune the model
- ❌ Do not modify the tokenizer (`tokenizer/artifacts/polycodegpt_tokenizer_v1_1.json`)
- ❌ Do not modify training configuration or data manifests
- ❌ Do not access `test.bin` or any test partition data
- ❌ Do not claim SOTA or benchmark-leading results; this is a lightweight integrity and evaluation audit
- ❌ Do not report results beyond the fixed 100-prompt benchmark + 20 functional correctness prompts
- ❌ Do not vary the prompt set, decoding parameters, or serialization format and claim comparability to the baseline