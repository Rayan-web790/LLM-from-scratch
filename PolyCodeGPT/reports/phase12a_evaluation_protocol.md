# Phase 12A — Validation and Development Evaluation Protocol

## Measurements used during development (allowed for selection/tuning)

1. **Phase 10 syntax benchmark** (`reports/phase10_generation_evaluation_*.json`,
   `scripts/evaluate_phase10_benchmark.py`). Metric: syntax-valid rate per language.
   Used as a fast, cheap development signal.
2. **Phase 11 functional benchmark** (`reports/phase11_functional_results.json`,
   `functional/evaluate_functional.py`). Metrics per language: syntax-valid rate,
   functional success rate, test-case pass rate, runtime-error rate, timeout rate,
   completion rate, edge-case success rate. Used as the development functional signal.
3. **Validation loss** on the fixed validation split
   (`data/processed/polycodegpt_corpus/valid.jsonl`, 64 blocks, batch size 8),
   recorded at every 500 steps. Used for early stopping and checkpoint selection
   among the four branches.

## Independent final benchmark (held out)

A separate 100-task functional benchmark (50 Python, 50 JavaScript) is designed in
`reports/phase12a_independent_benchmark_design.md`. It is **not** used for any
training decision: not for model selection, not for learning-rate tuning, not for
early stopping, not for checkpoint selection. It is evaluated exactly once per
branch after training completes, by a reviewer who did not tune the branch.

## Contamination checks (run before any benchmark is scored)

For every candidate benchmark task, reject it if any of the following hold
against the training corpus of the branch being evaluated:

- exact match on the normalised code hash;
- 5-gram token shingle Jaccard >= 0.8 against any training record;
- function name (normalised leaf) appears in the training corpus for the same
  language;
- source repository appears in the training corpus.

Every rejection is logged with the offending training record id. A branch whose
benchmark fails any of these checks is not scored; the benchmark is rebuilt.

## Constraints honoured

- The old test split (`data/processed/polycodegpt_corpus/test.jsonl` and the
  per-language `test.jsonl` files) is never opened by this protocol.
- The validation split is read-only; it is used for loss only, never for
  training or for benchmark construction.
- Phase 10/11 benchmarks are development measurements only; they are not
  training data and are not used to filter the training corpus.
