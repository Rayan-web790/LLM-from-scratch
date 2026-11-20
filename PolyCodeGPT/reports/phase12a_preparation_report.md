# Phase 12A — Preparation Report (JavaScript Training Data Improvement)

Phase 11 (functional evaluation of checkpoint 8130) is complete and untouched.
This phase prepares the next round of experiments without retraining, without
modifying checkpoint 8130, the tokenizer, the original datasets, or the old test
split. No experiment has been launched.

## Part A — Baseline preserved

- `reports/phase12a_baseline.json` / `.md`: 63 protected files hashed (checkpoints,
  tokenizer artifacts, data manifests, configs, reports).
- Checkpoint `step_008130.pt` (sha256
  `5562acc2fd58837f9d4245042b7f0b10de65520aa2ba292999c80045470590bd`) verified:
  step == 8130, tokenizer hash matches, data-manifest hash matches, model state
  loads strictly, optimizer state restores (moments for every tensor, single step
  counter, both decay/no-decay groups), scheduler state restores, RNG + loader
  epoch state present, token accounting consistent (partial final group of 14).
- Optimizer state is restorable consistently from a copy; the original checkpoint
  is opened read-only in every path.
- Result: **8/8 checks PASS**.

## Part B/C — JavaScript data audit and curated candidate set

Source: `data/processed/polycodegpt_corpus/train.jsonl`, language == javascript,
training split only (102,169 records). Syntax was validated with V8 compilation
(`new vm.Script` / `new vm.SourceTextModule`); **no corpus code was executed**.

- Original: 102,169 records (14,738 repos).
- Eligible (curated): **35,658** (34.9%).
- Rejected: 66,511, by reason:

| reason | count |
|---|---:|
| parse_invalid | 32,830 |
| stateful_or_method | 11,253 |
| description_fragment | 12,309 |
| depends_on_ecosystem | 6,679 |
| code_too_long | 2,399 |
| description_boilerplate | 367 |
| not_standalone_function | 497 |
| name_mismatch | 105 |
| near_duplicate | 72 |
| duplicate | 0 |
| description_missing | 0 |
| code_too_short | 0 |

- Curated output: `data/curated/js_v1/train.jsonl` (35,658 records, sha256
  `b0ae31c0d175...`, all fields preserved verbatim, all partitions "train").
- Exclusions: `data/curated/js_v1/rejected.csv` (66,511 rows + header).
- Manifests: `data/curated/js_v1/manifest.json`, `reports/phase12a_js_audit.json` /
  `.md`, `reports/phase12a_curated_dataset.md`.
- Distinct repos: 14,738 -> 8,341. Serialized tokens: median 178 -> 154; p95 816 -> 402.

## Part D — Experiment branches (planned, not run)

Configs in `configs/experiments/`:

| branch | config | JS data | language ratio |
|---|---|---|---|
| control | `control.yaml` | original (102,169) | original (~77/23) |
| A | `exp_a_curated_js_ratio.yaml` | curated, oversampled to original volume | original (~77/23) |
| B | `exp_b_language_balance.yaml` | original, oversampled | balanced 50/50 |
| C | `exp_c_curated_balanced.yaml` | curated, oversampled | balanced 50/50 |

All four use checkpoint 8130, AdamW, lr 3e-4, cosine schedule 8,129 steps,
micro-batch 4 x 8, context 512, seed 42, token budget 133,192,704.
Separate proposal (not a branch): `proposal_context_window_1024.yaml`.

## Part E — Evaluation

- Development protocol: `reports/phase12a_evaluation_protocol.md`.
- Independent final benchmark design: `reports/phase12a_independent_benchmark_design.md`.

## Part F — Resource estimates

- `reports/phase12a_resource_estimates.md` (all figures marked as estimates).

## Constraints checklist

- No retraining: yes (configs only).
- Checkpoint 8130 unmodified: yes (sha256 verified, read-only).
- Original datasets unmodified: yes.
- Tokenizer unmodified: yes (sha256 verified).
- Old test split not accessed: yes (never opened).
- No generated code executed outside a sandbox: yes (V8 compile only; no execution).
- Existing reports preserved: yes (only `phase12a_*` files added).
- Experiments not launched: yes.

## Review commands (run from `PolyCodeGPT/`)

```bash
# Re-verify the baseline hashes and checkpoint restore
../.venv/Scripts/python experiments/phase12a/baseline_manifest.py

# Re-run the JS audit (writes the curated dataset)
../.venv/Scripts/python experiments/phase12a/audit_js.py
# Audit only, without writing the curated dataset:
../.venv/Scripts/python experiments/phase12a/audit_js.py --dry-run

# Verify the curated dataset (counts, hashes, field equality, partitions)
../.venv/Scripts/python - <<'PY'
import json, hashlib, pathlib
cur = pathlib.Path("data/curated/js_v1/train.jsonl")
lines = [l for l in cur.read_text(encoding="utf-8").splitlines() if l.strip()]
orig = {}
with open("data/processed/polycodegpt_corpus/train.jsonl", encoding="utf-8") as f:
    for l in f:
        r = json.loads(l)
        if r.get("language") == "javascript":
            orig[r["record_id"]] = r
ids = [json.loads(l)["record_id"] for l in lines]
print("records:", len(lines), "unique:", len(set(ids)))
print("all in train:", all(i in orig for i in ids))
print("field mismatches:", sum(json.loads(l) != orig[json.loads(l)["record_id"]] for l in lines))
print("partitions:", set(json.loads(l).get("global_partition") for l in lines))
m = json.load(open("data/curated/js_v1/manifest.json", encoding="utf-8"))
print("sha ok:", hashlib.sha256(cur.read_bytes()).hexdigest() == m["output_sha256"])
PY

# Inspect experiment configs
ls configs/experiments/

# Re-run the Phase 11 test suite (sandbox + spec + evaluator)
../.venv/Scripts/python -m pytest tests/unit/test_functional_spec.py \
  tests/unit/test_functional_evaluator.py tests/unit/test_sandbox_runner.py \
  tests/integration/test_phase11_sandbox.py -q
```

## Implemented vs incomplete

Implemented in Phase 12A:
- Baseline freeze + verification (Part A).
- JS audit with V8 compile-only validation (Part B).
- Curated candidate dataset + filtering manifest + rejected list (Part C).
- Four experiment configs + context-window proposal (Part D).
- Evaluation protocol + independent benchmark design (Part E).
- Resource estimates (Part F).
- This report and review commands (Part G).

Deferred to later phases (intentionally not started):
- Any training run (all branches remain `planned_not_run`).
- Implementation of the independent final benchmark (design only).
- Context-window experiment (proposal only).
- Contamination-check automation (procedure documented; script deferred).
