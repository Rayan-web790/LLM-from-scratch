# Phase 12B — Final Implementation Gate

No training was started. No checkpoint, dataset, tokenizer or prior report was
modified. Original checkpoint `step_008130.pt` verified read-only (sha256
`5562acc2…90bd`, step 8130) by the runner.

## Production runner design

`scripts/run_phase12_experiment.py`:
- Loads exactly one experiment YAML (`--experiment` or `--config`).
- Validates top-level keys against a strict allowlist; rejects unknown keys.
- Verifies base checkpoint existence, SHA-256 and step (8130).
- Resolves token budgets, optimizer/scheduler/batch settings, data sources.
- `--dry-run` prints fully resolved settings and exits without writing files.
- Non-dry-run prepares `dataset_manifest.json` only in a separate output directory.
- Never writes to the original checkpoint path.

## Configuration-to-engine mapping

| YAML section | Engine setting |
|---|---|
| experiment.id | experiment ID |
| experiment.seed | seed |
| data.token_budget_per_language | Python/JS token budgets |
| data.english_doc_tokens | English doc tokens (2,847,020) |
| data.token_budget_total | total (133,192,957) |
| data.curated_filter | JS data source (original/curated) |
| data.oversampling / javascript_sampling | sampling policy |
| training.learning_rate | lr base 3e-4 |
| training.gradient_clip | 1.0 |
| training.micro_batch_size | 4 |
| training.gradient_accumulation_steps | 8 |
| training.context_length | 512 |
| training.max_steps | 16260 (expected final global step) |
| training.additional_steps | 8130 |
| schedule.total_steps | 8130 (scheduler horizon) |
| schedule.warmup_steps | 163 |
| schedule.min_lr_ratio | 0.1 |
| schedule.reset_schedule | true (fresh cosine) |
| schedule.reset_optimizer_step | false (AdamW moments preserved) |

## Dry-run output summary (all four branches)

| branch | JS source | Python tokens | JS tokens | English | total | additional updates | expected final |
|---|---|---:|---:|---:|---:|---:|---:|
| control | original | 98,807,136 | 31,538,801 | 2,847,020 | 133,192,957 | 8,130 | 16,260 |
| exp_a | curated | 98,807,136 | 31,538,801 | 2,847,020 | 133,192,957 | 8,130 | 16,260 |
| exp_b | original | 65,172,968 | 65,172,969 | 2,847,020 | 133,192,957 | 8,130 | 16,260 |
| exp_c | curated | 65,172,968 | 65,172,969 | 2,847,020 | 133,192,957 | 8,130 | 16,260 |

Scheduler: fresh cosine continuation, warmup 163, total_steps 8130, min_lr_ratio 0.1.
LR first 5 updates: ~1.84e-6 … 9.20e-6 (warmup). LR last 5: ~3.0e-5 (min LR).
Optimizer: AdamW moments preserved from checkpoint 8130.
Batch: micro 4 x accum 8 = 32 (16,384 tokens/step). Grad clip 1.0. Context 512.
Seed 42, deterministic true.

## Checkpoint hash verification

`step_008130.pt` sha256 = `5562acc2fd58837f9d4245042b7f0b10de65520aa2ba292999c80045470590bd`
step = 8130. Verified read-only by the runner for all four branches.

## Independent benchmark v3 limitations

- 100 tasks (50 Python, 50 JavaScript), 328 cases.
- Validated in sandbox: 100/100 tasks, 328/328 cases.
- Contamination check: `contamination_found: false`.
- Frozen with SHA-256 in `functional/independent_v3/freeze_manifest.json`.
- Not run against any model; reserved for Phase 14.
- v2 preserved and labelled superseded/provisional.

## Tests and commands

```
python scripts/run_phase12_experiment.py --experiment control --dry-run
python scripts/run_phase12_experiment.py --experiment exp_a --dry-run
python scripts/run_phase12_experiment.py --experiment exp_b --dry-run
python scripts/run_phase12_experiment.py --experiment exp_c --dry-run
python experiments/phase12b/validate_config.py configs/experiments/control.yaml
python functional/independent_v3/spec/build_manifest.py --check
python functional/independent_v3/reference/validate_references.py
python experiments/phase12b/contamination_check_v3.py
```

All pass.

## Unresolved issues

- Experiment YAMLs use nested structure; runner consumes them but a future
  production loader should replace the ad-hoc mapping.
- `max_steps` (16260) vs `total_steps` (8130) engine constraint must be
  reconciled before launch (engine validates max_steps <= total_steps).
- Deterministic-algorithm flags not encoded in YAML; run entry point must set them.
- Curated dataset `this`/import/export exclusion may be over-restrictive.
- No actual training performed in Phase 12B.
