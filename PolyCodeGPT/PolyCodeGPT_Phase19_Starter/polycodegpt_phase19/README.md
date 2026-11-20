# PolyCodeGPT Phase 19 — Scenario B pilot

This package adds a separate Phase 19 runner. It does not modify `scripts/train.py`,
`configs/model.yaml`, `configs/training.yaml`, the v1.1 tokenizer, or previous runs.

## Install the files

Copy `scripts/train_phase19.py` into the project's `scripts/` directory and copy
both YAML files into the project's `configs/` directory.

## Run a read-only preflight first

From the project root:

```cmd
python scripts/train_phase19.py --preflight-only
```

The preflight verifies the Phase 18 Scenario B verdict, v2 tokenizer hash and
special-token IDs, Scenario B binary hashes and token counts, all token IDs'
vocabulary range, and context-window sampling. It does not initialize a model
or open any test partition.

## Start the 200-step pilot

```cmd
python scripts/train_phase19.py
```

The runner performs a one-update CUDA memory probe, selecting the largest
configured micro-batch among 4, 2, and 1 that can complete a real optimizer
update. It preserves an effective batch of 32 sequences by adjusting gradient
accumulation. Training uses a fixed 256-block validation subset.

Optional step override:

```cmd
python scripts/train_phase19.py --max-steps 50
```

## Output isolation

Checkpoints: `checkpoints/polycodegpt_phase19_scenario_b/`

Logs: `logs/phase19_scenario_b/`

The run summary records dataset/tokenizer hashes, model/training settings,
validation history, memory probe results, checkpoint reload checks, and an
explicit `test_partition_opened: false` field.

## Important notes

- The pilot trains a randomly initialized v2 model; it does not resume v1.1.
- Scenario B is the only dataset used for training and validation.
- The validation metric is measured on a deterministic subset of 256 blocks,
  not the entire validation split.
- The script assumes the Phase 18 files and reports exist at their standard
  project paths and that the current project environment has its normal
  dependencies installed.
