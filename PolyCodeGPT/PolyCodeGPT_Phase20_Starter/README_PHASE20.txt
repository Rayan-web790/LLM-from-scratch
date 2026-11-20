PolyCodeGPT Phase 20 — Scenario B continuation starter

CONTENTS
  scripts/train_phase20.py
  configs/phase20_training.yaml

COPY INTO THE EXISTING PROJECT ROOT
  scripts/train_phase20.py          -> <project>\scripts\train_phase20.py
  configs/phase20_training.yaml     -> <project>\configs\phase20_training.yaml

Run all commands from:
  C:\Users\UltraPc\Desktop\LLM FROM SCRATCH\PolyCodeGPT

1) Preflight only (does not train):
  .venv_phase19\Scripts\python.exe scripts\train_phase20.py --preflight-only

2) Two-update restoration smoke test (uses separate smoke-test outputs):
  .venv_phase19\Scripts\python.exe scripts\train_phase20.py --smoke-test

3) Full one-pass continuation (only after smoke test passes):
  .venv_phase19\Scripts\python.exe scripts\train_phase20.py

4) Resume an interrupted Phase 20 run:
  .venv_phase19\Scripts\python.exe scripts\train_phase20.py --resume-phase20

IMPORTANT
- The Phase 19 checkpoint source is checkpoints\polycodegpt_phase19_scenario_b\latest.pt.
- The script validates the tokenizer and Scenario B manifest hashes before restore.
- It does not open the test partition.
- Phase 20 outputs go to checkpoints\polycodegpt_phase20_scenario_b and logs\phase20_scenario_b.
- Smoke-test outputs go to separate *_phase20_smoke locations.
- Phase 19 files are not intentionally modified by this script.
- Review the preflight and smoke-test output before launching the full pass.
- The full run writes full-size checkpoints every 1,000 global steps. Ensure sufficient free disk space.
