# Phase 15 — PolyCodeGPT experimental release packaging preflight

Strictly read-only preflight. No package was created, no checkpoint was copied or modified, no source artifact was modified, and no generated candidate code was executed. This report is the only new file.

Selected checkpoint: `experiments/phase12b/runs/exp_c/step_016260.pt`
Expected checkpoint SHA256: `cf0544099d73f85cdeb40bc1e0a93ce17bb790db4cb7136254be65d16c3b45b4`

## 1. Source artifacts preserved and verified

| Source artifact | SHA256 / size | Status |
|---|---:|---|
| `experiments/phase12b/runs/exp_c/step_016260.pt` | `cf0544099d73f85cdeb40bc1e0a93ce17bb790db4cb7136254be65d16c3b45b4` / 207,215,053 bytes | Verified, unchanged |
| `tokenizer/artifacts/polycodegpt_tokenizer_v1_1.json` | `0459f8556a6f79ca1ba1c3dc3f76c26ae2e55653ebbf97beaa9bd23467358a03` / 1,095,741 bytes | Verified, unchanged |
| `tokenizer/artifacts/tokenizer_metadata_v1_1.json` | metadata artifact / 41,093 bytes | Verified present |
| `configs/model.yaml` | `8b960967dea848ebb4cd428c48ad4155dc72c1c13e9f18aa23c54a22035ec8c2` / 686 bytes | Verified |
| `reports/phase12c_exp_c_audit.md` | `d589081edd250cbd17928db5c04399571c2be21f6070429280fdb4f6a305a80a` / 8,729 bytes | Provenance source |
| `reports/phase13_final_comparison.md` | `1f3fd5d533f3cf11fed00952bef1d92cc6acffc6c68e8e90c9d491e595081ea0` / 13,427 bytes | Provenance source |
| `reports/phase14_final_audit.md` | `24ed2aa8564b4962569a12fb9be4ab9dfd1f78696a224dda5ec7d38e1d103f2c` / 10,585 bytes | Provenance source |
| `reports/phase14_candidate_integrity.md` | `5c5903e8e0b83769d47a7ad836bcddb72e7fafff76be4739307ec8e2` / 6,477 bytes | Provenance source |

The checkpoint payload was loaded read-only and verified:

- `format_version`: 1
- `step`: 16260
- `stage_step`: 8130
- `initial_global_step`: 8130
- `tokens_seen`: 266,385,408
- `parameter_count`: 16,988,160
- `weights_tied`: true
- `precision`: bf16
- `best_val_loss`: 2.319764374755323 at step 16260
- `tokenizer_sha256`: `0459f8556a6f79ca1ba1c3dc3f76c26ae2e55653ebbf97beaa9bd23467358a03`
- `data_manifest_sha256`: `367636ca923c8e9b6d0693a7606ae06504e76067908c686bcbf0e746508f5544`

The checkpoint is a full training checkpoint. It contains optimizer, scheduler, scaler, RNG, metrics, and validation state in addition to `model_state`. The source checkpoint must remain untouched; the package may either copy it as-is or later derive an inference-only checkpoint from it after a separate explicit packaging step.

## 2. Verified inference and loading requirements

### Actual supported loading APIs

Two loading paths exist in the repository:

1. Training-checkpoint path:
   - `polycodegpt.training.checkpoint.load_checkpoint(path, map_location=...)`
   - `polycodegpt.training.rebuild_model(payload, device=...)`
   - This is the path that succeeds for `step_016260.pt`.
2. Model-checkpoint path:
   - `polycodegpt.model.checkpoint.load_checkpoint(path, device=...)`
   - Returns `(state_dict, ModelConfig, metadata)`.
   - This path is for the older `save_checkpoint` format and is not the format of the selected checkpoint.

For a self-contained package, the direct and minimal loading sequence is:

```python
import torch
from polycodegpt.model.config import ModelConfig
from polycodegpt.model.gpt import GPT

payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
config = ModelConfig.from_dict(payload["model_config"])
model = GPT(config)
model.load_state_dict(payload["model_state"])
if config.tie_embeddings:
    model.lm_head.weight = model.token_embedding.weight
model = model.to(device).eval()
```

This sequence was executed read-only and successfully rebuilt the selected model on CPU. The resulting model has 16,988,160 parameters and verified weight tying.

### Tokenizer requirements

- Library: `tokenizers` 0.22.2.
- Artifact: `polycodegpt_tokenizer_v1_1.json`.
- Vocabulary size: 16,000.
- Special-token contract: 11 tokens, IDs 0–10.
- EOS token: `<|eos|>`, ID 2.
- Language markers:
  - `<|lang:python|>` → ID 6
  - `<|lang:javascript|>` → ID 7
- Loading API: `polycodegpt.data.tokenizer_io.load_tokenizer(path, expected_sha256=..., vocab_size=...)`.
- The tokenizer must be verified by SHA256 before use.

### Model configuration

| Parameter | Value |
|---|---:|
| Architecture | `decoder_only_transformer` |
| Vocabulary size | 16,000 |
| Model dimension | 384 |
| Layers | 6 |
| Attention heads | 6 |
| FFN dimension | 1,536 |
| Context length | 512 |
| Dropout | 0.1 |
| Tie embeddings | true |
| LayerNorm epsilon | 1e-5 |
| Initialization std | 0.02 |

The configuration is stored both in `configs/model.yaml` and in the checkpoint's `model_config` field. They agree.

### Device selection and CPU/GPU behavior

- Verified environment: Python 3.11.9, PyTorch 2.15.0+cu132, CUDA 13.2, on NVIDIA GeForce RTX 5060 Laptop GPU.
- GPU loading and inference are supported and were used throughout the project.
- CPU loading was verified in this preflight.
- CPU inference is expected to work through `.to("cpu")`, but it is not performance-tested.
- `polycodegpt.training.require_cuda` refuses CPU and must not be used as the package's inference device resolver. The package must implement explicit `--device {cpu,cuda}` validation.
- `GPT.generate` temporarily switches the model to eval mode and restores the prior train/eval state.

### Generation interface and default decoding settings

Actual API:

```python
model.generate(
    input_ids,
    max_new_tokens=50,
    temperature=1.0,
    top_k=None,
    eos_token_id=None,
)
```

Important behavior:

- `input_ids` must be 2-D `[batch, sequence]`.
- `temperature == 0` uses `torch.argmax` (greedy decoding).
- For `temperature != 0`, optional `top_k` filtering and `torch.multinomial` sampling are used.
- Generation stops when `eos_token_id` is provided and all batch positions emit EOS.
- Output includes the original prompt tokens followed by generated tokens.
- Context is truncated to the last `context_length` tokens during generation.
- The project's qualitative script uses greedy decoding with `temperature=0`, `max_new_tokens=128`, and EOS ID 2.
- The Phase 10/11/14 benchmarks use greedy decoding with `temperature=0`, `max_new_tokens=64`, and no EOS-appended prompt.

For reproducible experimental inference, use `temperature=0.0` and a fixed `max_new_tokens`.

### Required source modules

Minimal inference package should include:

- `polycodegpt/__init__.py`
- `polycodegpt/model/__init__.py`
- `polycodegpt/model/config.py`
- `polycodegpt/model/gpt.py`
- `polycodegpt/model/layers.py`
- `polycodegpt/model/attention.py`
- `polycodegpt/data/__init__.py`
- `polycodegpt/data/tokenizer_io.py`
- `polycodegpt/data/integrity.py`
- `polycodegpt/tokenization/__init__.py`
- `polycodegpt/tokenization/serializer.py`
- A new minimal inference entry point, for example `polycodegpt/inference.py` and `polycodegpt/__main__.py`

The training engine, optimizer, scheduler, evaluation, and data pipeline modules are not required for inference. Including them would add unnecessary dependencies such as training-only code paths.

### Runtime assumptions and environment variables

- No required environment variables.
- `apply_determinism` in training sets optional CUDA determinism variables, but these are not required for inference.
- No network access is required at runtime.
- The tokenizer artifact must be present locally.
- The checkpoint must be present locally.
- Generated code must not be executed outside the existing Docker sandbox.

### Prompt serialization format

The model expects the project's serialized prompt format:

```text
<|lang:python|><|docstring|>{docstring}<|code|>{code_stub}
```

or

```text
<|lang:javascript|><|docstring|>{docstring}<|code|>{code_stub}
```

The corresponding tokenizer special-token IDs are 6 for Python and 7 for JavaScript. French and Arabic markers exist in the tokenizer, but this model was not trained on French or Arabic content and no French/Arabic capability should be claimed.

## 3. Proposed package contents and layout

Proposed package root:

```text
C:\Users\UltraPc\Desktop\LLM FROM SCRATCH\PolyCodeGPT\release\polycodegpt-exp-c-0.1.0\
```

Proposed layout:

```text
polycodegpt-exp-c-0.1.0/
├── README.md
├── pyproject.toml
├── requirements.txt
├── hashes.sha256
├── polycodegpt/
│   ├── __init__.py
│   ├── __main__.py
│   ├── inference.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── gpt.py
│   │   ├── layers.py
│   │   └── attention.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── tokenizer_io.py
│   │   └── integrity.py
│   └── tokenization/
│       ├── __init__.py
│       └── serializer.py
├── models/
│   └── exp_c_step_016260.pt
├── tokenizer/
│   ├── polycodegpt_tokenizer_v1_1.json
│   └── tokenizer_metadata_v1_1.json
├── configs/
│   └── model.yaml
├── examples/
│   ├── generate_python.py
│   └── generate_javascript.py
├── docs/
│   ├── limitations.md
│   └── benchmark_results.md
└── provenance/
    ├── phase12c_exp_c_audit.md
    ├── phase13_final_comparison.md
    ├── phase14_candidate_integrity.md
    └── phase14_final_audit.md
```

The package should include only files required to reproduce inference and document the experimental result. It must not include training datasets, private environment files, secrets, logs, caches, temporary artifacts, or the old test partition.

Estimated package size: approximately 199.3 MB, dominated by the 207,215,053-byte checkpoint.

## 4. Required files and hashes

| Package path | Source path | SHA256 | Size |
|---|---|---|---:|
| `models/exp_c_step_016260.pt` | `experiments/phase12b/runs/exp_c/step_016260.pt` | `cf0544099d73f85cdeb40bc1e0a93ce17bb790db4cb7136254be65d16c3b45b4` | 207,215,053 |
| `tokenizer/polycodegpt_tokenizer_v1_1.json` | `tokenizer/artifacts/polycodegpt_tokenizer_v1_1.json` | `0459f8556a6f79ca1ba1c3dc3f76c26ae2e55653ebbf97beaa9bd23467358a03` | 1,095,741 |
| `tokenizer/tokenizer_metadata_v1_1.json` | `tokenizer/artifacts/tokenizer_metadata_v1_1.json` | metadata artifact | 41,093 |
| `configs/model.yaml` | `configs/model.yaml` | `8b960967dea848ebb4cd428c48ad4155dc72c1c13e9f18aa23c54a22035ec8c2` | 686 |
| `provenance/phase12c_exp_c_audit.md` | `reports/phase12c_exp_c_audit.md` | `d589081edd250cbd17928db5c04399571c2be21f6070429280fdb4f6a305a80a` | 8,729 |
| `provenance/phase13_final_comparison.md` | `reports/phase13_final_comparison.md` | `1f3fd5d533f3cf11fed00952bef1d92cc6acffc6c68e8e90c9d491e595081ea0` | 13,427 |
| `provenance/phase14_candidate_integrity.md` | `reports/phase14_candidate_integrity.md` | `5c5903e8e0b83769d47a7ad836bcddb72e7fafff76be4739307ec8e2` | 6,477 |
| `provenance/phase14_final_audit.md` | `reports/phase14_final_audit.md` | `24ed2aa8564b4962569a12fb9be4ab9dfd1f78696a224dda5ec7d38e1d103f2c` | 10,585 |

Source-code file hashes for the minimal inference modules:

| Source module | SHA256 |
|---|---|
| `src/polycodegpt/model/config.py` | `2a5630150cf4545b5756820fd595001399f92e3b364ee4c1619f3ffb59d3809f` |
| `src/polycodegpt/model/gpt.py` | `fb1511a5de1721ead11401edd40e3f5a0bece0c20d8029b337e34076275aa9f7` |
| `src/polycodegpt/model/checkpoint.py` | `90ffcd66c25285c05316033c45195c3f12326269cf6172fff6edf6defd83e018` |
| `src/polycodegpt/data/tokenizer_io.py` | `bd17d5913e63dbc34c659d4efa2113129e39e4821b2338a4517a7963592420b5` |
| `src/polycodegpt/data/integrity.py` | `fc30a50c68f03f08c1a676c1e72c3d6630bb5649445dad7c3b1d5e4e4116b270` |
| `src/polycodegpt/tokenization/serializer.py` | `dd4d3712541fe9969142840ddb5ec1a0af16a1ff7199b3307cbeea217fe8a3fa` |

## 5. Packaging and validation commands

These commands are a plan for a future packaging step. They were not executed and do not create a package.

Dependency installation:

```bash
python -m pip install -r requirements.txt
python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cu132
```

Hash verification:

```bash
cd C:\Users\UltraPc\Desktop\LLM FROM SCRATCH\PolyCodeGPT\release\polycodegpt-exp-c-0.1.0
sha256sum -c hashes.sha256
```

Or in Python:

```python
import hashlib
from pathlib import Path

expected = {
    "models/exp_c_step_016260.pt": "cf0544099d73f85cdeb40bc1e0a93ce17bb790db4cb7136254be65d16c3b45b4",
    "tokenizer/polycodegpt_tokenizer_v1_1.json": "0459f8556a6f79ca1ba1c3dc3f76c26ae2e55653ebbf97beaa9bd23467358a03",
}
for rel, digest in expected.items():
    actual = hashlib.sha256(Path(rel).read_bytes()).hexdigest()
    assert actual == digest, (rel, actual, digest)
```

CPU load verification:

```python
import torch
from pathlib import Path
from polycodegpt.model.config import ModelConfig
from polycodegpt.model.gpt import GPT

payload = torch.load("models/exp_c_step_016260.pt", map_location="cpu", weights_only=True)
config = ModelConfig.from_dict(payload["model_config"])
model = GPT(config)
model.load_state_dict(payload["model_state"])
if config.tie_embeddings:
    model.lm_head.weight = model.token_embedding.weight
model = model.to("cpu").eval()
assert model.config.context_length == 512
assert sum(p.numel() for p in model.parameters()) == 16988160
```

Tokenizer verification:

```python
from polycodegpt.data.tokenizer_io import load_tokenizer, SPECIAL_TOKEN_IDS, EOS_ID

tokenizer = load_tokenizer(
    "tokenizer/polycodegpt_tokenizer_v1_1.json",
    expected_sha256="0459f8556a6f79ca1ba1c3dc3f76c26ae2e55653ebbf97beaa9bd23467358a03",
    vocab_size=16000,
)
assert tokenizer.get_vocab_size() == 16000
assert EOS_ID == 2
assert tokenizer.encode("<|lang:python|>", add_special_tokens=False).ids == [6]
assert tokenizer.encode("<|lang:javascript|>", add_special_tokens=False).ids == [7]
```

CUDA load and smoke test, only where CUDA is supported:

```python
import torch
model = model.to("cuda")
input_ids = torch.tensor([[6, 4]], dtype=torch.long, device="cuda")
with torch.no_grad():
    output = model.generate(input_ids, max_new_tokens=8, temperature=0, eos_token_id=2)
assert torch.isfinite(output).all()
```

Greedy reproducibility check:

```python
with torch.no_grad():
    first = model.generate(input_ids, max_new_tokens=16, temperature=0, eos_token_id=2)
    second = model.generate(input_ids, max_new_tokens=16, temperature=0, eos_token_id=2)
assert torch.equal(first, second)
```

Language-marker check:

```python
python_prompt = "<|lang:python|><|docstring|>Return x + 1.<|code|>def f(x):"
javascript_prompt = "<|lang:javascript|><|docstring|>Return x + 1.<|code|>function f(x) {"
assert tokenizer.encode(python_prompt, add_special_tokens=False).ids[0] == 6
assert tokenizer.encode(javascript_prompt, add_special_tokens=False).ids[0] == 7
```

Source-checkpoint non-modification check:

```python
import hashlib
source = hashlib.sha256(Path("experiments/phase12b/runs/exp_c/step_016260.pt").read_bytes()).hexdigest()
packaged = hashlib.sha256(Path("release/polycodegpt-exp-c-0.1.0/models/exp_c_step_016260.pt").read_bytes()).hexdigest()
assert source == packaged == "cf0544099d73f85cdeb40bc1e0a93ce17bb790db4cb7136254be65d16c3b45b4"
```

Package inventory check:

```bash
find release/polycodegpt-exp-c-0.1.0 -type f -print0 | sort -z | xargs -0 sha256sum
```

## 6. Performance and limitations to document in the README

The README must state the audited independent results honestly:

Exp C:

- Python complete tasks: 3/50.
- JavaScript complete tasks: 1/50.
- Total complete tasks: 4/100.
- Python cases passed: 27/164.
- JavaScript cases passed: 12/164.
- Total cases passed: 39/328.

Additional supporting results:

- Exp C generation syntax validity: 63/100 total, 40/50 Python, 23/50 JavaScript.
- Exp C EOS hits: 59/100 total, 37/50 Python, 22/50 JavaScript.
- Exp C mean unique-token ratio: 0.5121.
- Exp C mean repeated-line ratio: 0.0995.
- Exp C mean repeated-4-gram ratio: 0.1108.
- Exp C full validation loss: overall 2.3620, Python 2.3679, JavaScript 2.4161.
- Exp C training sampled best validation loss: 2.3198 at step 16260.
- Exp C training exposure: 266,385,408 cumulative tokens, 8,130 additional updates, zero skipped steps.

The README must clearly state:

- This is an experimental research model, not production-ready.
- The model was trained from random initialization with no pretrained weights.
- The contamination assessment was preliminary lexical evidence, not proof of complete independence.
- The benchmark contains 100 tasks and is not proof of general programming ability.
- No training or tuning used Phase 14 results.
- French and Arabic tokenizer markers exist, but French and Arabic capability is not claimed.
- Generated code must not be executed outside the existing sandbox.
- The model is strongly Python-biased; JavaScript performance is substantially weaker.

## 7. Risks and limitations

- The selected checkpoint is a full training checkpoint containing optimizer and scheduler state. This increases package size and includes training state that is not needed for inference. A future inference-only checkpoint could reduce size, but the source checkpoint must remain unchanged.
- `require_cuda` refuses CPU. The package's inference entry point must not reuse that helper for device resolution.
- CPU inference is not performance-tested, even though CPU loading was verified.
- The repository has no license file. Packaging must not invent a license; a license decision is required before public distribution.
- `pyproject.toml` currently declares only `requires-python = ">=3.11"` and does not declare runtime dependencies. The package must add an explicit dependency specification.
- The package's minimal module subset must preserve import paths for `ModelConfig`, `GPT`, tokenizer loading, and serializer constants.
- The model's context length is 512 tokens. Longer prompts are silently truncated during generation.
- Dropout is configured as 0.1 but is disabled in eval mode.
- The independent evaluation remains low: 4 complete tasks out of 100 and 39 passed cases out of 328. This is not release readiness.
- The contamination assessment is preliminary lexical evidence, not proof of full dataset independence.
- The benchmark is 100 tasks and is not proof of general programming ability.
- No training or tuning used Phase 14 results.

## 8. Confirmation

- No package was created.
- No source artifact was modified.
- No checkpoint, tokenizer, manifest, suite file, report, or configuration was modified.
- No candidate code was executed.
- No functional evaluation or generation evaluation was rerun.
- The old test partition was not accessed.
- Only `reports/phase15_packaging_preflight.md` was created.

## Final status

All required source artifacts, dependencies, entry points, checkpoint identity, tokenizer identity, model configuration, prompt format, and validation procedures were verified. The proposed package is feasible, but it remains an experimental research packaging plan and must not be described as production-ready.

PHASE 15 PREFLIGHT READY