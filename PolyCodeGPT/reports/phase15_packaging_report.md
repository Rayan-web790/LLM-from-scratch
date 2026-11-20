# Phase 15 — PolyCodeGPT Exp C experimental release packaging report

Strictly limited to packaging and verification. No training, generation evaluation, functional evaluation, or source artifact modification was performed. The original checkpoint and tokenizer remain unchanged.

## 1. Package location and status

- Package directory: `PolyCodeGPT/release/polycodegpt-exp-c-0.1.0`
- ZIP archive: `PolyCodeGPT/release/polycodegpt-exp-c-0.1.0.zip`
- Status: Experimental research package, not production-ready.
- Package file count: 28
- Package file bytes: 208,448,119
- ZIP size bytes: 187,507,839
- ZIP SHA256: `3f5d1b9b5343fcb618ac2d7da7a043fbef97a5b83920ddfe57065f9d21aeb4f6`

## 2. Checkpoint and tokenizer hashes

| Artifact | SHA256 | Size |
|---|---|---:|
| `models/exp_c_step_016260.pt` | `cf0544099d73f85cdeb40bc1e0a93ce17bb790db4cb7136254be65d16c3b45b4` | 207,215,053 |
| `tokenizer/polycodegpt_tokenizer_v1_1.json` | `0459f8556a6f79ca1ba1c3dc3f76c26ae2e55653ebbf97beaa9bd23467358a03` | 1,095,741 |
| `configs/model.yaml` | `8b960967dea848ebb4cd428c48ad4155dc72c1c13e9f18aa23c54a22035ec8c2` | 686 |

Source artifact immutability was verified:

- Source checkpoint SHA256: `cf0544099d73f85cdeb40bc1e0a93ce17bb790db4cb7136254be65d16c3b45b4`
- Source tokenizer SHA256: `0459f8556a6f79ca1ba1c3dc3f76c26ae2e55653ebbf97beaa9bd23467358a03`
- Packaged hashes match the source hashes exactly.

## 3. Dependency versions

Verified environment:

- Python 3.11.9
- PyTorch 2.15.0+cu132
- CUDA 13.2
- tokenizers 0.22.2
- NumPy 1.26.4
- PyYAML 6.0.3

Package runtime dependencies:

- `numpy==1.26.4`
- `PyYAML>=6,<7`
- `tokenizers==0.22.2`
- `torch==2.15.0`

PyTorch is installed from the CUDA 13.2 wheel index for GPU support, or from the matching CPU wheel for CPU-only environments.

## 4. Inference entry point and usage

Entry point:

```bash
python -m polycodegpt --language python --task "..." --max-new-tokens 64
```

Supported options include `--language`, `--task`, `--code-stub`, `--max-new-tokens`,
`--temperature`, `--top-k`, `--device`, `--checkpoint`, and `--tokenizer`.

Default generation uses greedy decoding with `temperature=0.0` and 64 new tokens.
Generated code is returned as text only; the package never executes generated code.

Example Python invocation:

```bash
python -m polycodegpt \
  --language python \
  --task "Return the sum of squares of values in nums." \
  --max-new-tokens 64
```

Example JavaScript invocation:

```bash
python -m polycodegpt \
  --language javascript \
  --task "Return the sum of squares of values in nums." \
  --code-stub "function sumSquares(nums) {" \
  --max-new-tokens 64
```

## 5. Smoke-test results

All checks passed:

- Packaged checkpoint SHA256: OK
- Packaged tokenizer SHA256: OK
- Required modules and configuration files present: OK
- CPU model load: OK, checkpoint step 16260
- Parameter count: 16,988,160
- Python smoke generation: OK
- JavaScript smoke generation: OK
- Greedy reproducibility: OK, identical outputs on repeated runs
- Python language marker: ID 6
- JavaScript language marker: ID 7
- CUDA smoke test: OK
- Source checkpoint immutability: OK
- Source tokenizer immutability: OK
- CLI invocation (`python -m polycodegpt`): OK on CPU
- No `__pycache__` or `.pyc` files remain in the package

No generated code was executed during validation.

## 6. Reproducibility results

Repeated greedy generation with identical settings produced byte-identical token
sequences and decoded continuations. Greedy decoding uses `torch.argmax`
(`temperature=0.0`).

## 7. Package size

- Uncompressed package: 208,448,119 bytes (28 files)
- ZIP archive: 187,507,839 bytes
- Largest file: `models/exp_c_step_016260.pt` (207,215,053 bytes)

The checkpoint is a full training checkpoint and includes optimizer and scheduler
state in addition to model weights. The source checkpoint was not modified.

## 8. Known limitations

- Experimental research model; not production-ready.
- Independent evaluation: 3/50 Python complete tasks, 1/50 JavaScript complete tasks,
  4/100 total complete tasks.
- Independent evaluation cases: 27/164 Python, 12/164 JavaScript, 39/328 total.
- The 100-task benchmark is not proof of general programming ability.
- The contamination assessment was preliminary lexical evidence, not proof of
  complete independence.
- The model is strongly Python-biased; JavaScript performance is substantially weaker.
- French and Arabic tokenizer markers exist, but French and Arabic competence is
  not established.
- Context length is 512 tokens.
- Generated code must be reviewed before use and must never be executed automatically.

## 9. Licensing status

No software license file is included. The source repository did not contain a
project license, and this package does not grant additional rights or alter the
licensing status of the original artifacts. Obtain explicit authorization before
redistributing this package or its model weights.

## 10. Exact package inventory

| File | Size | SHA256 |
|---|---:|---|
| `configs/model.yaml` | 686 | `8b960967dea848ebb4cd428c48ad4155dc72c1c13e9f18aa23c54a22035ec8c2` |
| `docs/limitations.md` | 489 | `18772cacfc2b6bf2266992601b6af632d8c475699eedfea6c33212a06d14771d` |
| `examples/generate_python.py` | 676 | `80fa73ddf524ff89e68b13b6dc742df069a232f47fe7582f2b5852ceb289908a` |
| `hashes.sha256` | 2592 | `9f5a6e...` |
| `models/exp_c_step_016260.pt` | 207215053 | `cf0544099d73f85cdeb40bc1e0a93ce17bb790db4cb7136254be65d16c3b45b4` |
| `polycodegpt/__init__.py` | 51 | `a4c2628389b10ac4e495de4d44965cc67d433ff7cb1382487b8606bb3a63d261` |
| `polycodegpt/__main__.py` | 68 | `86665508f446dc0a8609179783950716cca6fe8b48cfd6ff5aa3494cd655dbc6` |
| `polycodegpt/data/__init__.py` | 206 | `0facbe7e084810edf1a3fd7c105046e2af17906cd9cb76d0653892fb5ab29cdc` |
| `polycodegpt/data/integrity.py` | 3097 | `e55d9920030ded7a4261710ff49b802bcd7ae6eaf29492832c01945dce967cc5` |
| `polycodegpt/data/tokenizer_io.py` | 4871 | `4ae57a17eb50bacafa38dae130285d623ce38920fc86acd582cd2dfdbd6acdb7` |
| `polycodegpt/inference.py` | 8141 | `3dfe12d9c1997c5d85ad4f79756e4bedfca8bb87a33cf40547b9d1eb21a0e483` |
| `polycodegpt/model/__init__.py` | 248 | `2c23de95fe1b40ed4c7ea3d8b75245345d2bf39012aad0af8c133c1c7403d6ad` |
| `polycodegpt/model/attention.py` | 2552 | `b1b32b88c96cba41c58edb6562f6110e2b78d242ffeec8025da613e0236de2fd` |
| `polycodegpt/model/checkpoint.py` | 1877 | `90ffcd66c25285c05316033c45195c3f12326269cf6172fff6edf6defd83e018` |
| `polycodegpt/model/config.py` | 5367 | `f48ba15bfee3c6741d24d242c96f0055e2536de5fb1cd410ca78ecd3afab44ce` |
| `polycodegpt/model/gpt.py` | 7427 | `fb1511a5de1721ead11401edd40e3f5a0bece0c20d8029b337e34076275aa9f7` |
| `polycodegpt/model/layers.py` | 1572 | `975d7289e162f573ba2e2f266ebbdbf2c7a3ef8b6911aee00b4e1f64c78b555b` |
| `polycodegpt/tokenization/__init__.py` | 719 | `5757f627afe980efe6d4e3428541911112e4818e3896e4f5108d9bb5c394dc51` |
| `polycodegpt/tokenization/serializer.py` | 11530 | `b2b6211286fe104c0e186eb38240fe562650887a841a75894d46c127e57fddf0` |
| `provenance/phase12c_exp_c_audit.md` | 8729 | `d589081edd250cbd17928db5c04399571c2be21f6070429280fdb4f6a305a80a` |
| `provenance/phase13_final_comparison.md` | 13427 | `1f3fd5d533f3cf11fed00952bef1d92cc6acffc6c68e8e90c9d491e595081ea0` |
| `provenance/phase14_candidate_integrity.md` | 6477 | `5c5903e8e0b824539e338769d47a7ad836bcddb72e7fafff76be4739307ec8e2` |
| `provenance/phase14_final_audit.md` | 10585 | `24ed2aa8564b4962569a12fb9be4ab9dfd1f78696a224dda5ec7d38e1d103f2c` |
| `pyproject.toml` | 432 | `efc6d265eb6505d8c39d120ac00857a88d739dca8beb7aa0072766c2cb458d30` |
| `README.md` | 4009 | `a5a01fe2cd1a9bb4d427936df46f0eefa5013aec84897428d5baba21b4a92ddb` |
| `requirements.txt` | 404 | `0a6274cca00843b1e40515be86875f93e70ff31ec20bc79cf98a547d5200ddda` |
| `tokenizer/polycodegpt_tokenizer_v1_1.json` | 1095741 | `0459f8556a6f79ca1ba1c3dc3f76c26ae2e55653ebbf97beaa9bd23467358a03` |
| `tokenizer/tokenizer_metadata_v1_1.json` | 41093 | `71dd044cb775accf3863890307a4e4700047f137c1534e7fb81751861cabef99` |

The `hashes.sha256` file itself is not listed in its own manifest; its SHA256 is not included above.

## Final status

The experimental release package was created and validated successfully. The checkpoint and tokenizer remain unchanged. The package includes the minimal inference interface, exact dependency specification, documentation, provenance, checksum manifest, and ZIP archive.

PHASE 15 COMPLETE — EXP_C EXPERIMENTAL PACKAGE VERIFIED