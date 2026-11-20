# Phase 14 — Candidate integrity audit

Strict read-only audit of the two finalist generation reports:

- `reports/phase14_exp_b_generation.json`
- `reports/phase14_exp_c_generation.json`

Reference: `functional/independent_v3/manifest.json`

No candidate code was executed. No functional evaluation was run. The old test partition was not accessed. No existing file was modified; this report is the only new file.

## Audited report identities

| Report | SHA256 | Generation timestamp |
|---|---|---|
| `reports/phase14_exp_b_generation.json` | `216f99dcef549d81c921ecc61d75504c0db3394dacc1fbcda202b06782db8da4` | `20261118_150247` |
| `reports/phase14_exp_c_generation.json` | `842c33b75dd39af81ed29b426d52259864507646f47a0ce6ec3770637134bbac` | `20261118_150438` |

## 1. Exactly 100 candidate records

| Report | `per_task_results` records | Result |
|---|---:|---|
| Exp B | 100 | Pass |
| Exp C | 100 | Pass |

Each record is an object containing a candidate `task_id`, `language`, `docstring`, and `code_stub`.

## 2. Exactly 100 unique independent_v3 task IDs

| Report | Unique IDs in records | Top-level `task_ids` | Top-level order matches records | Result |
|---|---:|---:|---|---|
| Exp B | 100 | 100 | Yes | Pass |
| Exp C | 100 | 100 | Yes | Pass |

Both reports use `task_suite: independent_v3` and identify `functional\independent_v3\manifest.json` as the task manifest.

## 3. Exactly 50 Python and 50 JavaScript candidates

| Report | Python | JavaScript | Result |
|---|---:|---:|---|
| Exp B | 50 | 50 | Pass |
| Exp C | 50 | 50 | Pass |

The top-level `python_tasks` and `javascript_tasks` fields also report 50 and 50 in both files.

## 4. No missing, duplicate, unknown, or extra task IDs

Compared against the 100 task IDs in `functional/independent_v3/manifest.json`:

| Report | Missing | Duplicates | Unknown | Extra | Result |
|---|---:|---:|---:|---:|---|
| Exp B | 0 | 0 | 0 | 0 | Pass |
| Exp C | 0 | 0 | 0 | 0 | Pass |

All candidate IDs use the independent suite namespace (`py_v2_*`, `js_v2_*`) and exactly cover the manifest set.

## 5. Candidate identity fields match the manifest

For every candidate, the audit compared:

- `task_id`
- `language`
- `docstring`
- `code_stub` against the manifest `signature`

| Report | Field mismatches | Result |
|---|---:|---|
| Exp B | 0 / 100 | Pass |
| Exp C | 0 / 100 | Pass |

No language, task ID, docstring, or code-stub/signature mismatch was found.

## 6. Checkpoint SHA256

Artifact-level verification:

| Finalist | Checkpoint | Actual SHA256 | Expected SHA256 | Result |
|---|---|---|---|---|
| Exp B | `experiments/phase12b/runs/exp_b/step_016260.pt` | `fc70f5264a97d22c162986e172f8e70881713d5ae235c992e516a91ff67f5db6` | `fc70f5264a97d22c162986e172f8e70881713d5ae235c992e516a91ff67f5db6` | Match |
| Exp C | `experiments/phase12b/runs/exp_c/step_016260.pt` | `cf0544099d73f85cdeb40bc1e0a93ce17bb790db4cb7136254be65d16c3b45b4` | `cf0544099d73f85cdeb40bc1e0a93ce17bb790db4cb7136254be65d16c3b45b4` | Match |

Embedded report metadata:

| Report | `checkpoint_sha256` | `checkpoint_step` | Result |
|---|---|---|---|
| Exp B | `fc70f5264a97d22c162986e172f8e70881713d5ae235c992e516a91ff67f5db6` | 16260 | Pass |
| Exp C | `cf0544099d73f85cdeb40bc1e0a93ce17bb790db4cb7136254be65d16c3b45b4` | 16260 | Pass |

The checkpoint paths named in each report match the corresponding finalist.

## 7. Independent manifest SHA256

Actual file:

`2a7fbf3ddc43f6a2fee368d6faa711dbe8fa841343c7cf9507c919cd3baeb46c`

| Report | `task_manifest_sha256` | `task_manifest_hash` | Result |
|---|---|---|---|
| Exp B | `2a7fbf3ddc43f6a2fee368d6faa711dbe8fa841343c7cf9507c919cd3baeb46c` | same | Pass |
| Exp C | `2a7fbf3ddc43f6a2fee368d6faa711dbe8fa841343c7cf9507c919cd3baeb46c` | same | Pass |

Both reports also name `functional\independent_v3\manifest.json` as the manifest path.

## 8. Tokenizer SHA256

Actual tokenizer artifact:

`tokenizer/artifacts/polycodegpt_tokenizer_v1_1.json`

Actual SHA256:

`0459f8556a6f79ca1ba1c3dc3f76c26ae2e55653ebbf97beaa9bd23467358a03`

| Report | Embedded `tokenizer_sha256` | Tokenizer version | Result |
|---|---|---|---|
| Exp B | `0459f8556a6f79ca1ba1c3dc3f76c26ae2e55653ebbf97beaa9bd23467358a03` | `v1.1.0` | Pass |
| Exp C | `0459f8556a6f79ca1ba1c3dc3f76c26ae2e55653ebbf97beaa9bd23467358a03` | `v1.1.0` | Pass |

## 9. Generation settings

| Setting | Exp B | Exp C | Required | Result |
|---|---|---|---|---|
| `seed` | 42 | 42 | 42 | Pass |
| `temperature` | 0.0 | 0.0 | 0.0 | Pass |
| `max_new_tokens` | 64 | 64 | 64 | Pass |
| Device | `cuda` | `cuda` | Not specified | Recorded |

Greedy decoding was confirmed from the current generation implementation: `temperature == 0` selects `torch.argmax` and does not enter the `torch.multinomial` sampling branch.

Generated token counts stay within the 64-token ceiling:

| Report | Minimum | Maximum | All records <= 64 |
|---|---:|---:|---|
| Exp B | 16 | 64 | Yes |
| Exp C | 18 | 64 | Yes |

Both reports record `empty_generations: 0`.

## 10. No dependence on Phase 10 task records

| Check | Exp B | Exp C |
|---|---|---|
| `task_suite` is `independent_v3` | Yes | Yes |
| Manifest path is `functional\independent_v3\manifest.json` | Yes | Yes |
| All 100 candidate identities match independent_v3 | Yes | Yes |
| Candidate keys overlapping the 100 Phase 10 records | 0 | 0 |
| Phase 10 references in the JSON text | None | None |

Every candidate in both reports maps to an independent_v3 manifest task. No Phase 10 task identity or task record can supply any candidate in these reports.

## Limitations

- This is a static integrity audit. Candidate code was not executed, and no functional correctness was evaluated.
- Greedy decoding was verified from report metadata and the current generator code path; model generation was not replayed.
- Hash verification confirms that the audited bytes match the expected artifacts. It does not establish artifact provenance or signature.
- No existing checkpoint, manifest, tokenizer, report, or test-partition file was modified.

## Verdict

Both finalist generation reports pass all requested integrity checks: 100 records, 100 unique independent_v3 IDs, 50 Python and 50 JavaScript, no missing/duplicate/unknown/extra IDs, zero identity-field mismatches, correct checkpoint/manifest/tokenizer hashes, the required seed/temperature/token-limit settings, greedy decoding, and no Phase 10 task-record dependence.

CANDIDATES VERIFIED — FUNCTIONAL EVALUATION APPROVED