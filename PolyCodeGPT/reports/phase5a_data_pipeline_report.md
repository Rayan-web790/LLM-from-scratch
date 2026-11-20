# Phase 5A / 5A.1 — Tokenized Data Pipeline Report

**Project:** PolyCodeGPT
**Date:** 2026-11-15
**Scope:** Serialize and tokenize the approved corpus into memory-mappable token streams with a Dataset/DataLoader. **No model training.**
**Revision:** 5A.1 — window stride changed from `context_length + 1` (non-overlapping) to `context_length` (one-token overlap). Token streams were **not** regenerated.
**Verdict:** **DATA PIPELINE FINALIZED**

---

## 0. Revision 5A.1 — overlapping windows

### The defect

Phase 5A used `block_stride = context_length + 1`, so windows did not overlap. In a window of
`context_length + 1` tokens, the **last** token can never be a target, because no next token follows
it inside that window. With no overlap, that token also begins no other window, so exactly one
next-token prediction was lost at every block boundary.

### The fix

`block_stride = context_length`, so window `i` starts at `i * 512` and covers
`[i*512, i*512 + 513)`. Consecutive windows now share exactly one token — the boundary token — and
the target ranges tile the stream:

```
block i : tokens[i*512 : i*512 + 513]     input = [:-1], targets = [1:]
targets : [i*512 + 1 .. i*512 + 512]
```

Union over `i = 0 .. B-1` is exactly `[1 .. B*512]`: every next-token target is used **once**,
position 0 is context only, and only the incomplete tail is unused.

### Verification on the real streams

| Split | Tokens | Blocks (5A → 5A.1) | Tail (5A → 5A.1) | Next-token targets used |
|---|---:|---:|---:|---:|
| train | 133,192,957 | 259,635 → **260,142** | 202 → **252** | 133,192,704 of 133,192,956 (99.9998 %) |
| valid | 7,757,005 | 15,120 → **15,150** | 445 → **204** | 7,756,800 of 7,757,004 |
| test | 7,447,823 | 14,518 → **14,546** | 89 → **270** | 7,447,552 of 7,447,822 |
| **total** | 148,397,785 | 289,273 → **289,838** | 736 → **726** | 148,397,056 of 148,397,782 |

All three match the expected counts for this revision exactly. Identity checked per split:

```
tokens == blocks * window_stride + 1 + dropped_trailing_tokens
133,192,957 == 260,142 * 512 + 1 + 252   ✓
  7,757,005 ==  15,150 * 512 + 1 + 204   ✓
  7,447,823 ==  14,546 * 512 + 1 + 270   ✓
```

Measured on the real files:

| Split | `block(0) == tokens[0:513]` | `block(1) == tokens[512:1025]` | Shared boundary token | `targets[0][-1] == inputs[1][0]` |
|---|---|---|---|---|
| train | True | True | id 287 | True |
| valid | True | True | id 1921 | True |
| test | True | True | id 644 | True |

Block 0 is byte-identical to Phase 5A (`<|lang:python|><|docstring|>Trains a k-nearest neighbors
classifier…`), confirming the streams themselves did not move — only the offsets did.

### Code changes

| File | Change |
|---|---|
| `src/polycodegpt/data/stream.py` | `block_geometry(token_count, context_length, stride=None)` — stride defaults to `context_length`; returns `window_length`, `window_stride`, `window_overlap`, `block_span` (alias), `blocks`, `dropped_trailing_tokens`, `tokens_in_blocks`. Blocks = `1 + (tokens - window) // stride`. Rejects `stride > window` (would skip targets) and `stride <= 0`. |
| `src/polycodegpt/data/dataset.py` | `TokenBlockDataset(stride=None)` stores `window_stride`/`window_overlap`; `block(i)` reads `mm[i*stride : i*stride + window_length]`; `open_split` takes the stride from the manifest and raises if the dataset computes a different one. |
| `src/polycodegpt/data/prepare.py` | `MANIFEST_VERSION = "1.1.0"`; manifest now records `window_length`, `window_stride`, `window_overlap`, `block_scheme: "overlapping"` and `block_indexing`. |
| `scripts/update_manifest_geometry.py` | **New.** Recomputes block/tail counts in an existing `manifest.json` + `sources.csv` from the file sizes, after verifying every stream's size, token count and SHA-256. Never opens a stream for writing; records the previous geometry under `geometry_revision`. |
| `scripts/inspect_tokenized_data.py` | Prints window/stride/overlap; new "window geometry" section asserts `block(0)`/`block(1)` offsets, the one-token overlap and the coverage identity per split. |
| `scripts/prepare_data.py` | Summary prints window/stride/overlap/context. |
| `tests/unit/test_data_pipeline.py` | 11 new tests (below); existing geometry tests updated for the new stride. |

Command used to refresh the metadata (streams untouched):

```
"C:/Users/UltraPc/Desktop/LLM FROM SCRATCH/.venv/Scripts/python.exe" scripts/update_manifest_geometry.py
```

```
train  OK   133,192,957 tokens  6b8ffbf653e8a3b3dbe5c641...  (file untouched)
valid  OK     7,757,005 tokens  ee55019be57e1b53a0a74b23...  (file untouched)
test   OK     7,447,823 tokens  4ce477e357c2b3ab8bd7e5b7...  (file untouched)
blocks 259,635 -> 260,142 / 15,120 -> 15,150 / 14,518 -> 14,546
stream hashes unchanged after the metadata update
```

`manifest.json` is now version 1.1.0 and carries a `geometry_revision` block recording the previous
stride, the previous per-split counts, and `streams_regenerated: false`, `streams_modified: false`.

---

## 1. Files created or modified

| Path | Status | Purpose |
|---|---|---|
| `src/polycodegpt/data/__init__.py` | modified (was empty) | Public exports |
| `src/polycodegpt/data/tokenizer_io.py` | created (5A) | Verified tokenizer loading, special-token contract, `encode()` |
| `src/polycodegpt/data/integrity.py` | created (5A) | SHA-256, disk-space checks, `verify_outputs()` |
| `src/polycodegpt/data/stream.py` | created (5A), stride (5A.1) | Buffered uint16 `TokenStreamWriter`, window geometry |
| `src/polycodegpt/data/prepare.py` | created (5A), metadata (5A.1) | Batched serialize-to-tokenize preparation, manifest, provenance |
| `src/polycodegpt/data/dataset.py` | created (5A), stride (5A.1) | `TokenBlockDataset` (lazy memmap), `build_dataloader`, `open_split` |
| `scripts/prepare_data.py` | created (5A) | Preparation entry point |
| `scripts/inspect_tokenized_data.py` | created (5A), geometry checks (5A.1) | Hashes, marker provenance, samples, forward-only GPU check |
| `scripts/update_manifest_geometry.py` | created (5A.1) | Geometry-only manifest refresh, streams verified read-only |
| `tests/unit/test_data_pipeline.py` | created (5A), +11 (5A.1) | 42 tests |
| `.gitignore` | modified (5A) | `data/processed/tokenized_v*/`, `*.bin` |
| `reports/phase5a_data_pipeline_report.md` | created (5A), revised (5A.1) | This report |

Dataset outputs in `data/processed/tokenized_v1_1/`: `train.bin` (266,385,914 B), `valid.bin`
(15,514,010 B), `test.bin` (14,895,646 B), `manifest.json`, `sources.csv`. No `.bin` file was
regenerated or modified in this revision, and no source corpus, tokenizer artifact, model module or
checkpoint was touched.

---

## 2. Exact preparation command

Phase 5A (originally generated the streams, 117.74 s):

```
"C:/Users/UltraPc/Desktop/LLM FROM SCRATCH/.venv/Scripts/python.exe" scripts/prepare_data.py
```

Phase 5A.1 (metadata only — streams reused):

```
"C:/Users/UltraPc/Desktop/LLM FROM SCRATCH/.venv/Scripts/python.exe" scripts/update_manifest_geometry.py
```

Verification:

```
"C:/Users/UltraPc/Desktop/LLM FROM SCRATCH/.venv/Scripts/python.exe" scripts/inspect_tokenized_data.py --forward-check
```

---

## 3. Test results

```
python -m pytest -q
124 passed in 7.79s
```

113 existing (82 pre-5A + 31 pipeline) + **11 new** stride tests, **0 skipped, 0 xfail**.
`tests/unit/test_data_pipeline.py` alone: **42 passed in 6.85 s**.

The eleven new tests map directly to the 5A.1 requirements:

| # | Requirement | Test |
|---|---|---|
| 1 | First window uses `tokens[0:513]` | `test_first_window_uses_tokens_zero_to_context_length` |
| 2 | Second window uses `tokens[512:1025]` | `test_second_window_starts_at_context_length` |
| 3 | Consecutive windows overlap by exactly one token | `test_consecutive_windows_overlap_by_exactly_one_token` |
| 4 | Targets shifted by one position | `test_next_token_shift_is_exact` (asserts `target[t] == tokens[start+t+1]` for every position) |
| 5 | Every target used exactly once | `test_every_target_is_used_exactly_once` (asserts `sorted(targets) == range(1, used)`) |
| 6 | No target skipped at boundaries | `test_no_target_is_skipped_at_block_boundaries` |
| 7 | Final incomplete tail counted | `test_final_tail_is_counted_and_never_read`, `test_block_counting_and_remainder`, `test_short_stream_yields_no_blocks` |
| 8 | Splits stay independent | `test_splits_stay_independent_under_overlap`, `test_no_split_mixing` |
| 9 | Memory-mapped access still works | `test_memmap_offsets_follow_the_stride` (`mm[8:17]`, not `mm[9:18]`), `test_memmap_indexing_is_lazy_and_bounds_checked` |
| 10 | Deterministic indexing | `test_indexing_is_deterministic` (same index twice, same seed → same epoch order) |

Plus `test_open_split_uses_manifest_stride` (manifest stride is honoured and a wrong stride is
rejected rather than silently trusted) and `test_geometry_rejects_a_stride_that_would_skip_targets`
(`stride > window` refused).

Existing contract tests still pass unchanged: EOS handling, no double-EOS, no out-of-vocab IDs,
source-hash abort, corruption detection, manifest fields, DataLoader shuffle settings, special-token
IDs, synthetic end-to-end preparation.

---

## 4. Token counts, block counts and output sizes

| Split | Tokens | Blocks | Tail | Bytes | MiB |
|---|---:|---:|---:|---:|---:|
| train | 133,192,957 | 260,142 | 252 | 266,385,914 | 254.05 |
| valid | 7,757,005 | 15,150 | 204 | 15,514,010 | 14.80 |
| test | 7,447,823 | 14,546 | 270 | 14,895,646 | 14.21 |
| **total** | **148,397,785** | **289,838** | **726** | **296,795,570** | **283.05** |

Train = 130,345,937 code tokens (392,632 records) + 2,847,020 documentation tokens (500 files).
Every file size equals `tokens × 2 bytes`.

**Design decisions (unchanged except the stride):**

* Windows are `context_length + 1 = 513` tokens, taken from `configs/model.yaml`
  (`context_length: 512`); no other context length is hardcoded.
* Stride is `context_length = 512`, so consecutive windows overlap by exactly one token.
* `input_ids = window[:-1]`, `targets = window[1:]` — target *t* is exactly the token after input *t*.
* Splits are independent files; no window can cross a split boundary.
* A window may cross an example boundary inside its own split (EOS separates examples). Near-duplicate
  context between adjacent blocks is limited to that one token (0.2 % of a window).
* No padding. The trailing 726 tokens complete no window and are reported, never silently dropped.
* No example is truncated or dropped. Train order: code records first, then documentation.

---

## 5. Documentation inclusion verification

* `lang:en` in `train.bin` = **500**, matching the inventory (all `train_only`).
* All 500 files SHA-256-verified against the inventory before tokenization.
* `train.bin` tail decodes to CPython documentation and ends with `<|eos|>`.
* `lang:en` in `valid.bin` / `test.bin` = **0**; no validation or test documentation was created.

Marker provenance (whole streams, unchanged by the stride revision):

| Split | `lang:python` | `lang:javascript` | `lang:en` | `eos` | Examples | Manifest |
|---|---:|---:|---:|---:|---:|---:|
| train | 290,463 | 102,169 | 500 | 393,132 | 393,132 | 393,132 |
| valid | 14,671 | 6,294 | 0 | 20,965 | 20,965 | 20,965 |
| test | 15,381 | 5,699 | 0 | 21,080 | 21,080 | 21,080 |

These equal `_stats.json` exactly and `EOS == examples` in every split (exactly one EOS per example,
no double-EOS).

---

## 6. SHA-256 verification

`sha256sum` on the streams after the 5A.1 metadata update — **identical to the Phase 5A manifest**:

| File | SHA-256 | vs Phase 5A |
|---|---|---|
| `train.bin` | `6b8ffbf653e8a3b3dbe5c641463dd78366bdd3e4387758fbc98866a6aba6e021` | identical |
| `valid.bin` | `ee55019be57e1b53a0a74b232ae6d37b5e2935c4b6af95f70221e714fa7e52df` | identical |
| `test.bin` | `4ce477e357c2b3ab8bd7e5b751e2e3d23d75ecc2dcefd01ea63b400c0db24104` | identical |

`verify_outputs()` → `ok: True`. Inputs re-checked after all work:

| Input | Status |
|---|---|
| `tokenizer/artifacts/polycodegpt_tokenizer_v1_1.json` (`0459f855…`) | UNCHANGED |
| `…/polycodegpt_corpus/train.jsonl` (`2a9c7160…`) | UNCHANGED |
| `…/polycodegpt_corpus/valid.jsonl` (`2eb02406…`) | UNCHANGED |
| `…/polycodegpt_corpus/test.jsonl` (`93f624f6…`) | UNCHANGED |

---

## 7. DataLoader samples and decoded examples

```
[train] 260,142 blocks of 513 tokens (stride 512, overlap 1, tail 252)
  block 0: input (512,) target (512,) dtype torch.int64
  block 1: input (512,) target (512,) dtype torch.int64
[valid] 15,150 blocks of 513 tokens (stride 512, overlap 1, tail 204)
[test]  14,546 blocks of 513 tokens (stride 512, overlap 1, tail 270)
```

Batch: `input_ids (2, 512) torch.int64 on cuda:0`, `targets (2, 512)`.

```
[train] <|lang:python|><|docstring|>Trains a k-nearest neighbors classifier for face recognition.
        :param train_dir: directory that contains a sub-directory for each known person ... 
[valid] <|lang:python|><|docstring|>Train a deepq model.  Parameters ------- env: gym.Env ...
[test]  <|lang:python|><|docstring|>Deprecated: please use ipyvolume.quickvolshow ...
        <|code|>def volshow(*args, **kwargs): ...
```

Language and role markers appear where expected. Block 1 of each split starts at token 512, i.e.
mid-example — expected, since windows step over a flat stream.

---

## 8. CUDA forward-only check

```
device:          NVIDIA GeForce RTX 5060 Laptop GPU
input_ids:       (2, 512) torch.int64 on cuda:0
targets:         (2, 512) torch.int64
logits:          (2, 512, 16000) torch.float32
forward loss:    9.750282  (random init — not a training result)
memory allocated: 127.88 MiB
memory peak:      194.64 MiB
```

`torch.no_grad()` only — no `backward()`, no `optimizer.step()`, no checkpoint written.

---

## 9. Disk space

526.72 GB free before the Phase 5A preparation; outputs consume 283.05 MiB. The 5A.1 revision wrote
only `manifest.json` (≈6 KB) and `sources.csv` (≈1 KB).

---

## 10. Errors, exclusions, warnings, limitations

**Errors / exclusions:** none. `failures = []`, `completed_successfully = true`, no
`failures.jsonl`. All 434,677 records and all 500 documents are present.

**Warnings / limitations:**

1. **uint16 is safe only for a vocabulary ≤ 65,536** (`dtype_is_safe` enforces it). The dtype and
   byte order are recorded in the manifest.
2. **Windows cross example boundaries** inside a split; `<|eos|>` therefore appears mid-sequence and
   the loss is not masked there. Documented in the manifest
   (`blocks_may_cross_example_boundaries: true`).
3. **726 trailing tokens are unused** (252 / 204 / 270 per split); they complete no 513-token window.
   All other next-token positions are now used exactly once.
4. **`block_span` is kept as an alias of `window_length`** for readers written against manifest 1.0.0;
   `window_length` is the authoritative field.
5. **The `.bin` files are unchanged, so they carry no record of the stride.** The stride lives only in
   the manifest, which `open_split` enforces (a mismatched `window_stride` raises). Reverting the
   stride means re-running `scripts/update_manifest_geometry.py`.
6. **`num_workers = 0`** default (Windows spawn cost); multi-worker throughput was not benchmarked.
   `pin_memory` defaults to "CUDA available".
7. **`encode_batch` uses the tokenizers rayon pool**; tokenization is deterministic but wall-clock
   timings are machine-dependent and excluded from the deterministic manifest view.
8. **No validation/test documentation exists**, as required; evaluation stays code-only.

---

## Verdict

**DATA PIPELINE FINALIZED**

`data/processed/tokenized_v1_1/` — 148,397,785 tokens, 289,838 overlapping windows (stride 512,
context 512, overlap 1), 0 failures, stream hashes unchanged, 124 tests passing. Training has not
begun and no training code was written.
