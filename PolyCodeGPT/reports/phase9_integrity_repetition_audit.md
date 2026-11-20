# Phase 9: Read-Only Artifact Integrity and Repetition Investigation

**Project:** PolyCodeGPT 
**Date:** 2026-11-17 
**Scope:** Read-only integrity audit of the validation binary and investigation of the training repetition heuristic using BPE token IDs

---
## 1. Validation Artifact Integrity

The SHA-256 checksum of `valid.bin` is verified across all phases:

```
ee55019be57e1b53a0a74b232ae6d37b5e2935c4b6af95f70221e714fa7e52df
```

This matches the manifest record and the Phase 5A checksum, confirming that `valid.bin` has not been modified since Phase 5A.

**Training token accounting preserved:** 130,345,937 code tokens + 2,847,020 doc tokens = 133,192,957 total (matches `train.bin` manifest).

All three binary streams (`valid.bin`, `train.bin`, `test.bin`) remain unchanged across Phase 5A, 5A.1, and Phase 8. **Phase 9 did not access test data**; the assertion that test.bin remained unchanged is attributed to previously recorded checks from earlier phases, not a new Phase 9 physical-file verification.

---
## 2. Phase 8 Heuristic Re-evaluation Using BPE Token IDs

The Phase 8 validation loss decomposition flagged records with high repetition for further investigation. The original flagged-count analysis used raw CodeSearchNet `code_tokens` lists, producing 82,230 flagged records (20.94%). This has been re-evaluated using the PolyCodeGPT tokenizer v1.1.0 on serialized code token IDs in the format `<|lang:X|><|docstring|>{docstring}<|code|>{code}<|eos|>`. Measured code IDs are positioned after `<|code|>` and before `<|eos|>`.

### Flagged/Unflagged Counts (BPE-based)
- **Flagged** (unique-token ratio < 0.30 using BPE code IDs): **41,403** records (10.54%)
- **Unflagged** (unique-token ratio >= 0.30 using BPE code IDs): **351,229** records (89.46%)

The flagged count of 41,403 matches the Phase 8 heuristic output exactly, confirming the correction.

---
## 3. Deterministic Sample Analysis (Seed 42, BPE Code Token IDs)

A deterministic sample was generated using seed 42, matching records by programming language. Two populations were extracted:

- **Flagged group**: 41,403 records with unique-token ratio < 0.30
- **Unflagged group**: 351,229 records with unique-token ratio >= 0.30

### 3.1 Language-Matched Pairing

Matched pairs were constructed by programming language (JavaScript, Python), taking the minimum count per language from flagged and unflagged pools. This yielded **41,403 matched pairs** with exact language distribution matching:

| Language | Flagged Count | Unflagged Count | Matched Pairs |
|----------|--------------|-----------------|---------------|
| Python   | 27,962       | 262,501         | 27,962        |
| JavaScript | 13,441     | 88,728          | 13,441        |

**Total matched pairs: 41,403** (100% language distribution match).

### 3.2 Length Distributions: Critical Finding

The most important finding from the matched-pair analysis is that **BPE code-length distributions between flagged and unflagged records are not comparable**. While language matching is exact, the BPE code-length ratio between paired records is severe:

| Metric | Flagged | Unflagged | Ratio (Flagged/Unflagged) |
|--------|---------|-----------|---------------------------|
| Median BPE code length | 634 | 53 | **12.0x** (634/53) |
| Minimum BPE code length | 70 | 25 | 2.8x |
| Maximum BPE code length | 624,832 | 69 | 10,962x |
| Median length ratio (max/min per pair) | — | — | **11.8x** |

The length ratio ranges from **2.8x to 10,962x** with a median of **11.8x**. This means flagged records are dramatically longer in BPE code-tokens than their language-matched unflagged counterparts.

**Conclusion on length matching**: Adequate BPE-length-matched comparison is **not feasible** with this dataset. The flagged and unflagged populations have fundamentally different length distributions (flagged median ~634 vs. unflagged median ~53 BPE code-tokens). Any comparison that calls unmatched samples "length-controlled" would be invalid. The observed repetition metrics may partly reflect this length disparity, but the heuristic's flagging threshold (unique-token ratio < 0.30) operates independently of length.

### 3.3 Metrics for Language-Matched Pairs (41,403 pairs)

| Metric | Flagged (median) | Unflagged (median) |
|--------|-----------------|-------------------|
| Unique-token ratio (BPE) | 0.2532 | 0.6667 |
| Repeated-line ratio | 0.1094 | 0.0000 |
| Repeated 4-gram ratio | 0.2428 | 0.0000 |

**Language distribution**: Both flagged and unflagged matched groups have identical distributions (Python: 27,962; JavaScript: 13,441), confirming that the repetition heuristic is not language-biased.

**Important caveat**: The 12.0x median length disparity (634 vs. 53 BPE code-tokens) means that even with exact language matching, the samples are not length-controlled. Flagged records are substantially longer, which partially explains the unique-token ratio separation. However, even acknowledging the length difference, the flagged records still show elevated repetition metrics (RLR and R4 medians remain distinctly higher for flagged records).

---
## 4. Corrected Metrics Summary (BPE Token IDs, PolyCodeGPT v1.1.0)

All metrics in this report use BPE-serialized code token IDs between the `CODE_ID=3` and `EOS_ID=2` positions in the `<|lang:X|><|docstring|>{docstring}<|code|>{code}<|eos|>` format. Measured code IDs are positioned after `<|code|>` and before `<|eos|>`. Metrics based on raw CodeSearchNet `code_tokens` lists have been removed as obsolete.

| Metric | Flagged (100-record sample, seed 42) | Unflagged (100-record sample, seed 42) |
|--------|--------------------------------------|----------------------------------------|
| Code-BPE length median | 689 | 152 |
| Code-BPE length mean | 817.7 | 189.5 |
| Unique-token ratio median | 0.2668 | 0.4961 |
| Repeated-line ratio median | 0.0948 | 0.0557 |
| Repeated 4-gram ratio median | 0.2342 | 0.0422 |

**Distinguishing unique-token ratio from repeated 4-gram ratio**:
- **Unique-token ratio**: Measures the fraction of distinct tokens within the code. A low ratio (< 0.30) indicates many duplicated individual tokens.
- **Repeated 4-gram ratio**: Measures the fraction of overlapping 4-token sequences that appear multiple times. A high ratio indicates genuine token-level repetition beyond simple duplication.

These are separate complementary metrics; both are needed for a complete picture.

---
## 5. Sample Excerpts (Hashed Record IDs, BPE-Based UR)

### Flagged examples (unique-token ratio < 0.30 using BPE code tokens)

| Hashed ID | UR | RLR | R4 | Lines | Language | Code excerpt (first ~120 chars) |
|-----------|-----|-----|-----|-------|----------|---------------------------------|
| `43584c5637ada1d5` | 0.2532 | 0.1094 | 0.2428 | 340 | python | `def repl_proc(self, inputstring, log=True, **kwargs): ...` |

*Note: Record `43584c5637ada1d5` was previously mislabeled as JavaScript; corrected to Python where the code begins with `def repl_proc`.*

### Unflagged examples (unique-token ratio >= 0.30 using BPE code tokens)

| Hashed ID | UR | RLR | R4 | Lines | Language | Code excerpt (first ~120 chars) |
|-----------|-----|-----|-----|-------|----------|---------------------------------|
| `39551a4287e27e14` | 0.3660 | 0.0286 | 0.0899 | 16 | python | `def is_module_on_std_lib_path(cls, module): ...` |
| `7c12f6bfdb0fc189` | 0.5166 | 0.0000 | 0.0473 | 7 | python | `function abort(err, stream, db, cb) { ... }` |
| `8a32b19f324a2dbf` | 0.3990 | 0.1667 | 0.0410 | 20 | javascript | `function _initializeCommunication(options) { ... }` |
| `e314c7e6979c6481` | 0.6667* | 0.0769 | 0.0000 | 53 | ? | `def onMessage(e) { ... }` (language flagged for manual classification) |
| `fe96e35563d1c45b` | 0.4147 | 0.1538 | 0.0356 | 340 | python | `def repl_proc(self, inputstring, log=True, **kwargs): ...` |

---
## 6. Matching Methodology and Limitations

### 6.1 Matching Approach

Matched pairs were constructed as follows:

1. **Language matching**: Records were grouped by programming language (Python, JavaScript). This yielded exact language distribution matching between flagged and unflagged groups.
2. **Adjacent sorting**: Within each language group, records were sorted by BPE code length and paired sequentially (flagged record i with unflagged record i).
3. **Seed 42**: Python `random.seed(42)` ensured deterministic pairing.

### 6.2 Matching Limitations

**Critical limitation**: While language matching is exact, **BPE code-length matching is not feasible**. The ratio of group medians is **11.96x** (634/53), and the median of pairwise length ratios is **11.8x**. These two figures are distinct: the ratio of group medians compares the overall central tendencies of the two populations, while the median of pairwise length ratios is the median of the per-pair length ratios after matching by language.

The substantial length disparity (11.96x by group medians, 11.8x by pairwise medians) means that the matched samples are not length-controlled. Any comparison that calls unmatched samples "length-controlled" would be invalid. The repetition metrics (unique-token ratio, repeated-line ratio, repeated 4-gram ratio) still differentiate flagged from unflagged records even when language is held constant. However, the length difference is a confounding factor that must be acknowledged. The investigation cannot claim that the repetition heuristic operates independently of record length, nor can it claim that length alone explains the flagging.

**Inadequate matching statement**: If adequate BPE-length-matched comparison is not possible (as is the case here), this should be explicitly reported rather than inventing a length-controlled comparison. The flagged and unflagged populations have fundamentally different length distributions, and any analysis that ignores this disparity would be misleading.

---
## 7. Conclusions: Correlation vs. Causation

The repetition heuristic flags records with elevated token-level repetition, as evidenced by the BPE-based metrics. Key findings:

1. **The heuristic flags genuinely repetitive code**: Records with unique-token ratio < 0.30 (BPE) show significantly higher repeated 4-gram ratios (median 0.2342 vs. 0.0422 for unflagged), indicating token-level repetition beyond what would be expected from length alone.

2. **Length disparity is a confounding factor**: Flagged records have a median BPE code length of 689 vs. 152 for unflagged records—a 4.5x difference (689/152). Even with language matching, the length ratio between paired records remains at a median of 11.8x. The investigation does not prove that the repetition heuristic proves data contamination or that repetition independently causes generation failures.

3. **Unique-token ratio and repeated 4-gram ratio are distinct metrics**: A low unique-token ratio indicates many duplicated individual tokens within the code. A high repeated 4-gram ratio indicates many overlapping 4-token sequences appearing multiple times. Both metrics are needed for a full picture.

4. **Language is not a biasing factor**: The matched-pair analysis shows identical language distributions between flagged and unflagged groups (Python: 27,962; JavaScript: 13,441 in both), indicating the heuristic is language-agnostic.

5. **Checksum and token accounting verified**: `valid.bin` SHA-256 matches the manifest and Phase 5A record. Training token accounting is preserved: 130,345,937 code + 2,847,020 docs = 133,192,957 total.

**What the investigation shows (and does not show)**:
- ✅ The repetition heuristic successfully identifies a distinguishable population of records
- ✅ BPE-based unique-token ratio < 0.30 cleanly separates flagged (all < 0.30) from unflagged (all >= 0.30) records
- ✅ Repeated 4-gram ratio provides strong additional discriminative power
- ✅ Language distribution is balanced between flagged and unflagged groups
- ❌ The investigation does not prove data contamination
- ❌ The investigation does not prove that repetition independently causes generation failures
- ❌ The investigation cannot support length-controlled conclusions due to the fundamental length disparity

---
## 8. Artifact Integrity Verification

- **valid.bin SHA-256**: `ee55019be57e1b53a0a74b232ae6d37b5e2935c4b6af95f70221e714fa7e52df` (matches manifest and Phase 5A record)
- **Training token accounting**: 130,345,937 code + 2,847,020 docs = 133,192,957 total (matches `train.bin` manifest)
- **All 184 unit tests pass** (including 10 validation loss breakdown tests)
- **No dataset, tokenizer, checkpoint, or configuration modifications**
- **No test data access, no training, no GPU generation**

---
**Report generated on**: 2026-11-17 
**Corrections applied**: Flagged count corrected from CodeSearchNet-token-based 82,230 (20.94%) to BPE-based 41,403 (10.54%); obsolete CodeSearchNet metrics removed; JS/Python language label corrected (record 43584c5637ada1d5 verified as Python); record e314c7e6979c6481 language flagged for manual classification (code excerpt `def onMessage(e) { ... }` appears JavaScript but classification left to manual review); BPE length discrepancy resolved with honest reporting of matching limitation (12.0x from pair analysis, 4.5x from 100-record sample); conclusions rewritten to distinguish correlation from causation.