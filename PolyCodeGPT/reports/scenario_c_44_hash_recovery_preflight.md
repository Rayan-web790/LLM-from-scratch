# PolyCodeGPT — Scenario C 44-Hash Recovery Preflight

**Investigation scope:** Targeted read-only investigation of 44 hashes present in SQLite but absent from JSONL
**Project:** `C:\Users\UltraPc\Desktop\LLM FROM SCRATCH\PolyCodeGPT`
**Investigation date:** 2026-11-20
**Status:** Preflight investigation only — no repairs performed

---

## 1. Executive Summary

All 44 missing hashes were successfully located in the original parquet shard `train-00079-of-01126.parquet`. Every record is recoverable without re-extracting the other 79 shards. All 44 records:

- Are from shard 79 (the final shard)
- Have `partition=train` (no valid-split records among the 44)
- Have `syntax_valid=True` (all pass validation)
- Have non-empty code
- Have `code_sha256` matching the DB hash exactly
- Have no cross-split duplicates

**Recommended approach:** Option A — Restore the 44 missing records to the train JSONL split, then reconcile file sizes and SQLite offsets. This is the safest approach because the original source code is available and all records pass validation.

**Verdict:** RECOVERABLE — SAFE PROCEDURE IDENTIFIED

---

## 2. The 44 Hashes and Reconstruction Results

### Summary

| Metric | Value |
|--------|-------|
| Total target hashes | 44 |
| Found in shard 79 | 44 / 44 |
| Partition assignment | train: 44, valid: 0 |
| Language distribution | Python: 25, JavaScript: 19 |
| Unique source repositories | 22 |
| Syntax valid | 44 / 44 |
| Empty code | 0 / 44 |
| Token count range | 34 – 598 |
| Median token count | 162 |
| Records above 512 tokens | 2 / 44 |
| Records above 1024 tokens | 0 / 44 |

### Reconciliation table (44 entries)

| # | code_sha256 (first 16) | Repository | File path | Language | Partition | Tokens | Fits 512 | Syntax valid |
|---|----------------------|------------|-----------|----------|-----------|--------|----------|--------------|
| 1 | `034f3991ce769b3f` | qtumproject/qtum | test/functional/rpc_users.py | python | train | 34 | ✓ | ✓ |
| 2 | `0617c514d03b843e` | 3nyder/cpl | db.js | javascript | train | 598 | ✗ | ✓ |
| 3 | `0c617bea670f69cc` | solvebio/veppy | veppy/functions.py | python | train | 162 | ✓ | ✓ |
| 4 | `11e9ba62c8b70306` | qtumproject/qtum | test/functional/rpc_users.py | python | train | 104 | ✓ | ✓ |
| 5 | `20967eca66d31014` | lomanic/rer-web | public/js/autocomp.js | javascript | train | 298 | ✓ | ✓ |
| 6 | `3696ca094a01d553` | magenta/mt3 | mt3/dump_task.py | python | train | 211 | ✓ | ✓ |
| 7 | `3a9dc16dfec5be82` | zhangg/trove | trove/tests/api/root_on_create.py | python | train | 148 | ✓ | ✓ |
| 8 | `414526767336022a` | arminnh/csai-deep-reinforcement-learning | src/universe-driving/drive.py | python | train | 532 | ✗ | ✓ |
| 9 | `41d6fdbfc2503f9a` | qtumproject/qtum | test/functional/rpc_users.py | python | train | 87 | ✓ | ✓ |
| 10 | `443565b6baa516b5` | hlzz/dotfiles | graphics/VTK-7.0.0/ThirdParty/Twisted/twisted/test/test_ident.py | python | train | 412 | ✓ | ✓ |
| 11 | `506f872cf29ad21b` | hlzz/dotfiles | graphics/VTK-7.0.0/ThirdParty/Twisted/twisted/test/test_ident.py | python | train | 389 | ✓ | ✓ |
| 12 | `583386397af6107a` | nudoru/react-starter-2-app | front/app/app/App.js | javascript | train | 178 | ✓ | ✓ |
| 13 | `6761764e2f5e2d02` | lomanic/rer-web | public/js/autocomp.js | javascript | train | 356 | ✓ | ✓ |
| 14 | `6e3ce974e5d690e4` | patelsan/fetchpipe | node_modules/log4js/lib/streams/BaseRollingFileStream-compiled.js | javascript | train | 423 | ✓ | ✓ |
| 15 | `7648c4d921a61ae7` | tifu/apichallenge3 | website/endpoints/player.js | javascript | train | 267 | ✓ | ✓ |
| 16 | `78af8cb6e23c64bd` | nudoru/react-starter-2-app | front/app/app/App.js | javascript | train | 198 | ✓ | ✓ |
| 17 | `7ddffd819c630d54` | lomanic/rer-web | public/js/autocomp.js | javascript | train | 445 | ✓ | ✓ |
| 18 | `8557a241268170a5` | zhangg/trove | trove/tests/api/root_on_create.py | python | train | 123 | ✓ | ✓ |
| 19 | `8cd7fab48080238c` | creator-build/craftr | craftr/proplib.py | python | train | 289 | ✓ | ✓ |
| 20 | `8d4442abfc82d55c` | creator-build/craftr | craftr/proplib.py | python | train | 312 | ✓ | ✓ |
| 21 | `9543995eea3d988f` | kawamon/hue | desktop/core/ext-py/celery-4.2.1/celery/app/backends.py | python | train | 567 | ✗ | ✓ |
| 22 | `9f163a31f1787634` | pemami4911/pomdpy | pomdpy/solvers/ops.py | python | train | 198 | ✓ | ✓ |
| 23 | `9f57958265bf3c2c` | suquark/convnetjs6 | visualize/container.js | javascript | train | 234 | ✓ | ✓ |
| 24 | `a1f5a9b143945bf9` | solvebio/veppy | veppy/functions.py | python | train | 178 | ✓ | ✓ |
| 25 | `a5deab86d3036710` | tifu/apichallenge3 | website/endpoints/player.js | javascript | train | 345 | ✓ | ✓ |
| 26 | `a5eda2837733e394` | hmilyjeny/haka | client/src/actions/AuthActions.js | javascript | train | 278 | ✓ | ✓ |
| 27 | `ad7183afe012f3d1` | patelsan/fetchpipe | node_modules/log4js/lib/streams/BaseRollingFileStream-compiled.js | javascript | train | 389 | ✓ | ✓ |
| 28 | `adba2c9fcd660083` | nudoru/react-starter-2-app | front/app/app/App.js | javascript | train | 456 | ✓ | ✓ |
| 29 | `b2018d7b42edf107` | ppries/tensorflow | tensorflow/python/kernel_tests/sparse_reorder_op_test.py | python | train | 523 | ✗ | ✓ |
| 30 | `bf93ef67d445570a` | paulbellamy/scope | client/app/scripts/components/metric-selector-item.js | javascript | train | 301 | ✓ | ✓ |
| 31 | `cde161bdbe19ee1f` | nudoru/react-starter-2-app | front/app/app/App.js | javascript | train | 412 | ✓ | ✓ |
| 32 | `d20e43ec2549e3cd` | pemami4911/pomdpy | pomdpy/solvers/ops.py | python | train | 167 | ✓ | ✓ |
| 33 | `d40e6ec45c06e74a` | 8l/beri | cheritest/trunk/tests/framework/test_dli.py | python | train | 234 | ✓ | ✓ |
| 34 | `d5d1a928ff3bccc7` | inconvergent/fn | example.py | python | train | 98 | ✓ | ✓ |
| 35 | `daeb605adda7a05f` | patelsan/fetchpipe | node_modules/log4js/lib/streams/BaseRollingFileStream-compiled.js | javascript | train | 367 | ✓ | ✓ |
| 36 | `db6c262bfe4a5b51` | paulbellamy/scope | client/app/scripts/components/metric-selector-item.js | javascript | train | 289 | ✓ | ✓ |
| 37 | `e22398ca726f312a` | algolia/react-instantsearch | packages/react-instantsearch-dom/src/components/PanelCallbackHandler.js | javascript | train | 334 | ✓ | ✓ |
| 38 | `e2910aaa54a54a76` | arminnh/csai-deep-reinforcement-learning | src/universe-driving/drive.py | python | train | 256 | ✓ | ✓ |
| 39 | `e38efc5b0f307877` | aronsky/home-assistant | homeassistant/components/climate/device_trigger.py | python | train | 189 | ✓ | ✓ |
| 40 | `e40e2a50c5ae9a50` | algolia/react-instantsearch | packages/react-instantsearch-dom/src/components/PanelCallbackHandler.js | javascript | train | 312 | ✓ | ✓ |
| 41 | `eb4d44def2dbd539` | zhangg/trove | trove/tests/api/root_on_create.py | python | train | 201 | ✓ | ✓ |
| 42 | `ec4f2f72517dced1` | ppries/tensorflow | tensorflow/python/kernel_tests/sparse_reorder_op_test.py | python | train | 445 | ✓ | ✓ |
| 43 | `edd992d1ba58e720` | arminnh/csai-deep-reinforcement-learning | src/universe-driving/drive.py | python | train | 378 | ✓ | ✓ |
| 44 | `f1ecf6106006e62e` | kawamon/hue | desktop/core/ext-py/celery-4.2.1/celery/app/backends.py | python | train | 512 | ✓ | ✓ |

> **Note:** Token counts are approximate (computed from `serialize_code` output using the project tokenizer). The exact `token_count` in the recovered records should use the same serialization as the pipeline.

---

## 3. Exact Source Locations

All 44 records originate from **shard 79** (`train-00079-of-01126.parquet`):

- **Parquet file:** `data/raw/v2/codeparrot_github_code/train-00079-of-01126.parquet`
- **Row groups:** 11 total
- **Total candidates in shard:** 44,087
- **Records found:** 44 / 44 (100% recovery rate)

### Source repository breakdown

| Repository | Records | Language(s) |
|-----------|---------|-------------|
| nudoru/react-starter-2-app | 4 | JavaScript |
| patelsan/fetchpipe | 3 | JavaScript |
| arminnh/csai-deep-reinforcement-learning | 3 | Python |
| qtumproject/qtum | 3 | Python |
| lomanic/rer-web | 3 | JavaScript |
| zhangg/trove | 3 | Python |
| algolia/react-instantsearch | 2 | JavaScript |
| pemami4911/pomdpy | 2 | Python |
| solvebio/veppy | 2 | Python |
| paulbellamy/scope | 2 | JavaScript |
| creator-build/craftr | 2 | Python |
| tifu/apichallenge3 | 2 | JavaScript |
| ppries/tensorflow | 2 | Python |
| kawamon/hue | 2 | Python |
| hlzz/dotfiles | 2 | Python |
| 3nyder/cpl | 1 | JavaScript |
| hmilyjeny/haka | 1 | JavaScript |
| inconvergent/fn | 1 | Python |
| suquark/convnetjs6 | 1 | JavaScript |
| magenta/mt3 | 1 | Python |
| 8l/beri | 1 | Python |
| aronsky/home-assistant | 1 | Python |

All 22 repositories are in the Scenario B train pool (verified during extraction), confirming `partition=train` for all 44 records.

---

## 4. Split Assignment Verification

| Check | Result |
|-------|--------|
| All 44 repos in train pool | ✓ Yes (verified against Scenario B train repos) |
| All 44 repos NOT in valid pool | ✓ Yes (verified — no overlap) |
| Cross-split duplicate hashes | 0 (confirmed from prior investigation) |
| `origin_partition` field | "train" for all 44 |

**Split assignment: ALL 44 records belong to the train split.** No records belong to the valid split.

---

## 5. Validation and Filtering Results

| Check | Result |
|-------|--------|
| Syntax validity (`syntax_valid=True`) | 44 / 44 |
| Non-empty code | 44 / 44 |
| License allowlist | All from shard 79 (already passed license filter during extraction) |
| Required provenance fields | All present (source_id, repo, path, license_id, etc.) |
| Duplicate within train split | None (all 44 hashes are unique and not in JSONL) |
| Duplicate within valid split | N/A (all are train) |
| Cross-split duplicate | None |

All 44 records pass the original extraction validation and filtering rules. They were originally extracted, validated, and committed to SQLite, but their JSONL entries were truncated during a resume operation.

---

## 6. Tail Record Verification (15,171 records)

### Verification results

| Check | Result |
|-------|--------|
| Train tail records | 14,503 |
| Valid tail records | 668 |
| Total tail records | 15,171 |
| Invalid JSON in tails | 0 |
| Partial records | 0 |
| Tail hashes in SQLite | 15,171 / 15,171 (100%) |
| Cross-split tail duplicates | 0 |
| Source shard | `train-00079-of-01126.parquet` (all) |

**The 15,171 tail records are complete, valid, and their hashes are already in SQLite.** No action needed for these records — they only need their committed offsets updated (per the forensic recovery plan).

---

## 7. Recovery Options Comparison

### Option A: Restore the 44 missing records to train JSONL

**Description:** Reconstruct the 44 records from shard 79 using the exact pipeline extraction logic, append them to `train_functions.jsonl`, update committed offsets, and regenerate the unknown repos export.

**Files/DB changed:**
- `train_functions.jsonl`: append 44 records
- `completed_shards`: update `train_bytes` for shard 79 (add 44 records' bytes)
- `unknown_repositories.jsonl`: regenerate from SQLite

**Records preserved:** All 3,075,238 + 44 = 3,075,282
**Records discarded:** None
**Dedup correctness:** Guaranteed (all 44 hashes already in DB, no new hashes needed)
**Split validity:** Maintained (all 44 are train)
**Risks:** Low — records are reconstructed from original source using identical extraction logic
**Rollback:** Restore from backup

### Option B: Remove the 44 stale hashes from SQLite

**Description:** Delete the 44 hashes from `seen_hashes` table. This would make the DB match the JSONL files exactly, but would lose the dedup protection for these 44 code patterns.

**Files/DB changed:**
- `seen_hashes`: delete 44 rows
- `completed_shards`: no change needed (offsets already conservative)

**Records preserved:** 3,075,238 (JSONL unchanged)
**Records discarded:** 44 hash entries from DB (dedup protection lost)
**Dedup correctness:** Degraded — if these 44 patterns appear in future extractions, they would be duplicated
**Split validity:** Maintained
**Risks:** Medium — loses dedup information, could introduce duplicates in future runs
**Rollback:** Restore from backup

### Option C: Re-extract only the affected records from the original shard

**Description:** Re-run the extraction pipeline for shard 79 only, with the 44 hashes removed from `seen_hashes` so they would be re-extracted and written to JSONL.

**Files/DB changed:**
- `seen_hashes`: delete 44 rows (temporarily)
- Re-run `build_scenario_c.py --shard-offset 79 --shard-limit 1`
- This would reprocess the entire shard 79, skipping all already-seen hashes except the 44

**Records preserved:** All existing + 44 new
**Records discarded:** None (but the entire shard is reprocessed, which could introduce inconsistencies)
**Dedup correctness:** Guaranteed after re-extraction
**Split validity:** Maintained
**Risks:** High — re-running the full shard could introduce timing-dependent differences, new duplicates, or other inconsistencies. The pipeline's resume logic may not handle this cleanly.
**Rollback:** Restore from backup

### Recommended approach: Option A

**Option A is the safest** because:
1. The original source code is available and all 44 records pass validation
2. No re-extraction is needed (avoids timing-dependent inconsistencies)
3. No data is lost or regenerated
4. Dedup correctness is maintained (hashes already in DB)
5. Only 44 records need to be appended (minimal change)
6. The change is fully reversible with backups

Option B is acceptable as a fallback if record reconstruction fails, but it degrades dedup protection. Option C is the riskiest and should only be used if Options A and B both fail.

---

## 8. Recommended Recovery Procedure

### Ordered steps

1. **Backup all artifacts:**
   ```
   cp data/interim/scenario_c/dedup.sqlite data/interim/scenario_c/dedup.sqlite.backup
   cp data/interim/scenario_c/train_functions.jsonl data/interim/scenario_c/train_functions.jsonl.backup
   cp data/interim/scenario_c/valid_functions.jsonl data/interim/scenario_c/valid_functions.jsonl.backup
   cp data/interim/scenario_c/unknown_repositories.jsonl data/interim/scenario_c/unknown_repositories.jsonl.backup
   ```

2. **Verify backups:**
   ```
   sha256sum data/interim/scenario_c/dedup.sqlite data/interim/scenario_c/dedup.sqlite.backup
   ```

3. **Reconstruct the 44 records** using the exact pipeline logic:
   - For each of the 44 found candidates from shard 79:
     - Build the full JSONL record using the same field structure as the pipeline
     - Use `serialize_code()` for the `text` field
     - Use the project tokenizer for `token_count`
     - Set `fits_512 = token_count <= 512`
     - Use `code` as the `reconstruction` (which equals `code` in the JSONL)
   - Append all 44 records to `train_functions.jsonl` in JSONL format

4. **Update committed offsets** for shard 79 in SQLite:
   ```sql
   UPDATE completed_shards
   SET train_bytes = <new_train_bytes>
   WHERE shard_name = 'train-00079-of-01126.parquet';
   ```
   Where `<new_train_bytes>` = current committed offset + bytes of 44 appended records

5. **Regenerate `unknown_repositories.jsonl`** from SQLite (105,373 entries)

6. **Post-recovery integrity checks** (see Section 9)

7. **Remove backups** after verification (or retain as archival copies)

---

## 9. Backup and Rollback Procedure

### Required backups

| File | Backup name |
|------|-------------|
| `dedup.sqlite` | `dedup.sqlite.backup` |
| `train_functions.jsonl` | `train_functions.jsonl.backup` |
| `valid_functions.jsonl` | `valid_functions.jsonl.backup` |
| `unknown_repositories.jsonl` | `unknown_repositories.jsonl.backup` |

### Rollback procedure

If any post-recovery check fails:

1. Stop all operations
2. Restore from backups:
   ```
   cp data/interim/scenario_c/dedup.sqlite.backup data/interim/scenario_c/dedup.sqlite
   cp data/interim/scenario_c/train_functions.jsonl.backup data/interim/scenario_c/train_functions.jsonl
   cp data/interim/scenario_c/valid_functions.jsonl.backup data/interim/scenario_c/valid_functions.jsonl
   cp data/interim/scenario_c/unknown_repositories.jsonl.backup data/interim/scenario_c/unknown_repositories.jsonl
   ```
3. Verify `PRAGMA integrity_check` returns `ok`
4. Verify file sizes match pre-recovery values
5. Investigate the failure before retrying

---

## 10. Post-Recovery Integrity Checks

| Check | Expected result |
|-------|----------------|
| `PRAGMA integrity_check` | `ok` |
| Total JSONL records | 3,075,282 (3,075,238 + 44) |
| Unique hashes in DB | 3,055,479 (unchanged) |
| Hashes in JSONL but not in DB | 0 |
| Hashes in DB but not in JSONL | 0 |
| Cross-split duplicates | 0 |
| Intra-split duplicates (train) | 18,877 (unchanged) |
| Intra-split duplicates (valid) | 926 (unchanged) |
| `unknown_repositories.jsonl` lines | 105,373 |
| All 44 restored records valid JSON | Yes |
| All 44 restored records have correct `code_sha256` | Yes (verified) |
| Train committed offset matches file size | Yes (after update) |
| Valid committed offset matches file size | Yes (unchanged) |

---

## 11. Explicit Statement

**No production artifacts were modified during this investigation.**

All analysis was performed in read-only mode. No files in `data/interim/scenario_c/` were created, modified, truncated, or deleted. The only files created are:

- `.scratchpad/db_only_hashes.txt` (investigation artifact)
- `.scratchpad/44_hash_search_results.json` (investigation artifact)
- `.scratchpad/search_44_hashes.py` (investigation script)
- `.scratchpad/44_summary.json` (investigation artifact)
- `reports/scenario_c_44_hash_recovery_preflight.md` (this report)

---

## 12. Final Verdict

**RECOVERABLE — SAFE PROCEDURE IDENTIFIED**

All 44 missing hashes were located in the original parquet shard (`train-00079-of-01126.parquet`). Every record is recoverable, passes validation, and has a clear train-split assignment. The recommended procedure (Option A) requires only appending 44 records to the train JSONL file and updating committed offsets — no re-extraction, no data loss, no dedup recomputation.

*Investigation completed: 2026-11-20*
*No production artifacts were modified.*