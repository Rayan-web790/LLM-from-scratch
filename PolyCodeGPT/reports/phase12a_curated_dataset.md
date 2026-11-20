# Phase 12A — Curated JavaScript Candidate Dataset

Output: `data/curated/js_v1/train.jsonl`
SHA-256: `b0ae31c0d1757d120e84fc62d2fa4e7b8aa891e1487fd0b5f00767279d9e6bd7`
Records: 35,658

From 102,169 original JS training records, 35,658 survived (34.9%).

Every original field is preserved verbatim; no record was rewritten.
Exclusions are listed in `data/curated/js_v1/rejected.csv` (one row per rejected record).

## Filtering rules applied (in order)

1. `parse_invalid` — V8 rejected the snippet (syntax error)
1. `not_standalone_function` — not a single complete standalone function definition
1. `description_missing` — no description text
1. `description_fragment` — description is a comment fragment, not usable natural language
1. `description_boilerplate` — description is a licence/copyright preamble
1. `depends_on_ecosystem` — requires modules, imports/exports, or host globals
1. `stateful_or_method` — uses `this`/object state, so not a self-contained function
1. `code_too_short` — fewer than 12 code tokens
1. `code_too_long` — more than 512 serialized tokens
1. `name_mismatch` — declared function name disagrees with the corpus func_name
1. `duplicate` — identical normalized code already kept
1. `near_duplicate` — near-identical code already kept (minhash LSH)

Thresholds: MIN_DESCRIPTION_WORDS=4, MIN_DESCRIPTION_CHARS=20, MAX_DESCRIPTION_CHARS=1200, MIN_ALPHA_RATIO=0.5, MIN_CODE_TOKENS=12, MAX_CODE_TOKENS=500, MAX_SERIALIZED_TOKENS=512, NEAR_DUP=15 minhash values over 5-token shingles, 3 bands of 5
