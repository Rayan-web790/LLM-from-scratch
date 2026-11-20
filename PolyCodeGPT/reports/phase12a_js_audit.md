# Phase 12A — JavaScript Training Data Audit

Source: `data/processed/polycodegpt_corpus/train.jsonl` (training split only; validation and test splits were not read).

- Original records: **102,169**
- Eligible (curated): **35,658** (34.9%)
- Rejected: **66,511**

## Rejection reasons

| reason | count | meaning |
|---|---:|---|
| `parse_invalid` | 32,830 | V8 rejected the snippet (syntax error) |
| `not_standalone_function` | 497 | not a single complete standalone function definition |
| `description_missing` | 0 | no description text |
| `description_fragment` | 12,309 | description is a comment fragment, not usable natural language |
| `description_boilerplate` | 367 | description is a licence/copyright preamble |
| `depends_on_ecosystem` | 6,679 | requires modules, imports/exports, or host globals |
| `stateful_or_method` | 11,253 | uses `this`/object state, so not a self-contained function |
| `code_too_short` | 0 | fewer than 12 code tokens |
| `code_too_long` | 2,399 | more than 512 serialized tokens |
| `name_mismatch` | 105 | declared function name disagrees with the corpus func_name |
| `duplicate` | 0 | identical normalized code already kept |
| `near_duplicate` | 72 | near-identical code already kept (minhash LSH) |

## Parse status (V8 compile, no execution)

| status | count |
|---|---:|
| invalid | 32,830 |
| valid_script | 69,339 |

## Structural kinds

| kind | count |
|---|---:|
| function_decl | 102,167 |
| other | 2 |

## Description categories

| category | count |
|---|---:|
| plain_text | 100,389 |
| line_comment | 1,773 |
| jsdoc_block | 7 |

## Description quality

| quality | count |
|---|---:|
| meaningful | 83,668 |
| fragment | 18,053 |
| boilerplate | 448 |

## Duplicates

- Exact (normalized whitespace): 0
- Near (minhash LSH): 72

## Token lengths

| set | min | p25 | median | p75 | p95 | max | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| all serialized | 33 | 113 | 178 | 309 | 816 | 624879 | 309 |
| curated serialized | 36 | 106 | 154 | 236 | 402 | 512 | 183 |
| all code tokens | 22 | 56 | 91 | 164 | 443 | 671620 | 167 |
| curated code tokens | 22 | 51 | 76 | 123 | 229 | 441 | 96 |

## Repositories

- Distinct repos (original): 14,738
- Distinct repos (curated): 8,341

Top curated repos:

| repo | records |
|---|---:|
| eslint/eslint | 555 |
| angular/material | 233 |
| CuppaLabs/angular2-multiselect-dropdown | 194 |
| eventbrite/britecharts | 178 |
| GitbookIO/gitbook | 164 |
| TerriaJS/terriajs | 137 |
| feedhenry/fh-forms | 131 |
| sadiqhabib/tinymce-dist | 127 |
| ArnaudBuchholz/gpf-js | 123 |
| cemtopkaya/kuark-db | 122 |
| nhn/tui.editor | 102 |
| prebid/Prebid.js | 102 |
| clay/amphora | 100 |
| ota-meshi/eslint-plugin-lodash-template | 94 |
| zhanziyang/vue-croppa | 87 |

Filtering rules and thresholds are in `experiments/phase12a/rules.py`.
