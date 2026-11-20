# PolyCodeGPT — Phase 17: Data Engineering 2.0

**Status:** Design complete; read-only audit only  
**Date:** 2026-11-18  
**Scope:** Next-generation Python/JavaScript corpus design for a from-scratch decoder-only model  
**Baseline:** Exp C checkpoint and v0.1.0 package are immutable historical artifacts

---

## 0. Executive summary

Phase 16 found that the next generation should improve data quality, diversity, and coverage before increasing model capacity. The Phase 17 audit confirms that finding. The current corpus is a useful but narrow function-level baseline: **392,632 training records, 133,192,957 training tokens, 24,517 training repositories**, dominated by CodeSearchNet Python and a small, heavily repeated JavaScript subset.

The most important measured constraints are:

- **JavaScript is the binding constraint.** The corpus contains only **6,542,851 unique curated JavaScript tokens** (35,658 records, 8,341 repositories), repeated **9.96×** in Exp C. Even recovering every parse-valid CodeSearchNet JavaScript record would yield only about **31.5M unique JavaScript tokens**, far below what a balanced intermediate corpus needs.
- **The corpus is structurally narrow.** The JavaScript source is **102,167 `function_decl` records out of 102,169**, with only **7 JSDoc blocks**. Curated JavaScript deliberately removes methods, stateful code, ecosystem-dependent code, and long code. There is little class, module, async, test, or multi-file context.
- **Duplication and length handling are weak.** Python has no near-deduplication pass and **21,676 duplicate-docstring extra occurrences**; **6,931 Python and 22,507 JavaScript records** have a repeated-line ratio above 0.2. Python documentation is only **2.14%** of training tokens, and **85%** of documentation files exceed 512 tokens.
- **Evaluation mismatch is real.** In the Phase 14 Exp C evaluation, only **4/100 complete tasks** and **39/328 cases** passed; JavaScript syntax failures were dominant (**23/50** JavaScript syntax checks passed, versus **40/50** for Python). The current data does not teach the structures the benchmark asks for.
- **Source availability is not the same as usable data.** Several large public sources are promising, but The Stack v1/v2 and StarCoderData are gated; CodeParrot contains copyleft and low-quality files; APPS/CodeContests/MBPP are evaluation-first and carry contamination risk; MDN prose is CC-BY-SA; BugsInPy has no clear license. None of these is approved merely because it is publicly accessible.

### Recommendation

Adopt **Scenario B: 500M unique training tokens**, split as **250M Python / 225M JavaScript / 25M documentation**, approximately **1.5M code records**, and **~75,000 unique repositories**, **conditional on** licensed JavaScript expansion and successful quality gates. This is the smallest scenario that materially fixes the JavaScript and structural-coverage deficits while remaining feasible on the existing **RTX 5060 Laptop, 8 GB VRAM** with the Phase 16 intermediate model (≈65M parameters, 4–6 GB, 9–14k tokens/s, 6–18 hours per 300–600M-token pass).

Use **Scenario A (200M unique tokens)** as the fallback if the JavaScript sources cannot be licensed and validated at the required scale. Defer **Scenario C (1B unique tokens)** until storage, legal review, and compute are materially expanded. Do not fine-tune Exp C; all future models must be trained from scratch. Do not modify the v0.1.0 package, tokenizer, serialization pipeline, source, checkpoints, old test partition, or `functional/independent_v3/` artifacts.

No dataset was downloaded, created, transformed, or trained on during this assignment. No existing artifact was modified.

---

## 1. Phase 16 findings validated against the manifests

| Phase 16 finding | Evidence checked in Phase 17 | Implication |
|---|---|---|
| Narrow function-level distribution | CodeSearchNet JavaScript is **102,167 `function_decl` records out of 102,169**; only **7 JSDoc blocks** | The corpus teaches function bodies, not classes, modules, async flows, or applications |
| Limited unique JavaScript coverage | Curated JavaScript is **35,658 records / 6,542,851 tokens / 8,341 repos**; Exp C repeats it **9.96×** | More JavaScript must be new and unique, not oversampled |
| Excessive dependence on standalone functions | Curated JavaScript rejects methods, stateful code, ecosystem-dependent code, and long code; only **35,658** of **69,339** parse-valid records survive | The strict filter removes useful programming knowledge along with noise |
| Programming-task distribution mismatch | Phase 14 Exp C: **4/100 complete tasks**, **39/328 cases**; JavaScript syntax **23/50** | The data distribution does not match the requested programming behavior |
| Repeated-record exposure | Exp C stage unique tokens: **74,562,809** of **133,192,957** total; JavaScript repeated **9.96×** | More epochs on the same small JavaScript set cannot create new competence |
| Weak larger-structure coverage | No class/module/async/test category in the current schema; blocks are split at function level | Larger examples and multi-file relationships are absent or fragmented |
| Python quality issues | **21,676 duplicate-docstring extra occurrences**; **6,931 Python records** with repeated-line ratio > 0.2 | Python needs deduplication and repetition controls too, not only JavaScript work |
| Documentation exposure is low | **500 files / 2,847,020 tokens / 12.7 MB**, **2.14%** of training tokens; **85%** of docs exceed 512 tokens | Documentation is underused and poorly chunked |
| Syntax failures are a major failure mode | **32,830 JavaScript parse-invalid records** in curation; JavaScript syntax failures dominate Exp C | Syntax validity must be a mandatory gate, not a preference |

The Phase 16 diagnosis is therefore consistent with the manifests. The redesign should not assume that a larger model alone will fix these gaps.

---

## 2. Existing source inventory

### 2.1 Current training corpus

Counts below are from the existing manifests and training partition. The old test partition was **not opened** and is not used for sizing or design.

| Component | Records | Training tokens | Repositories | License/provenance | Current status |
|---|---:|---:|---:|---|---|
| CodeSearchNet Python (filtered) | 320,515 total; 290,507 train | 98.8M in train corpus | 10,862 total | Per-repository permissive filter; Zenodo v1 source; S3 v2 parity not established | `FILTERED_PENDING_REVIEW` |
| CodeSearchNet JavaScript (filtered) | 114,162 total; 102,169 train | 31.5M in train corpus | 16,366 total | Per-repository permissive filter; Zenodo v1 source; S3 v2 parity not established | `FILTERED_PENDING_REVIEW` |
| Curated JavaScript v1 | 35,658 | 6,542,851 unique | 8,341 | Derived from filtered CodeSearchNet JavaScript; no new license | Historical baseline |
| Python documentation (English) | 500 files | 2,847,020 | 1 (CPython docs) | PSF-2.0 prose; code examples dual PSF-2.0/0BSD | `APPROVED_CANDIDATE` |
| **Combined v0.1.0 training corpus** | **392,632** | **133,192,957** | **24,517** | Mixed; per-source registry | Historical baseline |
| Validation partition | 20,965 | Not used for sizing | 1,352 | Same source filters | Historical baseline |
| Test partition | Not opened | Not opened | Not opened | Same source filters | Must remain untouched |

The current corpus is **2.07 GB** of training JSONL and **0.27 GB** of tokenized training `.bin`; validation is **0.115 GB** JSONL and **0.0155 GB** `.bin`. The whole `data/` tree is about **8.7 GB**, and the existing experiments/checkpoints/release artifacts occupy additional space (see §10).

### 2.2 Source registry status

The registry contains **14 entries**:

- **1 approved candidate:** Python official documentation.
- **2 filtered pending review:** CodeSearchNet Python and JavaScript Zenodo copies.
- **7 pending review:** CodeSearchNet Python/JavaScript registry entries, The Stack v1, The Stack v2, Python docs French, MDN, Stack Overflow.
- **3 unverified:** Python docs Arabic, MDN French, MDN Arabic.
- **1 rejected:** `free-programming-books` (links only, no content).

This distribution matters: the only source currently marked approved is Python documentation. The code sources that dominate the corpus are still pending review, and the large external sources are not approved.

### 2.3 Registry and candidate source details

| Source | Identity / provenance | Languages | Approx. volume | Format | License / redistribution | Quality limitations | Contamination / duplication risk | Usable now? |
|---|---|---|---|---|---|---|---|---|
| CSN-PY | CodeSearchNet v2 registry; filtered Zenodo v1 | Python | 320,515 accepted records; 10,862 repos | JSONL with provenance | Permissive per repo; attribution; registry pending | Function+docstring only; duplicate docstrings; no classes/modules | Repository/source overlap with any future CodeSearchNet use | Seed only; legal review pending |
| CSN-JS | CodeSearchNet v2 registry; filtered Zenodo v1 | JavaScript | 114,162 accepted records; 16,366 repos | JSONL with provenance | Permissive per repo; attribution; registry pending | 32,830 parse-invalid; 102,167/102,169 `function_decl`; 7 JSDoc | Same | Seed only; legal review pending |
| PYDOCS | CPython v3.11.9 docs sparse checkout | Python (docs) | 500 files; 2.847M tokens; 12.74 MB | reStructuredText + manifest | PSF-2.0; code examples PSF-2.0/0BSD; attribution | Long prose; 85% >512 tokens; not code-adjacent enough | Low; docs may describe benchmark tasks indirectly | Approved candidate |
| PYDOCS-FR | python-docs-fr | Python/French docs | ~56 MB source | PO/reStructuredText | PSF-2.0 + CC0 contributions; pending | Derivative translation; review needed | Low | No |
| PYDOCS-AR | python-docs-ar | Python/Arabic docs | Unverified | PO | PSF-2.0 derivative; unverified | Unverified license | Unknown | No |
| MDN | MDN Web Docs | JavaScript/web | ~100+ MB content repo | HTML/Markdown + code blocks | Prose CC-BY-SA 2.5; code CC0 post-2010-08-20, MIT earlier | Date/provenance ambiguity; prose ShareAlike; snippets often partial | High overlap with web examples and benchmarks | No; code-only after review |
| SE | Stack Overflow / Stack Exchange | Multiple | ~50+ GB full dump | XML/JSON | CC-BY-SA by date; attribution + ShareAlike | Q&A noise; code fragments; license complexity | Very high; widely copied code | No |
| STACK-V1 | BigCode The Stack v1.1/v1.2 | 358 languages | 3–6 TB near-dedup | Parquet | 193 permissive licenses; copyleft excluded; original licenses; gated HF + terms | File-level, not function-level; variable quality; generated/minified files | High; may contain benchmark solutions | No; gated and unapproved |
| STACK-V2 | BigCode The Stack v2 | 600+ languages | 6+ TB | Parquet + SWH IDs | Permissive/varies; original licenses; gated HF + Software Heritage agreement | Large, heterogeneous; SWH content retrieval; license metadata gaps | High | No; gated and unapproved |
| STARCODERDATA | BigCode StarCoderData | 86 languages | 783 GB; ~250B tokens | JSONL | Gated Stack terms; original licenses | Includes issues, notebooks, commits; noisy; near-dedup/PII | Very high | No; gated |
| CODEPARROT | CodeParrot `github-code` | 32 languages | 115M files; ~1 TB | HF dataset/Parquet | 15 license types including copyleft; per-file metadata | Lower quality; generated/vendored/minified; license mix | High | After allowlist and quality filtering |
| COMMITPACKFT | CommitPackFT | 277 languages | ~2 GB; 702.1K rows | JSONL | MIT compilation; per-sample licenses | Commit-level, not standalone examples; noisy diffs | High; benchmark code may appear in commits | After per-sample license filter |
| CPYTHON | CPython repository | Python | Full stdlib/tests/docs | Git/source | PSF-2.0; docs examples 0BSD | Tests are not tutorials; docs examples need extraction | Medium; benchmark tasks may derive from stdlib | Legal review; code/tests/docs separately |
| NODE | Node.js repository | JavaScript | Full core/tests/docs | Git/source | MIT core; bundled third-party notices | Tests are framework-specific; docs partial | Medium | Legal review; docs/tests separately |
| APPS | APPS benchmark repository | Python | 10k problems; ~250k solutions; ~1.3 GB | JSON | Repo MIT; problem statements may carry third-party terms | Competed/crowd-sourced; variable quality; solution leakage | Very high; benchmark contamination | Evaluation-first; legal review |
| CODECONTESTS | DeepMind CodeContests | Python/C++/Java | ~4,044 problems in current release | JSON | Code Apache-2.0; non-code CC-BY-4.0 | Competitive programming; not application code | Very high | Evaluation-first |
| MBPP | Mostly Basic Python Problems | Python | 974 tasks; 831 KB | JSONL | CC-BY-4.0 | Small; basic only; benchmark | Very high | Evaluation-only |
| BUGSJS | BugsJS | JavaScript | 453 bugs; 10 Node projects; 444k LOC | JSON/repo | Repo MIT; project code varies | Bug-fix pairs, not general corpus; project-specific | High | Evaluation-first; per-project review |
| BUGSINPY | BugsInPy | Python | 493 bugs; 17 projects | JSON/repo | License not clearly stated | No clear license; project-specific | High | No |
| STACKOVERFLOW | Stack Exchange dump | Multiple | ~50+ GB | XML | CC-BY-SA; attribution/ShareAlike | Fragments, noise, duplication | Very high | No |

**Source-availability note:** The Stack v1/v2 and StarCoderData are source-available but gated. CodeParrot and CommitPackFT are publicly downloadable but still require per-file/per-sample license filtering. The table above describes source-reported volumes, not locally validated usable volumes.

### 2.4 What is actually usable today

Today, without new approvals or downloads, the usable seed is:

- The current filtered CodeSearchNet train/valid records.
- The 500 Python documentation files.
- The existing curated JavaScript records, which are a subset of CodeSearchNet JavaScript.

That seed cannot reach Scenario B on its own. Even using every parse-valid CodeSearchNet JavaScript record and every current Python record, the corpus remains far below 500M unique tokens and remains structurally narrow. The large external sources are **candidates**, not inventory.

---

## 3. Identified data deficiencies

| ID | Deficiency | Severity | Evidence | Required correction |
|---|---|---|---|---|
| D1 | Function-level monoculture | High | 102,167/102,169 JavaScript records are `function_decl`; schema has one docstring-to-code task | Add class, module, file, multi-file, tests, fill, explain, and fix tasks |
| D2 | JavaScript supply is too small | High | 6.54M unique curated tokens; 9.96× repetition | Acquire licensed, unique JavaScript; recover filtered valid records with context |
| D3 | Strict curation removes useful code | High | 33,681 parse-valid records rejected for description, state, ecosystem, length, or standalone rules | Replace binary rejection with labeled classification and context-aware tasks |
| D4 | Duplicate and repeated content | High | 21,676 duplicate-docstring extras; 6,931 Python / 22,507 JavaScript repeated-line >0.2; JavaScript 9.96× | Global exact/normalized/near-dedup; per-record exposure caps |
| D5 | Documentation is underused and badly chunked | Medium | 2.14% of tokens; 85% of files >512 tokens | Code-adjacent extraction; 5–8% target; AST/section-aware chunks |
| D6 | No module/import context | High | Code records are isolated functions; no imports or file context | Add imports/file/repo context and module-level tasks |
| D7 | Almost no tests or debugging data | High | No test/debug task category in the schema; Phase 14 failure modes are runtime/incorrect output | Add tests, expected behavior, buggy/fix, and edge-case examples |
| D8 | Task distribution mismatch | High | 4/100 complete tasks; 39/328 cases; JavaScript syntax 23/50 | New category quotas and a new benchmark designed before training |
| D9 | License/provenance is incomplete | High | Only 1 approved source; 2 filtered pending; large sources gated/unapproved | Registry approval, SPDX allowlist, per-file provenance, attribution |
| D10 | Weak larger real-world structures | High | Blocks split at function level; no multi-function modules | Add 50–300-line modules, class hierarchies, and 2–3-file examples |
| D11 | Syntax validity is not universal | High | 32,830 JavaScript parse-invalid records; JavaScript syntax failures dominate Exp C | Mandatory static parse gate; quarantine invalid records |
| D12 | Benchmark reuse risk | High | `independent_v3` was used for selection; old test must remain untouched | New frozen benchmark; no tuning; contamination checks |

The deficiencies are coupled. Adding more function-level records would improve D2 only superficially; it would not fix D1, D6, D7, or D10. The data redesign must change **what a record is**, not only how many records exist.

---

## 4. Proposed dataset schema

### 4.1 Decision

**Keep the v1 schema frozen for v0.1.0. Introduce a v2 schema for the next from-scratch generation.** Do not modify the current tokenizer or serialization pipeline in Phase 17. The v2 schema is a design proposal for Phase 18 dataset engineering.

The existing format:

```text
<|lang:X|><|docstring|>description<|code|>implementation<|eos|>
```

is adequate for one task: generate a function body from a docstring. It cannot express class context, imports, files, tests, partial completion, explanations, or repair. A richer envelope is justified because the next model is trained from scratch, so it can learn new special tokens without checkpoint compatibility constraints.

### 4.2 Proposed v2 envelope

Use a task-first envelope with explicit optional sections:

```text
<|bos|><|lang:python|><|task:function|><|docstring|>...<|code|>...<|eos|>
```

General form:

```text
<|bos|><|lang:LANG|><|task:TASK|>[optional sections in fixed order]<|eos|>
```

Fixed section order:

```text
<|file|>path
<|context|>surrounding code or repository context
<|imports|>import statements
<|docstring|>natural-language specification
<|signature|>declaration or expected signature
<|prefix|>code before the completion point
<|suffix|>code after the completion point
<|buggy|>code containing a defect
<|tests|>tests or assertions
<|code|>target implementation
<|explanation|>natural-language explanation
```

Not every section is present in every record. Sections must appear in the fixed order above so the model can learn boundaries rather than memorizing arbitrary layouts.

### 4.3 Task types

| Task tag | Meaning | Typical sections | Why it is needed |
|---|---|---|---|
| `<|task:function|>` | Generate a function from a docstring/signature | docstring, signature, code | Preserves the current core task |
| `<|task:class|>` | Generate a class or method set from a specification | context, imports, docstring, code | Teaches OOP, state, methods, inheritance |
| `<|task:module|>` | Generate a module/file from a spec or imports | file, imports, docstring, code | Teaches imports, exports, module structure |
| `<|task:repo|>` | Generate or complete a small multi-file example | file, context, imports, code | Teaches multi-file relationships within context limits |
| `<|task:fill|>` | Fill a prefix/suffix gap | prefix, suffix, code | Teaches partial completion and local context |
| `<|task:tests|>` | Generate tests or expected behavior | code, tests | Teaches assertions, edge cases, testing APIs |
| `<|task:explain|>` | Explain code | code, explanation | Teaches code understanding and docstring quality |
| `<|task:fix|>` | Repair buggy code given tests/spec | buggy, tests, code | Teaches debugging and error handling |

The task tag is the **primary objective**. Additional labels (language, category, construct, dependencies, difficulty) belong in metadata and manifests, not necessarily in the serialized text. This avoids turning the sequence into a metadata dump.

### 4.4 Proposed special tokens

The v2 tokenizer should reserve dedicated IDs for:

- **Control:** `<|bos|>`, `<|eos|>`, `<|pad|>`, `<|unk|>` (as required by the training code).
- **Language:** `<|lang:python|>`, `<|lang:javascript|>`.
- **Task:** `<|task:function|>`, `<|task:class|>`, `<|task:module|>`, `<|task:repo|>`, `<|task:fill|>`, `<|task:tests|>`, `<|task:explain|>`, `<|task:fix|>`.
- **Section:** `<|file|>`, `<|context|>`, `<|imports|>`, `<|docstring|>`, `<|signature|>`, `<|prefix|>`, `<|suffix|>`, `<|buggy|>`, `<|tests|>`, `<|code|>`, `<|explanation|>`.

**Justification:** These tokens are structural and frequent in v2. A from-scratch tokenizer can assign them dedicated IDs, and a from-scratch model learns embeddings for them normally. They are not rare Unicode strings that would fragment unpredictably. The current v1 tokenizer and serialization code remain untouched.

**Cost:** The embedding table grows by roughly two dozen rows, which is negligible for a 34M–110M parameter model. The real cost is dataset engineering and validation, not vocabulary size.

### 4.5 Schema examples

**Function generation (Python):**

```text
<|bos|><|lang:python|><|task:function|><|docstring|>Return the sum of all even integers in a list.<|signature|>def sum_even(values):<|code|>def sum_even(values):
    total = 0
    for value in values:
        if value % 2 == 0:
            total += value
    return total<|eos|>
```

**Class generation (JavaScript):**

```text
<|bos|><|lang:javascript|><|task:class|><|context|>A small inventory used by a store.<|imports|>import { formatPrice } from "./price.js";<|docstring|>Implement an Inventory class with add, remove, and total methods.<|code|>export class Inventory {
  constructor() { this.items = []; }
  add(item) { this.items.push(item); }
  remove(name) { this.items = this.items.filter((item) => item.name !== name); }
  total() { return this.items.reduce((sum, item) => sum + item.price, 0); }
}<|eos|>
```

**Module generation (JavaScript):**

```text
<|bos|><|lang:javascript|><|task:module|><|file|>src/math/stats.js<|imports|>import { mean } from "./mean.js";<|docstring|>Export functions for mean, variance, and standard deviation.<|code|>export { mean };
export function variance(values) { ... }
export function standardDeviation(values) { ... }<|eos|>
```

**Prefix/suffix fill (Python):**

```text
<|bos|><|lang:python|><|task:fill|><|prefix|>def clamp(value, low, high):
    if value < low:
        return low<|suffix|>    return value<|code|>    if value > high:
        return high<|eos|>
```

**Tests (JavaScript):**

```text
<|bos|><|lang:javascript|><|task:tests|><|code|>export function slugify(text) { ... }<|tests|>import { slugify } from "./slugify.js";
test("lowercases and replaces spaces", () => {
  expect(slugify("Hello World")).toBe("hello-world");
});<|eos|>
```

**Explanation-to-code (Python):**

```text
<|bos|><|lang:python|><|task:explain|><|code|>def memoize(fn):
    cache = {}
    def wrapper(*args):
        if args not in cache:
            cache[args] = fn(*args)
        return cache[args]
    return wrapper<|explanation|>This decorator caches results by positional arguments and returns a wrapper function.<|eos|>
```

**Bug fix (JavaScript):**

```text
<|bos|><|lang:javascript|><|task:fix|><|buggy|>function average(values) {
  return values.reduce((a, b) => a + b) / values.length;
}<|tests|>expect(average([])).toBe(0);<|code|>function average(values) {
  if (values.length === 0) return 0;
  return values.reduce((a, b) => a + b, 0) / values.length;
}<|eos|>
```

### 4.6 Compatibility and migration rules

1. **v1 is frozen.** Existing records, tokenizer, and serialization stay as they are for v0.1.0 reproducibility.
2. **v2 is a new corpus and tokenizer.** Train the v2 tokenizer on the v2 mixture, including special tokens; do not retrofit v1.
3. **No fabricated docstrings.** Convert a v1 function+docstring pair to `<|task:function|>` only when the docstring is genuine. If there is no docstring, use fill, code-only, or tests, not an invented explanation.
4. **No lossy concatenation.** Do not glue unrelated functions into a fake module. A module record must come from a real file or a coherent multi-function unit.
5. **Fixed section order.** Serialization must enforce order and reject malformed section sequences.
6. **Round-trip tests.** Parse a v2 record back into structured fields and compare to the manifest; fail on loss.
7. **Metadata stays external where possible.** Category, license, repo, and quality flags belong in manifests; only task/section/language tags need to be in the text.

---

## 5. Programming-category coverage plan

### 5.1 Evidence rules

Categories are **primary labels** with optional multi-label tags. Do not force an example into a category without evidence:

- A class with methods is `oop`; a method that is also async is `oop` + `async`.
- A tree implementation is `data_structures`; its traversal algorithm is `algorithms`.
- A file-reading script is `file_data`; if it uses `json` and `pathlib`, tag `stdlib`.
- A test file is `testing_debugging`; if it tests a class, tag `oop`.
- If the evidence is ambiguous, keep the record with `unclassified` metadata and exclude it from category quotas. Do not relabel it to satisfy a target.

### 5.2 Target category shares

Shares apply to **code tokens**, not documentation tokens. Python and JavaScript targets differ because their ecosystems and failure modes differ.

| # | Category | Python target | JavaScript target | Why the difference |
|---|---|---:|---:|---|
| 1 | Basic programming and expressions | 8% | 8% | Both need a foundation, but not the majority |
| 2 | Functions and recursion | 12% | 11% | Python recursion and closures; JavaScript functions/arrow functions |
| 3 | Algorithms and computational reasoning | 12% | 10% | Python has more algorithm-heavy educational code; JavaScript needs enough to generalize |
| 4 | Lists, arrays, strings, dictionaries | 16% | 16% | Highest-frequency practical category in both languages |
| 5 | Trees, graphs, stacks, queues, other structures | 8% | 8% | Necessary for non-trivial reasoning; not dominant |
| 6 | Classes, objects, OOP | 8% | 12% | JavaScript currently lacks classes/prototypes; more repair needed |
| 7 | Modules, imports, multi-file relationships | 7% | 8% | JavaScript module systems are more varied (`import`, `require`, exports) |
| 8 | Exceptions and error handling | 6% | 6% | Both need explicit error paths |
| 9 | File and data processing | 5% | 4% | Python is more file/data-oriented; JavaScript uses Node/browser APIs differently |
| 10 | Standard-library usage | 7% | 5% | Python stdlib is broader and more commonly used in examples |
| 11 | Asynchronous programming | 2% | 6% | JavaScript async is a core language competency; Python async is narrower |
| 12 | Testing, debugging, edge cases | 6% | 4% | Python testing ecosystem is more common in standalone examples |
| 13 | Larger real-world examples | 3% | 2% | Expensive to curate; important but not the bulk |
| | **Total** | **100%** | **100%** | |

Documentation exposure is separate: target **5–8% of training tokens**, code-adjacent, not raw prose. Scenario B uses **5% (25M tokens)**.

### 5.3 Absolute Scenario B targets

Scenario B code tokens: **250M Python + 225M JavaScript**. Applying the shares gives the following approximate primary-label targets. These are planning envelopes, not quotas that justify mislabeling.

| Category | Python (M tokens) | JavaScript (M tokens) |
|---|---:|---:|
| Basic | 20.0 | 18.0 |
| Functions/recursion | 30.0 | 24.75 |
| Algorithms | 30.0 | 22.5 |
| Collections | 40.0 | 36.0 |
| Data structures | 20.0 | 18.0 |
| OOP | 20.0 | 27.0 |
| Modules/imports | 17.5 | 18.0 |
| Exceptions | 15.0 | 13.5 |
| File/data | 12.5 | 9.0 |
| Standard library | 17.5 | 11.25 |
| Async | 5.0 | 13.5 |
| Testing/debugging | 15.0 | 9.0 |
| Larger examples | 7.5 | 4.5 |
| **Total** | **250.0** | **225.0** |

Scenarios A and C use the same shares, scaled to their token totals. This keeps comparisons clean; it does not imply that every category is equally easy to source.

### 5.4 Category census and coverage gates

Before sampling, run a **construct census** over candidate records:

- Python parse via `ast`; JavaScript parse via a pinned static parser.
- Extract function/class/module counts, imports, async markers, exception handlers, test framework markers, file I/O, and common stdlib/API symbols.
- Assign primary category from structural evidence; assign secondary tags separately.
- Report coverage by language, source, repository, and category.
- Require each category in each language to reach at least **60% of its target share** before the corpus is considered balanced; do not fill gaps by duplicating records.
- Mark underfilled categories explicitly in the manifest. A corpus with a documented gap is better than a corpus padded with mislabeled or duplicated examples.

---

## 6. JavaScript expansion strategy

### 6.1 The current ceiling

Current CodeSearchNet JavaScript contains **102,169 training records** and about **31.5M tokens**. Curation keeps **35,658 records / 6.54M tokens**. Even if every parse-valid record were recovered, the source would provide only about **31.5M unique tokens**, roughly **14%** of the 225M JavaScript target in Scenario B. JavaScript expansion therefore requires new sources, not just looser filtering.

### 6.2 Reclassify instead of binary reject

The current curation pipeline rejects a record if it is not a standalone, documented, stateless function. That rule is appropriate for a narrow v1 benchmark, but it discards useful programming knowledge. Replace it with a **classification system**:

| Current rejection reason | Count | Proposed handling | Required context |
|---|---:|---|---|
| `parse_invalid` | 32,830 | Reject; cannot safely train syntax | None |
| `description_fragment` | 12,309 | Keep code; mark description quality `fragment`; do not use docstring-to-code | Code-only or tests |
| `stateful_or_method` | 11,253 | Keep as method/class example; mark receiver/state | Class context, `this`, constructor |
| `depends_on_ecosystem` | 6,679 | Keep as module example with imports/package metadata | Imports, package.json, module context |
| `code_too_long` | 2,399 | Keep with AST-aware chunking; do not truncate mid-function | File/module context |
| `not_standalone_function` | 497 | Keep as module/file fragment; mark top-level | File context |
| `name_mismatch` | 105 | Quarantine; attempt AST-derived name repair only if unambiguous | Metadata repair |
| `near_duplicate` | 72 | Remove duplicate | Global dedup index |
| `description_boilerplate` | 367 | Keep code; drop boilerplate description | Code-only or tests |

This can recover up to **33,681 parse-valid records** currently discarded, but recovery is not free: each recovered record needs context or a new task type. A method without its class is often unintelligible; a module fragment without imports is often unparseable in context. The pipeline must attach context or reject the record.

### 6.3 JavaScript construct targets

The expanded JavaScript corpus should include, with measured counts:

- Function declarations and expressions.
- Arrow functions, closures, and callbacks.
- Classes, constructors, methods, `extends`, and prototypes.
- ES modules (`import`/`export`) and CommonJS (`require`/`module.exports`).
- Arrays, objects, `Map`, `Set`, strings, and iteration.
- Promises, `async`/`await`, timers, and event-loop patterns.
- `try`/`catch`/`finally`, custom errors, and validation.
- Common standard APIs: `fs`, `path`, `URL`, `fetch`, `JSON`, `Math`, `Date`, `Intl`, `crypto`.
- Algorithms and data structures implemented in JavaScript.
- Tests using Jest/Mocha/Node `assert`, and bug-fix examples.

### 6.4 JavaScript source mix for Scenario B

The 225M target cannot come from one source. A defensible mix is:

| Source family | Planning range | Notes |
|---|---:|---|
| Recovered current CodeSearchNet JavaScript | 15–31M | After classification and context; parse-valid only |
| The Stack v1/v2 JavaScript subset | 140–175M | Gated; per-file license allowlist; quality filters |
| CodeParrot JavaScript subset | 20–40M | Public but mixed licenses; allowlist and dedup |
| MDN code examples | 1–3M | Code only; date/provenance verification |
| Node.js docs/tests | 3–8M | MIT core; per-file review; tests need context |
| CommitPackFT JavaScript | 1–3M | Bug-fix/editing examples, not standalone functions |
| BugsJS | <1M | Evaluation-first; do not train until benchmark separation is proven |

These ranges overlap deliberately. They are **source-availability estimates**, not validated yields. Final counts come from the Phase 18 census. If the licensed total is below **180M JavaScript tokens**, Scenario B should not proceed; use Scenario A or renegotiate source access.

### 6.5 JavaScript quality rules

- Parse-valid is mandatory.
- A record must be intelligible with its attached context; if it needs a package not present, either attach the import/package metadata or mark it `dependency_heavy` and exclude it from standalone-function quotas.
- Do not convert every method into a fake standalone function. Use `<|task:class|>` or `<|task:module|>`.
- Do not use minified, generated, vendored, or bundled files unless they are explicitly reconstructed and verified.
- Keep browser and Node contexts distinct in metadata; do not mix `window` and `process` examples without labels.
- Cap per-repository and per-file exposure to prevent a few large projects from dominating.

---

## 7. Data quality and leakage controls

### 7.1 Gate sequence

Every candidate record passes through these gates in order. A failure at a mandatory gate quarantines the record; it does not silently disappear.

| Gate | Check | Mandatory? | Failure action |
|---|---|---|---|
| G1 Source | Approved source, version, retrieval date, license evidence | Yes | Reject until approved |
| G2 Provenance | Repo, commit, path, URL, source ID, raw hash | Yes | Quarantine |
| G3 Decode | Valid UTF-8; JSON/schema decode; no empty fields | Yes | Quarantine |
| G4 Syntax | Python `ast.parse`; JavaScript static parser | Yes | Quarantine |
| G5 Shape | Minimum code length; no placeholder/truncated AST; no generated/minified markers | Yes | Quarantine |
| G6 License | SPDX allowlist at file and repo level; attribution present | Yes | Reject |
| G7 Identity | Declared name/signature matches AST; docstring/signature consistency | Yes for docstring tasks | Quarantine or relabel |
| G8 Dedup | Exact, normalized, near-duplicate | Yes | Drop duplicate |
| G9 Split | Repository-level partition assignment; no cross-partition repo | Yes | Reassign before freeze |
| G10 Contamination | Benchmark and partition overlap checks | Yes | Drop or quarantine |
| G11 Schema | v2 round-trip; section order; special token validity | Yes | Reject |
| G12 Final | Counts, hashes, manifest completeness | Yes | Block corpus freeze |

### 7.2 Static syntax validation

- **Python:** `ast.parse` with a fixed Python version; reject syntax errors; record AST node counts and top-level structure.
- **JavaScript:** use a pinned static parser. The current project virtual environment has no JavaScript parser dependency, so Phase 18 must add one explicitly (for example, a pinned `tree-sitter`/`tree-sitter-javascript` or an equivalent parser) and record the version. Do not rely on Node execution for syntax validation.
- **No execution:** validation must not import, evaluate, or run corpus code. Execution is reserved for the evaluation sandbox.
- **AST-aware chunking:** split long functions/classes/modules at statement or declaration boundaries; never truncate in the middle of a string, template literal, or block.

### 7.3 Duplicate detection

Run deduplication **globally before splitting**, then verify after splitting.

1. **Exact duplicate:** SHA-256 of normalized text; drop duplicates, keeping the best provenance.
2. **Normalized duplicate:** strip comments and insignificant whitespace; for Python, compare canonical AST dumps; for JavaScript, compare parser-normalized token/AST forms. Do not collapse semantically different code.
3. **Near-duplicate:** MinHash/LSH over 5-token shingles; Jaccard threshold calibrated on a labeled sample; drop near-duplicates above threshold.
4. **Cross-source duplicates:** the same code may appear in CodeSearchNet, The Stack, CodeParrot, and CommitPackFT. Dedup across sources, not only within a source.
5. **Exposure caps:** cap records per repository, per file, and per normalized template. A large project cannot supply an unlimited share of a category.

### 7.4 Identifier and signature consistency

- Parse the declared function/class name and compare it to manifest metadata.
- For docstring-to-code records, verify that the signature parameters appear in the code and that the docstring's referenced names are not contradictory.
- For tests, verify that referenced imports and function names exist in the attached code/context; do not execute the test.
- Quarantine mismatches rather than auto-rewriting code. Auto-repair can fabricate behavior.

### 7.5 License filtering

- Use an explicit SPDX allowlist: MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, ISC, 0BSD, Unlicense, CC0 for code where applicable.
- Exclude copyleft, non-commercial, no-derivatives, unknown, and custom licenses unless the project owner obtains explicit legal approval.
- Apply **both** per-file and per-repository checks; a permissively licensed file in a copyleft repository needs review.
- Preserve attribution and license evidence in the manifest.
- Treat MDN prose, Stack Overflow, and CC-BY-SA sources separately; do not mix them into the permissive code pool without review.
- This report is a technical audit, not legal advice. Final approval belongs to the project owner.

### 7.6 Repository-level splitting

- Canonicalize repository identity: normalize URL, strip `.git`, lowercase host, detect forks/mirrors where feasible.
- Assign partitions by a hash of the canonical repo ID, stratified by language, source, size, and category.
- **No repository may appear in more than one partition.** All files, classes, tests, and multi-file examples from a repository go to the same partition.
- For monorepos, use the repository root as the split unit unless a documented subproject boundary is independently licensed and independently identified.
- Deduplicate globally first; then split; then verify no exact/near duplicate crosses partitions.

### 7.7 Train/validation leakage

- Check exact and near-duplicate code across train/validation/test.
- Check same repository across partitions (must be zero).
- Check docstring/code pairs where the docstring or code appears in another partition.
- Check test files and expected outputs for overlap with training.
- Check the new benchmark separately; benchmark tasks are never part of the corpus.
- The old test partition is not read, concatenated, or used for model selection; it remains a historical artifact.

### 7.8 Provenance record

Every retained record should carry:

```text
source_id, dataset_name, dataset_version, retrieval_date,
repo_url, repo_commit, file_path, file_url,
license_spdx, license_evidence_url, attribution,
raw_sha256, normalized_sha256, near_dup_cluster_id,
transform_version, parent_record_ids, split, category, tags
```

If a field is unavailable, the record is not eligible for the corpus. "Unknown" is not a license.

### 7.9 Benchmark contamination controls

- Maintain a frozen benchmark manifest before corpus construction.
- Check exact/near code overlap, AST similarity, identifier overlap, test overlap, and repository overlap.
- Exclude any candidate record whose repository or task appears in the benchmark.
- Re-run contamination checks after every corpus build; a passing check on an earlier build does not cover a later one.
- Do not tune on the benchmark; use the validation split for model selection.

---

## 8. Three corpus-scale proposals

### 8.1 Shared assumptions

- Tokens are **unique training tokens**, not epoch exposures.
- Code records average roughly 300 tokens for planning; documentation is counted separately.
- Repository counts are planning estimates; they are not source-reported certainties.
- All scenarios use the same category shares from §5 and the same 5% documentation target.
- All scenarios require the quality gates in §7. A scenario that cannot pass the gates is not a smaller version of success; it is a failed build.

### 8.2 Scenario comparison

| Dimension | A — Conservative | B — Intermediate | C — Ambitious |
|---|---:|---:|---:|
| Total unique training tokens | **200M** | **500M** | **1B** |
| Python code tokens | 110M | 250M | 480M |
| JavaScript code tokens | 80M | 225M | 470M |
| Documentation tokens | 10M (5%) | 25M (5%) | 50M (5%) |
| Approx. code records | ~650k | ~1.5M | ~3.1M |
| Approx. unique repositories | ~35k | ~75k | ~130k |
| Relative to current 133.2M | 1.5× | 3.75× | 7.5× |
| JavaScript relative to current 6.54M | 12.2× | 34.4× | 71.9× |
| Python relative to current 98.8M | 1.11× | 2.53× | 4.86× |
| Storage (additional, planning) | 10–20 GB | 20–35 GB | 40–60 GB |
| Curation complexity | Low–medium | Medium–high | High |
| Curation estimate | 2–4 engineer-weeks | 5–9 engineer-weeks | 10–16 engineer-weeks |
| Main constraint | JavaScript supply | Licensed JavaScript + quality | Legal access + storage + compute |
| Expected limitation | Still narrow; modest JS gain | Best feasible balance | High risk; not justified on 8 GB |

### 8.3 Category distribution for each scenario

The category shares are identical across A/B/C by design. The absolute tokens differ. Scenario B is shown in §5.3; A and C scale those values by 0.44 and 2.0 respectively (using code-token totals of 190M and 950M). This is intentional: changing category shares between scenarios would confound data-scale comparisons.

Absolute planning envelope for primary categories, Python + JavaScript combined (millions of code tokens):

| Category | A (190M) | B (475M) | C (950M) |
|---|---:|---:|---:|
| Basic | 15.2 | 38.0 | 76.0 |
| Functions/recursion | 22.0 | 54.8 | 109.3 |
| Algorithms | 21.2 | 52.5 | 104.6 |
| Collections | 30.4 | 76.0 | 152.0 |
| Data structures | 15.2 | 38.0 | 76.0 |
| OOP | 18.4 | 47.0 | 94.8 |
| Modules/imports | 14.1 | 35.5 | 71.2 |
| Exceptions | 11.4 | 28.5 | 57.0 |
| File/data | 8.7 | 21.5 | 42.8 |
| Standard library | 11.7 | 28.8 | 57.1 |
| Async | 7.0 | 18.5 | 37.8 |
| Testing/debugging | 9.8 | 24.0 | 47.6 |
| Larger examples | 4.9 | 12.0 | 23.8 |
| **Total** | **190.0** | **475.0** | **950.0** |

These values are derived from the language-specific shares in §5.2 and are planning envelopes, not validated yields. Documentation is not part of the 13 code categories. It is a separate 5% exposure target: 10M / 25M / 50M tokens.

### 8.4 Records, repositories, and source mix

| Scenario | Python source mix | JavaScript source mix | Documentation mix |
|---|---|---|---|
| A | Current CSN Python + CPython stdlib/tests subset + limited Stack/CodeParrot Python | Recovered CSN JS + CodeParrot/Stack JS subset | CPython docs + selected MDN/Node code examples |
| B | Current CSN Python + Stack/CodeParrot Python + CPython stdlib/tests + APPS train (after holdout) | Recovered CSN JS + Stack/CodeParrot JS + MDN + Node + CommitPackFT JS | CPython docs + MDN code + Node docs, code-adjacent |
| C | All of B plus larger Stack/StarCoderData subsets | All of B plus larger Stack/StarCoderData JavaScript | All of B plus broader docs, still code-adjacent |

"Stack/CodeParrot" here means **licensed and validated subsets**, not bulk downloads. Scenario C's source mix is not currently approved.

### 8.5 Storage estimates

Planning assumptions:

- Current raw JSONL is **15.5 bytes/token** (2.07 GB for 133.2M tokens).
- Tokenized `.bin` is **2 bytes/token** for uint16.
- Manifests, dedup indexes, license evidence, and temporary files add 1–3 GB for A, 3–5 GB for B, and 6–10 GB for C.
- Checkpoints add 2–4 GB, 4–8 GB, and 8–15 GB respectively for the Phase 16 model sizes, assuming multiple checkpoints.

| Scenario | Raw JSONL | Tokenized `.bin` | Indexes/manifests | Checkpoints | Additional working storage | Existing project footprint | Plan for total |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | ~3.1 GB | ~0.4 GB | 1–2 GB | 2–4 GB | ~10–20 GB | 16.6 GB measured | 25–40 GB |
| B | ~7.8 GB | ~1.0 GB | 3–5 GB | 4–8 GB | ~20–35 GB | 16.6 GB measured | 40–55 GB |
| C | ~15.5 GB | ~2.0 GB | 6–10 GB | 8–15 GB | ~40–60 GB | 16.6 GB measured | 60–80 GB |

Keep at least **2× the planned free space** during dedup and corpus builds; dedup indexes and temporary sorts can spike. The current project footprint is dominated by `data/` (8.7 GB), `experiments/` (2.5 GB), `checkpoints/` (5.1 GB), and `release/` (0.38 GB).

### 8.6 Curation cost, complexity, and limitations

**A — Conservative (200M):**

- Reuses the current adapters and most of the current pipeline.
- Adds a modest JavaScript expansion and a first construct census.
- Needs static JavaScript parsing and near-dedup, but not at 1B scale.
- Expected limitation: still function-heavy; JavaScript remains much smaller than Python; 512-token context may still fragment modules.
- Best use: fallback if external licensing stalls.

**B — Intermediate (500M):**

- Requires multiple new source adapters, a real JavaScript classifier, repo-level dedup at scale, and a v2 schema/tokenizer.
- Needs legal review for Stack/CodeParrot/MDN/Node and a benchmark freeze before training.
- Expected limitation: 8 GB VRAM keeps the model around 65M parameters; 500M unique tokens is still a modest token/parameter ratio; quality depends on JavaScript source quality.
- Best use: the recommended target.

**C — Ambitious (1B):**

- Requires multi-source legal agreements, multi-TB staging, expensive global dedup, and 30–80 hours per training pass on the current GPU.
- JavaScript would need to roughly double again beyond B.
- Expected limitation: high risk of license gaps, low-quality bulk code, and thermal/time constraints; no evidence that the current hardware can iterate effectively at this scale.
- Best use: defer until hardware and legal review improve.

### 8.7 Source availability versus usable data

| Scenario | Source-available? | Usable after current approvals? | Verdict |
|---|---|---|---|
| A | Yes: current corpus + public CodeParrot subset + docs | Partially; CodeParrot still needs license and quality filtering | Feasible with a focused build |
| B | Yes in principle: Stack/CodeParrot/MDN/Node/CommitPackFT | No; gated sources and legal review are not complete | Conditional; recommended only if approvals and JS yield succeed |
| C | Yes in principle: Stack/StarCoderData | No; bulk agreements, storage, and compute are not in place | Not recommended now |

The distinction is deliberate. "The Stack has 6 TB" does not mean 6 TB is licensed, clean, deduplicated, and usable for this project.

---

## 9. Recommended target

**Recommend Scenario B: 500M unique training tokens — 250M Python, 225M JavaScript, 25M documentation; ~1.5M code records; ~75,000 repositories.**

### Why B

1. **It addresses the binding constraint.** JavaScript rises from 6.54M to 225M unique tokens, a 34× increase, instead of repeating the same small set.
2. **It supports structural coverage.** 500M tokens provide room for classes, modules, async, tests, and larger examples without forcing every record into a function-level quota.
3. **It fits the measured hardware envelope.** Phase 16's intermediate model (≈65M parameters) is estimated at 4–6 GB VRAM and 9–14k tokens/s; a 300–600M-token pass is 6–18 hours. A 500M pass is roughly 10–15 hours at 12k tokens/s, leaving room for iteration on an 8 GB GPU.
4. **It is the conditional Phase 16 recommendation, not an automatic one.** B is approved only after the JavaScript source yield, license review, and quality gates succeed.
5. **It avoids the C trap.** A 1B corpus on 8 GB VRAM would require 30–80 hours per pass and much more legal/storage work, with no evidence that the extra tokens would be clean enough to help.

### Conditions that must be met before B is built

- At least **180M validated unique JavaScript tokens** after license and quality filtering; target 225M.
- At least **220M validated unique Python tokens** after dedup and quality filtering; target 250M.
- Documentation at **5%**, code-adjacent, with no raw-prose dump.
- All mandatory gates in §7 pass; no unknown licenses; no repository overlap; no benchmark contamination.
- A new benchmark is frozen before training; `independent_v3` remains historical only.
- The v2 tokenizer and schema pass round-trip and special-token tests.
- v0.1.0 remains immutable; all models train from scratch.

### Fallback and deferral

- **A is the fallback** if JavaScript licensing/yield fails. A still improves quality and diversity, but it will not close the JavaScript gap.
- **C is deferred** until storage, legal review, and compute are materially expanded. Do not start C merely because more data is available.
- **Do not fine-tune Exp C.** Fine-tuning would inherit the old distribution and violate the from-scratch requirement.

### Separation from architecture and training

This section recommends a **data target**, not an architecture change. The Phase 16 model sizes and throughput figures are planning envelopes for feasibility, not commitments. Architecture, context length, and training schedule remain Phase 18 decisions after the corpus is measured.

---

## 10. Storage and hardware implications

### 10.1 Disk

- Current project footprint is about **16.6 GB** (`data/` 8.7 GB, `experiments/` 2.5 GB, `checkpoints/` 5.1 GB, `release/` 0.38 GB).
- Scenario B adds roughly **20–35 GB** of working data, indexes, and checkpoints; plan for **40–55 GB total** and keep **2× free space** during dedup.
- Raw JSONL is the largest single artifact; tokenized `.bin` is much smaller. Do not delete raw data until the manifest and hashes are frozen.
- Keep the v0.1.0 release and old checkpoints in a separate immutable archive; do not let a new corpus build overwrite them.

### 10.2 CPU and RAM

- Data curation is CPU/disk-bound, not GPU-bound.
- Python AST parsing and JavaScript parsing are parallelizable across processes; budget CPU time for 1–3M candidate records, not just the retained set.
- MinHash/LSH dedup at 500M tokens needs a memory-conscious implementation (sharded indexes, disk-backed buckets). Do not assume a single in-memory index fits.
- Tokenizer training on a 500M-token corpus is feasible on the existing machine, but it needs a representative sample and a memory budget; do not fit on the full raw JSONL indiscriminately.

### 10.3 GPU

Measured baseline from Phase 16: **17M parameters, 1,150.93 MiB, 44,268 tokens/s, 3,008.8 seconds per 133M-token pass** on the RTX 5060 Laptop with 8 GB VRAM.

| Scenario | Phase 16 model | Est. VRAM | Est. throughput | Est. time per pass | 8 GB verdict |
|---|---:|---:|---:|---:|---|
| A | ≈34M | 2–3 GB | 20–30k tok/s | 1.5–3.5 h per 150–250M | Comfortable |
| B | ≈65M | 4–6 GB | 9–14k tok/s | 6–18 h per 300–600M | Feasible with small batches/gradient accumulation |
| C | ≈110M | 5–7 GB | 5–8k tok/s | 30–80 h per 800M–1.5B | Tight; long iteration; not recommended |

At 500M tokens and 12k tokens/s, one pass is about **11.6 hours**; at 9k tokens/s it is about **15.4 hours**. This is compatible with overnight runs, but not with rapid experimentation. Use checkpointing, resumable data loaders, and a fixed evaluation cadence.

### 10.4 Operational implications

- Build the corpus in stages; do not wait for the full 500M before validating the pipeline.
- Freeze a **1–5M-token pilot** first, run all gates, and inspect category coverage before scaling.
- Keep raw, filtered, deduplicated, tokenized, and split artifacts separate with manifests.
- Record every transform version and hash; a corpus that cannot be reproduced is not a baseline.
- The old test partition and `functional/independent_v3/` are not part of this pipeline and must not be read during corpus construction.

---

## 11. Fresh evaluation strategy

### 11.1 New partitions

- Build new **train/validation/test** partitions from the v2 corpus, independent of the old partitions.
- Use **repository-level splitting**: no repository crosses partitions; all files/classes/tests from a repo stay together.
- Target split: **97% train / 1.5% validation / 1.5% test**, stratified by language, source, repository size, and category. Ensure validation/test each contain at least **2M tokens** and all 13 categories in both languages where possible.
- The test partition is opened only for final evaluation, never for model selection or hyperparameter tuning.
- The old test partition remains untouched. If v0.1.0 train/valid records are reused as seed data, re-filter and re-split them; do not read the old test partition to construct the new split.
- The new benchmark is separate from the new test partition. A model can be selected on validation, but the benchmark is a held-out final exam.

### 11.2 New benchmark design (design only — do not create yet)

Proposed name: **PolyCodeBench v2** (working name).

- **Size:** ~300 tasks, **150 Python + 150 JavaScript**.
- **Difficulty:** 50 basic, 60 intermediate, 40 advanced per language.
- **Shape:** ~60% function-level, ~30% class/module-level, ~10% multi-file or repository-context.
- **Category coverage:** at least 10 tasks per category across the two languages; async and testing/debugging must be represented in both.
- **Task format:** natural-language specification, signature/interface, 1–2 public examples, **5–12 hidden tests**, no reference solution in the public manifest.
- **Execution:** sandboxed, no network, timeouts, memory limits, deterministic seeds where possible. Execution is an evaluation-time activity, not corpus validation.
- **Metrics:** syntax validity, pass@1, pass@k, case pass rate, runtime-error rate, timeout rate, EOS correctness, repetition rate, and per-language/per-category breakdowns.

### 11.3 Provenance and hidden tests

Every task records:

```text
task_id, language, category, difficulty, shape,
source_type (authored/adapted), source_repo, source_commit, source_path,
license_spdx, author/reviewer, creation_date,
spec_sha256, public_examples_sha256, hidden_tests_sha256,
provenance_notes
```

- Adapted tasks must come from permissively licensed sources and must not be in the training corpus.
- Authored tasks must be reviewed by a second person.
- Hidden tests are stored separately from public task descriptions.
- The manifest and all task/test hashes are frozen before training starts.

### 11.4 Contamination checks

Run all of these before freezing the benchmark and again after the corpus build:

- Exact code/test hash overlap with train/validation/test.
- Normalized and near-duplicate overlap.
- AST similarity for Python and JavaScript.
- Identifier/API overlap that indicates a copied solution.
- Repository overlap: no benchmark repo in training.
- Task-spec overlap: benchmark prompts/descriptions not present in documentation or code comments.
- Source-level overlap with APPS, CodeContests, MBPP, and `independent_v3`; exclude any task with a known provenance conflict.

A benchmark task that fails contamination is replaced before freezing, not after training.

### 11.5 Freeze and use

- Freeze the benchmark manifest, public examples, hidden tests, and hashes before training.
- No benchmark-driven data selection, prompt tuning, or checkpoint selection.
- Use validation for model selection; use the benchmark once for the final report.
- Keep the old `functional/independent_v3/` benchmark as a historical selection benchmark only. Do not inspect individual task examples or cases, do not train on it, and do not tune against it.

### 11.6 Why a new benchmark is necessary

The current benchmark was used for model selection and is tied to the old data distribution. Reusing it would reward the old function-level pattern and make the new corpus look better or worse for the wrong reasons. A fresh, provenance-tracked, hidden-test benchmark is the only credible way to measure whether the new data improves Python and JavaScript programming competence.

---

## 12. Exact implementation roadmap

Phase 17 is design-only. The following is the Phase 18 dataset-engineering sequence. Each phase has a deliverable and an exit gate; do not start the next phase until the gate passes.

| Phase | Work | Deliverable | Exit gate |
|---|---|---|---|
| **17.0 Freeze** | Archive v0.1.0, current tokenizer, manifests, old checkpoints; record hashes; confirm read-only boundary | Freeze record; checksums | v0.1.0 artifacts immutable; old test untouched |
| **17.1 Source approvals** | Complete legal/license matrix for CSN, Stack, CodeParrot, CommitPackFT, MDN, CPython, Node, APPS; add to registry; approve/reject each | Approved source list with evidence | Only approved sources proceed; no bulk download before approval |
| **17.2 Adapters** | Build per-source adapters that preserve repo, commit, path, license, URL, and raw hash; immutable raw landing zone | Adapter code + provenance schema + pilot outputs | 100% provenance on pilot; raw data hashed |
| **17.3 Static quality** | Python AST parser; pinned JavaScript parser; syntax/shape/identifier checks; quarantine store | Quality report + quarantine manifest | Parse rate and failure reasons measured; no execution of corpus code |
| **17.4 Dedup** | Exact, normalized, near-duplicate, cross-source dedup; exposure caps; sharded indexes | Dedup index + duplicate report | Duplicate rates below thresholds; no cross-source duplicates left |
| **17.5 Construct census** | Classify functions/classes/modules/async/tests/etc.; category and construct counts by language/source/repo | Census report + category coverage chart | Every category measured; underfills documented, not padded |
| **17.6 Schema v2** | Implement v2 envelope, special tokens, fixed section order, round-trip serializer | v2 spec + serializer + tests | Round-trip passes; malformed sections rejected; v1 unchanged |
| **17.7 Tokenizer v2** | Train a new tokenizer on the v2 mixture; verify special tokens and code/doc boundaries | Tokenizer v2 + vocab/hash manifest | Special-token IDs stable; no v1 mutation; sample encode/decode clean |
| **17.8 Pilot build** | Build a 1–5M-token pilot across categories and sources; run all gates | Pilot corpus + manifests + report | All gates pass; category coverage acceptable; no leakage |
| **17.9 Scenario B build** | Scale to 500M unique tokens (250M Python / 225M JS / 25M docs); freeze manifests and hashes | B corpus + split + manifests | ≥180M JS and ≥220M Python validated; 5% docs; no repo overlap; no benchmark contamination |
| **17.10 Benchmark v2** | Design ~300 tasks, hidden tests, provenance; run contamination checks; freeze | Frozen PolyCodeBench v2 | 150 Python + 150 JS; all categories; hashes frozen; no training overlap |
| **17.11 Final audit** | Re-run all gates on the frozen corpus; produce reproducibility report; hand off to Phase 18 training | Audit report + handoff package | Every gate green; limitations documented; training can start from scratch |

### Roadmap guardrails

- No dataset download, curation, or training during Phase 17.
- No modification of `src/`, tokenizer, serialization, checkpoints, release artifacts, or the old test partition.
- No use of `functional/independent_v3/` for training, tuning, or individual example inspection.
- No model training until the corpus and benchmark are frozen.
- If any gate fails, stop and fix the cause; do not lower the gate to preserve a schedule.

---

## 13. Risks and open questions

| Risk | Impact | Mitigation |
|---|---|---|
| Stack/CodeParrot license review is incomplete | B cannot be built | Approve per-file allowlist; fall back to A |
| JavaScript validated yield is below 180M | B is not balanced | Add sources or reduce to A; do not oversample |
| JavaScript parser dependency is not pinned | Syntax gate is irreproducible | Pin parser/version in Phase 18; record in manifest |
| Global dedup at 500M is expensive | Build time and memory blow up | Shard indexes; pilot first; measure |
| v2 special tokens fragment or are misused | Model learns noisy boundaries | Dedicated IDs; round-trip tests; fixed order |
| 8 GB VRAM limits batch size | Training is slow or unstable | Gradient accumulation; small batch; checkpointing |
| Benchmark contamination | Inflated results | Frozen manifest; multi-method contamination checks; no tuning |
| Source drift or version changes | Corpus is not reproducible | Pin versions, retrieval dates, hashes; immutable raw zone |
| No execution during validation | Runtime errors survive into training | Static checks plus evaluation sandbox; do not execute corpus code |
| Old test/benchmark leakage | Invalid historical comparison | Old test untouched; new benchmark fresh; strict separation |

Open questions for Phase 18:

1. Which JavaScript parser will be pinned, and what AST normalization does it support?
2. Which Stack/CodeParrot subsets can be licensed and staged within the available disk budget?
3. Can MDN code examples be dated and attributed reliably enough for CC0/MIT separation?
4. Is the 5% documentation target achievable with code-adjacent extraction rather than raw prose?
5. What is the measured JavaScript yield after context attachment and dedup?
6. Does the v2 tokenizer preserve enough indentation/structure for Python and JavaScript?
7. What is the actual VRAM headroom for the 65M model at 1024 context on 8 GB?

---

## 14. Decision

The audit supports a data-first redesign. The current corpus is a valid historical baseline but cannot support the next generation's Python/JavaScript competence on its own. The recommended path is **Scenario B, conditional on licensed JavaScript expansion and all quality gates**, with **Scenario A as fallback** and **Scenario C deferred**. All future models must be trained from scratch; Exp C must not be fine-tuned; v0.1.0 must remain immutable.

PHASE 17 DESIGN COMPLETE — READY FOR DATASET ENGINEERING
