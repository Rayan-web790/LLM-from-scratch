# Phase 10F — Final Verification and Reconciliation

Read-only audit. No retraining, no checkpoint change, no corpus/tokenizer
modification, no access to the old test split, no generated code executed, no
100-task benchmark run. All pre-existing reports are untouched; this file and
`reports/phase10f_label_examples.json` are new.

- Model: `checkpoints/polycodegpt_first_pass/step_008130.pt` (step 8130, verified by the loader)
- Tokenizer: `tokenizer/artifacts/polycodegpt_tokenizer_v1_1.json`
- Diagnostic source data: `reports/phase10f_decoding_diagnostics.json`

---

## A. Decoding configuration verification

### A1 — Exact arguments passed

`scripts/phase10f_decoding.py` calls, for every task:

```python
full = model.generate(
    inp,                      # torch.tensor([prompt_ids], dtype=torch.long, device="cuda")
    max_new_tokens=64,
    temperature=temperature,
    top_k=top_k,              # Python None for "disabled", 50 for "top-50"
    eos_token_id=2,           # <|eos|>
)
```

The two settings from the settings table therefore differ only in one keyword:

| setting | `temperature` | `top_k` passed |
|---|---:|---|
| `sampling_t0.4_topkNone` | 0.4 | `None` |
| `sampling_t0.4_topk50` | 0.4 | `50` |
| `sampling_t0.8_topkNone` | 0.8 | `None` |
| `sampling_t0.8_topk50` | 0.8 | `50` |
| `greedy_temp0` | 0.0 | `None` |

### A2 — `None` genuinely disables top-k

`GPT._generate_loop` (`src/polycodegpt/model/gpt.py`):

```python
if temperature == 0:
    next_token = torch.argmax(logits, dim=-1, keepdim=True)
else:
    if temperature != 1.0:
        logits = logits / temperature
    if top_k is not None and top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
```

`top_k=None` fails the `is not None` guard, so no `topk`/`masked_fill` runs and
the full 16,000-entry softmax is sampled: full-support sampling. `top_k=50`
masks everything below the 50th logit to `-inf`, then renormalizes. Neither path
throws, and there is no default top-k applied inside `generate()`
(`top_k: Optional[int] = None`). At `temperature == 0` the top-k argument is not
read at all, because the branch goes straight to `argmax`. `top_p` and
`repetition_penalty` do not exist as parameters, so they were reported as
unsupported rather than approximated.

### A3 — Why top-k disabled and top-k=50 matched at T=0.4 but not at T=0.8

The script reseeds the global torch RNG identically before each seed's run
(`torch.manual_seed(seed)`), so both configurations start from the same RNG
state and consume draws in the same order. Any difference in output must come
from the distribution, not from the seed. Measured on the first decoding step of
the first three prompts (single forward pass, `softmax(logits / T)`):

| prompt | T | top-50 probability mass | max single token | mass outside top-50 |
|---|---:|---:|---:|---:|
| task1 | 0.4 | 1.000000 | 0.4325 | 2.4e-07 |
| task1 | 0.8 | 0.998023 | 0.3131 | 2.0e-03 |
| task2 | 0.4 | 1.000000 | 0.4354 | 4.2e-07 |
| task2 | 0.8 | 0.997674 | 0.3121 | 2.3e-03 |
| task3 | 0.4 | 1.000000 | 0.5300 | 2.4e-07 |
| task3 | 0.8 | 0.998292 | 0.3571 | 1.7e-03 |

At T=0.4 the distribution is sharp enough that top-50 truncation removes about
3e-7 of the mass, i.e. below float32 resolution after renormalization; every
sampled index is unchanged, so all 30 generations are byte-identical to the
full-support run. At T=0.8 roughly 2e-3 of the mass sits outside the top 50, so
truncation removes real alternatives and one changed token cascades through the
rest of the sequence. Divergence at T=0.8 (from the stored report, comparing the
same seed):

| seed | identical generations (T=0.8, none vs 50) |
|---|---:|
| 42 | 5/10 |
| 123 | 5/10 |
| 999 | 7/10 |

So the earlier statement "top-k disabled and top-k=50 are identical across all
three seeds" is only true for T=0.4; it is false for T=0.8. The bottleneck is
temperature, not top-k.

### A4 — Candidates are generated independently and stored separately

Each of the 130 stored generations came from its own `model.generate` call on a
freshly built prompt tensor (no cache, no reuse, no shared output buffer). In the
stored JSON, every run contains 10 distinct `candidate` strings
(10/10 unique in all 13 runs), and each entry carries its own `generated_text`,
`candidate`, token count, ratios, EOS flag and syntax result. Greedy runs are
deterministic by construction; sampling runs differ across seeds.

### A5 — Seeds reach the real PyTorch RNG

`torch.manual_seed(seed)` seeds the CPU and CUDA default generators, which is the
generator `torch.multinomial` uses because `generate()` passes no `generator=`.
Direct check on task 2 (`max_new_tokens=32`, T=0.8, top_k=None):

- seed 42 twice → identical token sequences (`True`)
- seed 42 vs seed 43 → different token sequences (`True` for difference)
- greedy twice → identical (`True`)

`random.seed(seed)` in the script is inert: Python's `random` module is not used
anywhere in the generation path. The seed is set once per seed, before the ten
tasks, so the ten generations of one run form a reproducible *sequence* rather
than ten independently seeded draws. That is deterministic and reproducible, but
it means per-task results are not independent samples; the three seeds give three
replicate sequences, not thirty independent ones.

---


### A6 — Exact per-task results for every configuration and seed

`eos` = `<|eos|>` emitted within 64 tokens; `ok` = `node --check` accepts stub+generation; `toks` = generated tokens kept (EOS excluded); `UTR` = unique-token ratio; `R4` = repeated-4-gram ratio.

#### A6.1 Greedy, temperature=0, top_k=None

`max_new_tokens=64`, `top_k=None`, `temperature=0.0`

seed n/a (greedy) — EOS 0/10, syntax-valid 0/10

| task | eos | ok | toks | UTR | R4 |
|---|:--:|:--:|---:|---:|---:|
| 1 | N | N | 64 | 0.078 | 1.000 |
| 2 | N | N | 64 | 0.125 | 1.000 |
| 3 | N | N | 64 | 0.172 | 0.952 |
| 4 | N | N | 64 | 0.109 | 1.000 |
| 5 | N | N | 64 | 0.172 | 0.963 |
| 6 | N | N | 64 | 0.125 | 1.000 |
| 7 | N | N | 64 | 0.219 | 1.000 |
| 8 | N | N | 64 | 0.188 | 1.000 |
| 9 | N | N | 64 | 0.172 | 0.952 |
| 10 | N | N | 64 | 0.188 | 0.000 |

#### A6.2 Sampling T=0.4, top-k disabled

`max_new_tokens=64`, `top_k=None`, `temperature=0.4`

seed 42 — EOS 2/10, syntax-valid 2/10

| task | eos | ok | toks | UTR | R4 |
|---|:--:|:--:|---:|---:|---:|
| 1 | N | N | 64 | 0.125 | 1.000 |
| 2 | N | N | 64 | 0.125 | 1.000 |
| 3 | Y | Y | 58 | 0.431 | 0.000 |
| 4 | N | N | 64 | 0.125 | 0.793 |
| 5 | Y | Y | 55 | 0.491 | 0.421 |
| 6 | N | N | 64 | 0.125 | 1.000 |
| 7 | N | N | 64 | 0.266 | 0.938 |
| 8 | N | N | 64 | 0.141 | 0.826 |
| 9 | N | N | 64 | 0.141 | 0.931 |
| 10 | N | N | 64 | 0.219 | 1.000 |

seed 123 — EOS 0/10, syntax-valid 0/10

| task | eos | ok | toks | UTR | R4 |
|---|:--:|:--:|---:|---:|---:|
| 1 | N | N | 64 | 0.109 | 0.905 |
| 2 | N | N | 64 | 0.125 | 1.000 |
| 3 | N | N | 64 | 0.312 | 0.000 |
| 4 | N | N | 64 | 0.344 | 0.690 |
| 5 | N | N | 64 | 0.328 | 0.294 |
| 6 | N | N | 64 | 0.094 | 1.000 |
| 7 | N | N | 64 | 0.125 | 1.000 |
| 8 | N | N | 64 | 0.172 | 0.706 |
| 9 | N | N | 64 | 0.250 | 0.667 |
| 10 | N | N | 64 | 0.281 | 0.667 |

seed 999 — EOS 7/10, syntax-valid 7/10

| task | eos | ok | toks | UTR | R4 |
|---|:--:|:--:|---:|---:|---:|
| 1 | N | N | 64 | 0.125 | 0.909 |
| 2 | Y | Y | 24 | 0.667 | 0.333 |
| 3 | Y | Y | 22 | 0.727 | 0.000 |
| 4 | Y | Y | 28 | 0.536 | 0.000 |
| 5 | Y | Y | 43 | 0.395 | 0.684 |
| 6 | Y | Y | 24 | 0.667 | 0.333 |
| 7 | Y | Y | 42 | 0.452 | 0.200 |
| 8 | N | N | 64 | 0.219 | 1.000 |
| 9 | Y | Y | 30 | 0.633 | 0.000 |
| 10 | N | N | 64 | 0.188 | 0.000 |

#### A6.3 Sampling T=0.8, top-k disabled

`max_new_tokens=64`, `top_k=None`, `temperature=0.8`

seed 42 — EOS 3/10, syntax-valid 3/10

| task | eos | ok | toks | UTR | R4 |
|---|:--:|:--:|---:|---:|---:|
| 1 | N | N | 64 | 0.219 | 0.762 |
| 2 | N | N | 64 | 0.344 | 0.217 |
| 3 | Y | Y | 16 | 0.688 | 0.000 |
| 4 | N | N | 64 | 0.344 | 0.421 |
| 5 | N | N | 64 | 0.641 | 0.000 |
| 6 | Y | Y | 35 | 0.571 | 0.000 |
| 7 | N | N | 64 | 0.469 | 0.182 |
| 8 | Y | Y | 50 | 0.400 | 0.000 |
| 9 | N | N | 64 | 0.516 | 0.000 |
| 10 | N | N | 64 | 0.141 | 0.914 |

seed 123 — EOS 2/10, syntax-valid 2/10

| task | eos | ok | toks | UTR | R4 |
|---|:--:|:--:|---:|---:|---:|
| 1 | Y | N | 27 | 0.741 | 0.000 |
| 2 | N | N | 64 | 0.453 | 0.118 |
| 3 | N | N | 64 | 0.578 | 0.300 |
| 4 | N | N | 64 | 0.547 | 0.000 |
| 5 | N | N | 64 | 0.375 | 0.000 |
| 6 | Y | Y | 59 | 0.390 | 0.000 |
| 7 | N | N | 64 | 0.281 | 0.290 |
| 8 | N | N | 64 | 0.406 | 0.483 |
| 9 | N | Y | 64 | 0.359 | 0.000 |
| 10 | N | N | 64 | 0.469 | 0.000 |

seed 999 — EOS 9/10, syntax-valid 8/10

| task | eos | ok | toks | UTR | R4 |
|---|:--:|:--:|---:|---:|---:|
| 1 | Y | Y | 21 | 0.857 | 0.000 |
| 2 | Y | Y | 21 | 0.810 | 0.000 |
| 3 | Y | Y | 19 | 0.842 | 0.000 |
| 4 | Y | Y | 23 | 0.826 | 0.000 |
| 5 | Y | N | 37 | 0.568 | 0.353 |
| 6 | Y | Y | 24 | 0.833 | 0.000 |
| 7 | Y | Y | 22 | 0.727 | 0.000 |
| 8 | Y | Y | 54 | 0.315 | 0.875 |
| 9 | Y | Y | 26 | 0.846 | 0.000 |
| 10 | N | N | 64 | 0.781 | 0.000 |

#### A6.4 Sampling T=0.8, top_k=50

`max_new_tokens=64`, `top_k=50`, `temperature=0.8`

seed 42 — EOS 3/10, syntax-valid 3/10

| task | eos | ok | toks | UTR | R4 |
|---|:--:|:--:|---:|---:|---:|
| 1 | N | N | 64 | 0.219 | 0.762 |
| 2 | N | N | 64 | 0.344 | 0.217 |
| 3 | Y | Y | 16 | 0.812 | 0.000 |
| 4 | N | N | 64 | 0.297 | 0.474 |
| 5 | N | N | 64 | 0.500 | 0.148 |
| 6 | Y | Y | 35 | 0.571 | 0.000 |
| 7 | N | N | 64 | 0.469 | 0.182 |
| 8 | Y | Y | 50 | 0.400 | 0.000 |
| 9 | N | N | 64 | 0.562 | 0.000 |
| 10 | N | N | 64 | 0.422 | 0.000 |

seed 123 — EOS 2/10, syntax-valid 3/10

| task | eos | ok | toks | UTR | R4 |
|---|:--:|:--:|---:|---:|---:|
| 1 | Y | Y | 22 | 0.818 | 0.000 |
| 2 | N | N | 64 | 0.453 | 0.118 |
| 3 | N | N | 64 | 0.391 | 0.000 |
| 4 | N | N | 64 | 0.500 | 0.000 |
| 5 | N | N | 64 | 0.375 | 0.000 |
| 6 | Y | Y | 59 | 0.390 | 0.000 |
| 7 | N | N | 64 | 0.266 | 0.290 |
| 8 | N | N | 64 | 0.406 | 0.483 |
| 9 | N | Y | 64 | 0.359 | 0.000 |
| 10 | N | N | 64 | 0.234 | 0.000 |

seed 999 — EOS 9/10, syntax-valid 8/10

| task | eos | ok | toks | UTR | R4 |
|---|:--:|:--:|---:|---:|---:|
| 1 | Y | Y | 21 | 0.857 | 0.000 |
| 2 | Y | Y | 19 | 0.842 | 0.000 |
| 3 | Y | Y | 19 | 0.842 | 0.000 |
| 4 | Y | Y | 23 | 0.826 | 0.000 |
| 5 | Y | N | 37 | 0.568 | 0.353 |
| 6 | Y | Y | 24 | 0.833 | 0.000 |
| 7 | Y | Y | 22 | 0.727 | 0.000 |
| 8 | Y | Y | 54 | 0.315 | 0.875 |
| 9 | Y | N | 25 | 0.680 | 0.000 |
| 10 | N | Y | 64 | 0.484 | 0.786 |

#### A6.5 Sampling T=0.4, top_k=50

`max_new_tokens=64`, `top_k=50`, `temperature=0.4`

seed 42 — EOS 2/10, syntax-valid 2/10

| task | eos | ok | toks | UTR | R4 |
|---|:--:|:--:|---:|---:|---:|
| 1 | N | N | 64 | 0.125 | 1.000 |
| 2 | N | N | 64 | 0.125 | 1.000 |
| 3 | Y | Y | 58 | 0.431 | 0.000 |
| 4 | N | N | 64 | 0.125 | 0.793 |
| 5 | Y | Y | 55 | 0.491 | 0.421 |
| 6 | N | N | 64 | 0.125 | 1.000 |
| 7 | N | N | 64 | 0.266 | 0.938 |
| 8 | N | N | 64 | 0.141 | 0.826 |
| 9 | N | N | 64 | 0.141 | 0.931 |
| 10 | N | N | 64 | 0.219 | 1.000 |

seed 123 — EOS 0/10, syntax-valid 0/10

| task | eos | ok | toks | UTR | R4 |
|---|:--:|:--:|---:|---:|---:|
| 1 | N | N | 64 | 0.109 | 0.905 |
| 2 | N | N | 64 | 0.125 | 1.000 |
| 3 | N | N | 64 | 0.312 | 0.000 |
| 4 | N | N | 64 | 0.344 | 0.690 |
| 5 | N | N | 64 | 0.328 | 0.294 |
| 6 | N | N | 64 | 0.094 | 1.000 |
| 7 | N | N | 64 | 0.125 | 1.000 |
| 8 | N | N | 64 | 0.172 | 0.706 |
| 9 | N | N | 64 | 0.250 | 0.667 |
| 10 | N | N | 64 | 0.281 | 0.667 |

seed 999 — EOS 7/10, syntax-valid 7/10

| task | eos | ok | toks | UTR | R4 |
|---|:--:|:--:|---:|---:|---:|
| 1 | N | N | 64 | 0.125 | 0.909 |
| 2 | Y | Y | 24 | 0.667 | 0.333 |
| 3 | Y | Y | 22 | 0.727 | 0.000 |
| 4 | Y | Y | 28 | 0.536 | 0.000 |
| 5 | Y | Y | 43 | 0.395 | 0.684 |
| 6 | Y | Y | 24 | 0.667 | 0.333 |
| 7 | Y | Y | 42 | 0.452 | 0.200 |
| 8 | N | N | 64 | 0.219 | 1.000 |
| 9 | Y | Y | 30 | 0.633 | 0.000 |
| 10 | N | N | 64 | 0.188 | 0.000 |

---

## B. Dataset labels — what they actually are

These labels are **exploratory heuristic classifications produced by fixed rules
in code**. They are not manual semantic annotation, and no human reviewed the
sample. The rule set lives in `scripts/phase10f_labels.py` (shared by the Phase
10E audit and this verification; the audit now imports it, and re-running the
audit reproduced the published counts exactly).

Sample: `random.seed(42)`, 200 records per language drawn from `train.jsonl` in
file order, both samples drawn from the same RNG stream (JavaScript sampled
second), so the sample is reproducible.

| category | exact rule |
|---|---|
| `concise_standalone` | `len(code.splitlines()) <= 12` and stripped code starts with `def ` (Python) or `function ` (JS) |
| `algorithmic` | case-insensitive `\b(sort|search|factorial|fibonacci|prime|gcd|binary|bubble|merge)\b` matches `code + docstring` |
| `framework_application` | case-insensitive `\b(react|angular|vue|express|webpack|babel|grunt|gulp|jquery|bootstrap|element)\b` matches `code + docstring` |
| `compiler_tooling` | case-insensitive `\b(typescript|compiler|babel|eslint|webpack|parser|syntax|ast)\b` matches `code + docstring` |
| `api_wrapper` | case-insensitive `\b(fetch|axios|request|api|http|endpoint)\b` matches `code + docstring` |
| `incomplete_or_no_closing_brace` | JS only: stripped code does not end with `}` |
| `incomplete_or_no_colon` | Python only: stripped code does not end with `:` or `)` |
| `comment_like_description` | docstring starts with `#`, `//`, `/*`, `*`, or has fewer than 4 words |
| `meaningful_description` | complement of `comment_like_description` |

Overlaps are structural: the four keyword categories are independent regexes
over the same concatenated text, so a record can be `algorithmic` +
`api_wrapper` + `meaningful_description` at once (JavaScript example 7 below is
exactly that). `meaningful_description` / `comment_like_description` are the only
mutually exclusive pair. `concise_standalone` and the `incomplete_*` categories
can co-occur with any keyword category.

Known weaknesses, stated explicitly:

- `meaningful_description` is a *length-and-prefix* test, not a quality test. It
  counts structured JSDoc/`@param`/`{type}` blocks as meaningful prose even when
  they are machine-generated tag soup, and it accepts terse fragments such as
  "Populate the resolutions array". It cannot detect a docstring that merely
  repeats the signature, is a bare URL, or is boilerplate. It also rejects very
  short but perfectly good descriptions ("Returns the hash") because they fall
  under four words.
- Keyword hits are **not** proof of genre. `ast`, `parser`, `element` and `api`
  occur in ordinary identifiers and comments; a match only proves the token
  exists somewhere in the record. No claim that a record "is algorithmic",
  "is framework code" or "is compiler tooling" is supported by these labels.
- `incomplete_or_no_colon` fires on roughly half of Python records mostly because
  the trailing character of a multi-line function is often not `:` or `)`; it is
  a crude proxy, not a completeness measure.

Ten complete records per language, with description, code and heuristic labels,
are in **`reports/phase10f_label_examples.json`** (UTF-8, `ensure_ascii=False`,
generated by `scripts/phase10f_label_examples.py`), each with `record_id`, full
`description`, full `code` and `heuristic_labels`. Labels for those twenty
records:


**python** (from 290,463 records)

- `01` `335a76e23ae980897c1af9d41b5805f95682d1ea2439621ed065d7c26e15ed21` — `incomplete_or_no_colon`, `meaningful_description`
- `02` `2172af287fa54bdbc91f643eda52f48d826057bd2e47a192d1b5ea376b9b14fa` — `meaningful_description`
- `03` `9ca9f630584e4ec09a0c1a2a24640db795433e3fe33f8cd5ba3a307e03c191d8` — `concise_standalone`, `meaningful_description`
- `04` `8e5ddcb05c249e4ab692d49dd9b5e7ca452aa84b4d29baea0d5501993257d549` — `meaningful_description`
- `05` `5b1ab96afdd6b0a658968be290dba23fd5c8a1a488636105a245e27555318bf6` — `concise_standalone`, `incomplete_or_no_colon`, `meaningful_description`
- `06` `ad4bca49f409ce3ddc5ba220395a5068e3155bc7f4750d5e576279abdf1d485a` — `compiler_tooling`, `concise_standalone`, `meaningful_description`
- `07` `8301b83e0bf780b9260917be10e2ed15c7d3f00f7c2832131a15cd7e123914b5` — `meaningful_description`
- `08` `69f5b938426f053af9f38287032023d851502a09b42081542bf69a4839f145fc` — `incomplete_or_no_colon`, `meaningful_description`
- `09` `9fb775498730325bc24b7e8243537d92fe8872295148ebe92fc8de6433377057` — `meaningful_description`
- `10` `f86b1b863c3f8d961cc830fbd4c9b69bb0d4f6ed25a2a2aa2de989a14d6facc6` — `meaningful_description`

**javascript** (from 102,169 records)

- `01` `b068cb16a57ae463af3d8c53d1f288b43b7887b9ccff48d654a6fce9a22b0517` — `meaningful_description`
- `02` `53e4b4179e062fac9499b8264338a6a0f9d0006defe1d5447bdecda9a954d3cc` — `meaningful_description`
- `03` `35388d6db58c4b6adc52fe92805a38dc7b06dc7bfd06f65a7276c8ca51dde3ab` — `concise_standalone`, `meaningful_description`
- `04` `6438780ab15a381a3d7e424d003f33fd228b51c15f01251b24385ca83dcc458d` — `api_wrapper`, `meaningful_description`
- `05` `b5d19d63af9d5c513539c1360ab88dcea70d661146a3ff99b8d90861065fe79c` — `framework_application`, `meaningful_description`
- `06` `998c419c593b83b493ab80e6efeadbac93a85408609c783667a5d4df8578b563` — `algorithmic`, `meaningful_description`
- `07` `5d59814512e7a070854027da6e588f55335475e49e8a34857fa7bf95c31b6f32` — `algorithmic`, `api_wrapper`, `meaningful_description`
- `08` `4abd572c6b65696143b692a307274327d0b9fe49b44343ba4c2b903c1c177a2d` — `meaningful_description`
- `09` `0d818705f91e66495a5d5bf8552985580653665db9fb28921712048fb9a64e81` — `meaningful_description`
- `10` `ea7473cfa8a94099bdf7355db36bf211a18211831afaedd572999f7729f262d2` — `concise_standalone`, `meaningful_description`

---

## C. Token accounting reconciliation

`data/processed/tokenized_v1_1/manifest.json` reports:

| split | stream tokens | blocks | dropped trailing |
|---|---:|---:|---:|
| train | 133,192,957 | 260,142 | 252 |
| valid | 7,757,005 | 15,150 | 204 |
| test | 7,447,823 | 14,546 | 270 |

The test row is quoted from manifest metadata only; the test split data was not
opened, encoded or evaluated in this phase.

The audited Python + JavaScript totals sum to 130,345,937, which is
**2,847,020 tokens short** of the train stream. That gap is the English
documentation set: `prepare_dataset` builds `_train_code.bin` from the corpus,
then builds `_train_docs.bin` from `python_docs_training_corpus_inventory.csv`
(500 documentation files) and concatenates docs **after** the code, then deletes
the parts. The manifest lists two train sources:

```
code_records   records=392,632  partition=train
documentation  records=500      partition=train_only
```

Verified directly on `train.bin`:

- `train.bin` length = 133,192,957 tokens
- token at index 130,345,937 = `8` = `<|lang:en|>`; the preceding token = `2` = `<|eos|>`
- documentation region = 133,192,957 − 130,345,937 = **2,847,020 tokens**
- no third code language exists: `train.jsonl` contains exactly
  `{python: 290,463, javascript: 102,169}` and `valid.jsonl`
  `{python: 14,671, javascript: 6,294}`

So the reconciliation is:

| item | tokens |
|---|---:|
| Python code | 98,807,136 |
| JavaScript code | 31,538,801 |
| code subtotal | 130,345,937 |
| English documentation (500 files) | 2,847,020 |
| **train stream** | **133,192,957** |

Validation has no documentation, and its arithmetic closes exactly:

| item | tokens |
|---|---:|
| Python | 5,813,662 |
| JavaScript | 1,943,343 |
| **valid stream** | **7,757,005** |

Block consumption (see section D for the formula):

- train: 260,142 × 512 = 133,192,704 input positions, + 1 leading context token
  = 133,192,705 consumed; 133,192,957 − 133,192,705 = **252 trailing**
- valid: 15,150 × 512 = 7,756,800 input positions, + 1 = 7,756,801 consumed;
  7,757,005 − 7,756,801 = **204 trailing**

The "one-token difference" is the leading boundary token: `tokens_in_blocks =
(blocks − 1) × stride + window_length = blocks × 512 + 1`, because the 513-token
window is shared by neighbouring blocks rather than duplicated. Equivalently
`blocks × 513 − (blocks − 1) = blocks × 512 + 1` (train:
133,452,846 − 260,141 = 133,192,705; valid: 7,771,950 − 15,149 = 7,756,801).

The trailing counts are the un-windowed tail of each stream: they are the final
`token_count − tokens_in_blocks` tokens, which no 513-token window covers. For
train those 252 tokens sit at the very end of the documentation region (the token
at index 133,192,705 is `1885`, ordinary documentation text, not a marker or a
code example). No record is removed; only the useless partial tail is never
sampled.

### C1 — Records longer than the 512-token context

Measured by serializing every record with the production serializer and encoding
with tokenizer v1.1 (record-level token counts, one record at a time):

| split | language | records | > 512 tokens | share |
|---|---|---:|---:|---:|
| train | Python | 290,463 | 48,127 | 16.57% |
| train | JavaScript | 102,169 | 11,639 | 11.39% |
| valid | Python | 14,671 | 2,902 | 19.78% |
| valid | JavaScript | 6,294 | 741 | 11.77% |

Individual records are **never truncated and never split as records**: the whole
serialized sequence, including its `<|eos|>`, is written to the stream
contiguously. A long record is simply covered by several windows, so its
continuation appears as the beginning of later blocks.

---

## D. Block geometry, precisely

Building the blocks (`src/polycodegpt/data/dataset.py`):

```python
def block(self, index: int) -> np.ndarray:
    start = index * self.window_stride
    window = np.array(self.mm[start : start + self.window_length], dtype=np.int64, copy=True)
    ...
    return window

def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
    window = self.block(index)
    input_ids = torch.from_numpy(window[:-1])   # [512]
    targets = torch.from_numpy(window[1:])      # [512]
    return input_ids, targets
```

Geometry from `src/polycodegpt/data/stream.py::block_geometry`:

```python
window_length = context_length + 1                  # 513
stride = context_length if stride is None else stride   # 512
blocks = 1 + (token_count - window_length) // stride
tokens_in_blocks = (blocks - 1) * stride + window_length
dropped_trailing_tokens = token_count - tokens_in_blocks
```

- **Input length** 512 tokens; **target length** 512 tokens (target is the input shifted by one).
- **Stride** 512, equal to the context length; overlap between windows is 1 token.
- **Do adjacent input blocks overlap?** No. Block `i` uses stream positions
  `[i*512, i*512+512)` and block `i+1` uses `[(i+1)*512, (i+1)*512+512)`. The
  input slices are disjoint and cover the stream exactly once. What overlaps is
  the 513-token *window*: position `i*512+512` is the last target of block `i`
  and the first input of block `i+1`. Calling the inputs "overlapping" is wrong;
  the shared token is counted once as a target and once as context.
- **Why 513?** Each 513-window yields 512 supervised next-token pairs. If a
  512-token window were used with stride 512, the last token of every window
  could never be predicted (no target inside the window) and one prediction per
  boundary would be silently lost. With 513 the target ranges
  `[i*512+1, i*512+512]` tile the stream exactly: every position from 1 onward is
  a target exactly once, position 0 is context only.
- **Records crossing a boundary** are cut by the window, with no padding,
  sentinel or mask. The model therefore sees mid-function continuations, and a
  window may start in the middle of a docstring. Measured on the first 20,000
  train blocks, only 52 (0.26%) start on a language marker, i.e. record
  boundaries almost never align with block boundaries.
- **EOS at boundaries** is preserved wherever it falls, including in the middle
  of a block. In the first 2,000 train blocks, 1,340 (67%) contain at least one
  EOS. Because records are concatenated, the token after an EOS is the next
  record's `<|lang:...|>` marker (verified: first EOS at index 957 is followed by
  token `6`, `<|lang:python|>`), so the model is trained to emit EOS and then
  continue into an unrelated record rather than to stop.
- **Attention context resets at every block.** Training calls
  `model(inputs, targets=targets)` (`scripts/train.py`) with no `attn_mask`, no
  KV cache and no carried state; each block is an independent causal sequence of
  at most 512 tokens. Any dependency spanning a block boundary is simply
  invisible to the model.

---

## E. Serialization evidence (complete real training sequences)

### E1 — Python (`2a0704d65bcaf8df8c0358c426f27d34c8c6ad6bf89d44f98e44ec8f87593c88`)

- record: `ageitgey/face_recognition` / `examples/face_recognition_knn.py`
- serialized length: **958 tokens**; first id `6` (`<|lang:python|>`), last id `2` (`<|eos|>`), EOS count `1`
- marker IDs: `<|docstring|>` = `4` at position 1, `<|code|>` = `3` at position 235, `<|eos|>` = `2` at position 957

Complete serialized sequence (literal, unmodified; long code shown in full):

```text
<|lang:python|><|docstring|>Trains a k-nearest neighbors classifier for face recognition.

    :param train_dir: directory that contains a sub-directory for each known person, with its name.

     (View in source code to see train_dir example tree structure)

     Structure:
        <train_dir>/
        ├── <person1>/
        │   ├── <somename1>.jpeg
        │   ├── <somename2>.jpeg
        │   ├── ...
        ├── <person2>/
        │   ├── <somename1>.jpeg
        │   └── <somename2>.jpeg
        └── ...

    :param model_save_path: (optional) path to save model on disk
    :param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified
    :param knn_algo: (optional) underlying data structure to support knn.default is ball_tree
    :param verbose: verbosity of training
    :return: returns knn classifier that was trained on the given data.<|code|>def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    """
    Trains a k-nearest neighbors classifier for face recognition.

    :param train_dir: directory that contains a sub-directory for each known person, with its name.

     (View in source code to see train_dir example tree structure)

     Structure:
        <train_dir>/
        ├── <person1>/
        │   ├── <somename1>.jpeg
        │   ├── <somename2>.jpeg
        │   ├── ...
        ├── <person2>/
        │   ├── <somename1>.jpeg
        │   └── <somename2>.jpeg
        └── ...

    :param model_save_path: (optional) path to save model on disk
    :param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified
    :param knn_algo: (optional) underlying data structure to support knn.default is ball_tree
    :param verbose: verbosity of training
    :return: returns knn classifier that was trained on the given data.
    """
    X = []
    y = []

    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf<|eos|>
```

Matching generation prefix (docstring kept, code replaced by the first code line as the stub):

```text
<|lang:python|><|docstring|>Trains a k-nearest neighbors classifier for face recognition.

    :param train_dir: directory that contains a sub-directory for each known person, with its name.

     (View in source code to see train_dir example tree structure)

     Structure:
        <train_dir>/
        ├── <person1>/
        │   ├── <somename1>.jpeg
        │   ├── <somename2>.jpeg
        │   ├── ...
        ├── <person2>/
        │   ├── <somename1>.jpeg
        │   └── <somename2>.jpeg
        └── ...

    :param model_save_path: (optional) path to save model on disk
    :param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified
    :param knn_algo: (optional) underlying data structure to support knn.default is ball_tree
    :param verbose: verbosity of training
    :return: returns knn classifier that was trained on the given data.<|code|>def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
```

- prefix length 270 tokens, first six ids `[6, 4, 3788, 1467, 282, 904]`, `<|eos|>` present: `False`
- `full[:270]` minus the stub equals the prefix ids: the prefix is an exact token-level prefix of the training sequence

### E2 — JavaScript (`09428776b30d3c2daab225810a86c3c010520f2b7f0bddb7ced88d030ab23381`)

- record: `Microsoft/vscode` / `build/lib/treeshaking.js`
- serialized length: **272 tokens**; first id `7` (`<|lang:javascript|>`), last id `2` (`<|eos|>`), EOS count `1`
- marker IDs: `<|docstring|>` = `4` at position 1, `<|code|>` = `3` at position 11, `<|eos|>` = `2` at position 271

Complete serialized sequence (literal, unmodified; long code shown in full):

```text
<|lang:javascript|><|docstring|>#region Discovery, LanguageService & Setup<|code|>function createTypeScriptLanguageService(options) {
    // Discover referenced files
    const FILES = discoverAndReadFiles(options);
    // Add fake usage files
    options.inlineEntryPoints.forEach((inlineEntryPoint, index) => {
        FILES[`inlineEntryPoint.${index}.ts`] = inlineEntryPoint;
    });
    // Add additional typings
    options.typings.forEach((typing) => {
        const filePath = path.join(options.sourcesRoot, typing);
        FILES[typing] = fs.readFileSync(filePath).toString();
    });
    // Resolve libs
    const RESOLVED_LIBS = {};
    options.libs.forEach((filename) => {
        const filepath = path.join(TYPESCRIPT_LIB_FOLDER, filename);
        RESOLVED_LIBS[`defaultLib:${filename}`] = fs.readFileSync(filepath).toString();
    });
    const compilerOptions = ts.convertCompilerOptionsFromJson(options.compilerOptions, options.sourcesRoot).options;
    const host = new TypeScriptLanguageServiceHost(RESOLVED_LIBS, FILES, compilerOptions);
    return ts.createLanguageService(host);
}<|eos|>
```

Matching generation prefix (docstring kept, code replaced by the first code line as the stub):

```text
<|lang:javascript|><|docstring|>#region Discovery, LanguageService & Setup<|code|>function createTypeScriptLanguageService(options) {
```

- prefix length 22 tokens, first six ids `[7, 4, 13, 3428, 4813, 6245]`, `<|eos|>` present: `False`
- `full[:22]` minus the stub equals the prefix ids: the prefix is an exact token-level prefix of the training sequence

Marker order is identical everywhere: `<|lang:X|>` → `<|docstring|>` → text → `<|code|>` → code → `<|eos|>`. The generation prefix emits the first four and stops after the stub, omitting only the `<|eos|>` continuation terminator (and, by construction, the code that would follow the stub).

---

## F. Verified findings, open questions, sources

### Verified

1. `top_k=None` disables truncation; `top_k=50` masks below the 50th logit; `temperature=0` bypasses both and uses `argmax`. No hidden defaults, no top-p, no repetition penalty.
2. T=0.4 with and without top-50 produce byte-identical output because the removed mass is ~3e-7; at T=0.8 the removed mass is ~2e-3 and 3–5 of 10 sequences diverge. The earlier "identical across all seeds" claim holds only for T=0.4.
3. Seeds are applied to the PyTorch global RNG, reproduce exactly, and change output; greedy is deterministic. Seeds are set once per run, so the ten tasks within a run share one RNG stream.
4. Every stored candidate is an independent generation and is saved separately (10/10 unique per run).
5. Dataset labels are mechanical heuristics, multi-label and overlapping; `meaningful_description` is a word-count/prefix test that over-counts structured JSDoc.
6. Train stream 133,192,957 = 130,345,937 code (Python 98,807,136 + JavaScript 31,538,801) + 2,847,020 English documentation tokens from 500 files. Validation 7,757,005 is exactly Python + JavaScript.
7. Block arithmetic closes: train 260,142 × 512 + 1 = 133,192,705 used, 252 trailing; valid 15,150 × 512 + 1 = 7,756,801 used, 204 trailing. Trailing tokens are an un-windowed documentation tail, not dropped records.
8. Input slices are disjoint; only the 513-token windows overlap, by exactly the boundary token that each block learns to predict. Records are never truncated or split as records, but windows cut them, may start mid-record, and keep EOS mid-block (67% of sampled blocks contain an EOS).
9. Attention resets per block: no mask, no cache, no cross-block state.
10. Serialization order is `<|lang:X|>` → `<|docstring|>` → text → `<|code|>` → code → `<|eos|>`, identical in the stream and in generation prefixes, which omit only the EOS terminator.

### Unresolved

- Whether the sampling gains (up to 9/10 EOS at seed 999) are a property of the model or of one lucky RNG stream; three correlated replicate sequences on ten prompts cannot separate the two, and per-task independent seeds were not run.
- Why seed 999 is consistently far better than 42 and 123 on these ten prompts is unexplained; it suggests high variance from a single early token choice and needs more prompts to characterise.
- Whether the 16.6% / 11.4% of records above 512 tokens degrade training in a way that matters, and whether the mid-block EOS continuation behaviour harms stopping, is a hypothesis, not a measured result.
- The heuristic labels cannot separate "comment-like" from "tag-structured" descriptions; a manual or LLM-assisted audit of a stratified sample is still needed before any data change.
- English documentation (2.85M tokens) participates in training but was not evaluated for language-mixing effects.

### Inspected sources

- `src/polycodegpt/model/gpt.py` — `generate`, `_generate_loop` (top-k, temperature, EOS, RNG).
- `scripts/phase10f_decoding.py` — settings, seeding, candidate capture, ratios.
- `src/polycodegpt/data/prepare.py` — `validate_example`, `prepare_code_split`, `prepare_documents`, `concatenate_streams`, `prepare_dataset` (docs appended last).
- `src/polycodegpt/data/stream.py` — `block_geometry`, `TokenStreamWriter`.
- `src/polycodegpt/data/dataset.py` — `TokenBlockDataset.block`, `__getitem__`, `build_dataloader`, `open_split`.
- `src/polycodegpt/tokenization/serializer.py` — `serialize_code_record`, `serialize_documentation`.
- `scripts/evaluate_phase10_benchmark.py` — `build_generation_prefix`, `tokenize_prompt`, `compute_syntax_validity`.
- `data/processed/tokenized_v1_1/manifest.json` — split tokens, blocks, trailing counts, sources.

### Tests run

```bash
cd PolyCodeGPT
../.venv/Scripts/python -m pytest tests/unit/test_data_pipeline.py tests/unit/test_validation_loss_breakdown.py -q
```

Result: 52 passed. These cover `block_geometry` invariants
(`token_count == blocks*stride + 1 + dropped_trailing_tokens`), window length 513,
stride 512, manifest/dataset agreement, and `TokenBlockDataset` slicing.

### Commands used for this reconciliation

```bash
cd PolyCodeGPT
../.venv/Scripts/python scripts/phase10e_data_audit.py
../.venv/Scripts/python scripts/phase10f_label_examples.py
../.venv/Scripts/python scripts/phase10f_report.py
```

No diagnostic was re-run: source inspection plus the three-prompt measurement
showed the stored experiment was valid, so the existing
`reports/phase10f_decoding_diagnostics.json` was reused unchanged.

### Artifacts

- `reports/phase10f_reconciliation.md` (new, this file)
- `reports/phase10f_label_examples.json` (new, UTF-8, 20 complete records)
- `reports/phase10f_decoding_diagnostics.json` (pre-existing, unchanged)
- `reports/phase10e_inference_diagnostics.json`, official Phase 10 report, checkpoint step 8130 — untouched
