# Phase 16 — Deep Technical Diagnosis

Strictly read-only investigation. No model was trained, no benchmark was run, no candidate code was executed, and no existing script, config, checkpoint, tokenizer, dataset, manifest, report, or log was modified. The old test partition was not opened. Individual tasks, cases, prompts, or outputs from `functional/independent_v3/` were not inspected; only aggregate historical results were used. This report is the only new file.

---

## 1. Executive summary

PolyCodeGPT v0.1.0 (Exp C) is a clean, well-instrumented 16,988,160-parameter decoder-only Transformer trained from scratch on CodeSearchNet Python and JavaScript plus 500 Python documentation files. Its engineering is substantially stronger than its modelling results. The training pipeline is reproducible, hash-checked, resumable, and honest about failures. The model itself is still a small, undertrained, function-level code language model with weak algorithmic competence and a severe Python/JavaScript imbalance.

The most important diagnosis is **not** “the model is too small.” The more consequential problems are:

1. **The effective unique training corpus is small and narrow.** 133,192,957 training tokens is roughly 7.8 tokens per parameter. Exp C does not add new unique data: its second 133M-token stage is a re-weighting of the same records, with the curated JavaScript subset repeated about 10 times.
2. **The data is function-level library code, not task-level algorithm code.** CodeSearchNet records are real functions with docstrings, often dependent on project context and not self-contained. The benchmark asks for standalone algorithm implementations. That is a distribution mismatch in *kind*, not just in size.
3. **JavaScript data quality is the binding constraint for JavaScript.** The original JavaScript corpus is 102,169 functions, but only 35,658 survive curation, and those 35,658 contain only 6,542,851 tokens. Exp C fills 65.2M JavaScript tokens by repeating that small set 9.96 times.
4. **The architecture is a sound baseline but is not the right long-context or efficiency design for the next generation.** Learned absolute positions do not extrapolate, attention is manual O(T²), there is no Flash/SDPA path, and the 512-token window prevents multi-function and module-level context.
5. **The training schedule and validation protocol are adequate for a pilot but too weak for model selection.** A fixed 64-block validation subset is reused every 100 steps, full validation happens only at the end, and there is no early stopping. Best checkpoints were generally the final step.
6. **Functional performance is low and error patterns are diagnostic, not random.** Exp C passes 4/100 independent tasks and 39/328 cases. Failures concentrate in syntax for JavaScript, runtime errors from invented attributes/methods in Python, and incorrect return values/logic. This is consistent with a model that has learned surface form and local patterns, not reliable program semantics.

The correct next step is a **data-and-evaluation-first generation**, followed by a modest architecture upgrade and a controlled capacity increase. Proposal **B (≈65M parameters, 1024 context, 300–600M tokens)** is the best balance only if the JavaScript and Python corpora are materially improved first. Proposal **C (≈110M parameters)** is not justified yet and is not guaranteed to fit or finish comfortably on the 8 GB laptop GPU. Proposal **A (≈34M parameters)** is the lowest-risk fallback.

---

## 2. Architecture investigation

### 2.1 Measured configuration

Source: `configs/model.yaml`, `src/polycodegpt/model/config.py`, `gpt.py`, `layers.py`, `attention.py`, and the trained checkpoint metadata.

| Property | Measured value |
|---|---|
| Architecture | Decoder-only Transformer |
| Parameters | 16,988,160 |
| Vocabulary | 16,000 |
| Layers | 6 |
| Attention heads | 6 |
| Model dimension | 384 |
| Head dimension | 64 |
| FFN dimension | 1,536 (4× model dimension) |
| Context length | 512 |
| Positional encoding | Learned absolute embeddings |
| Normalization | Pre-LayerNorm, `eps=1e-5` |
| Activation | GELU |
| Embedding tying | Input embedding tied to output projection |
| Dropout | 0.1 |
| Attention | Manual scaled dot-product, causal mask |
| Initialization | Normal(0, 0.02); residual projections scaled by `0.02 / sqrt(2·n_layers)` |

The analytical parameter count in `ModelConfig.estimate_parameters()` matches the trained model exactly: **16,988,160**.

### 2.2 Parameter allocation

| Component | Parameters | Share |
|---|---:|---:|
| Token embedding (tied, counted once) | 6,144,000 | 36.2% |
| Positional embedding | 196,608 | 1.2% |
| Attention projections (6 layers) | 3,548,160 | 20.9% |
| Feed-forward networks (6 layers) | 7,089,408 | 41.7% |
| LayerNorm parameters | 9,984 | 0.06% |
| **Total** | **16,988,160** | **100%** |

The feed-forward network is the largest trainable block, as expected. The token embedding is unusually large relative to the rest of the model because the vocabulary is 16,000 and the model dimension is only 384. Weight tying avoids a second 6.14M-parameter output matrix; without tying the model would be 23.1M parameters. Tying is the right choice at this scale, but it also means the same 384-dimensional space must serve both input semantics and output logits, which limits specialization.

### 2.3 Architectural strengths

- **Pre-LayerNorm residual blocks** are stable at this depth and avoid the warmup sensitivity of post-LN Transformers.
- **Weight tying** is appropriate for a small model and materially reduces parameters.
- **Causal masking is explicit and correct**; future positions are masked before softmax.
- **The model is small enough to iterate quickly** on the available GPU and to inspect end-to-end.
- **The implementation is readable and testable**: dimensions are validated, sequence length is checked, and the forward path has no hidden framework abstractions.
- **Residual projection scaling** (`init_std / sqrt(2·n_layers)`) is a reasonable small-scale stabilization.
- **BF16 autocast with FP32 master weights** is used in training, and the code refuses silent CPU fallback.

### 2.4 Architectural limitations

**Attention implementation.** Attention is written manually with separate Q, K, V, and output projections. Scores are materialized as a `[B, heads, T, T]` tensor, a boolean causal mask is allocated on every forward pass, and softmax is followed by `torch.nan_to_num`. There is no `scaled_dot_product_attention`, FlashAttention, paged attention, or memory-efficient kernel. At 512 tokens this is tolerable. At 1,024–2,048 tokens it becomes a real memory and throughput problem, and it prevents the next generation from using longer context cheaply.

**Positional encoding.** Learned absolute positions work for the training window but do not extrapolate. A model trained at 512 cannot be evaluated at 1,024 without either interpolation or retraining. Absolute positions also give the model no explicit relative-distance signal, which matters for code structure, matching delimiters, and long-range identifier references.

**Context window.** 512 tokens is enough for a single short function, but not for a function plus its surrounding class/module, multiple related functions, or a docstring plus a nontrivial implementation. The training stream does concatenate examples, so the model does see cross-example transitions, but it cannot condition on a full file. The benchmark prompts are short, which hides this limitation; real code assistance will expose it.

**Normalization and residual path.** LayerNorm is standard and stable, but RMSNorm would be cheaper and is now common in code LLMs. There is no QK normalization. At 6 layers this is fine; at 12+ layers with higher learning rates, attention logits can grow and destabilize training. The residual path has no depth-scaled residual lambda beyond the output-projection initialization.

**Embedding and output projection.** Tying is efficient, but the 16,000-token byte-level BPE vocabulary is small for code. Identifiers, framework names, and long literals are split into many subwords. A larger vocabulary (24k–32k) would shorten sequences and reduce attention cost, but it requires retraining the tokenizer and rebuilding the dataset, and it changes the checkpoint compatibility contract.

**Numerical stability.** BF16 autocast is stable in the observed runs: no skipped optimizer steps, finite losses, and finite gradient norms. The manual attention path uses `-inf` masking and `nan_to_num`, which can hide a fully masked row instead of failing loudly. The training loop checks for non-finite loss and gradient norm, which is good. There is no gradient clipping issue, but there is also no QK norm or logit cap for deeper models.

**Efficiency.** The model uses a separate `lm_head` call over the full vocabulary at every position. With tied weights this is a 384×16,000 matrix multiply, which is fine at 17M parameters but becomes a larger share of compute at longer contexts and larger vocabularies. There is no fused cross-entropy, no fused optimizer, and no activation checkpointing in the current model code.

### 2.5 Is the architecture appropriate for the next generation?

**As a baseline: yes.** It is a legitimate, from-scratch decoder-only Transformer with the right basic ingredients.

**As the final architecture for the next generation: not without changes.** The next model should keep the decoder-only, pre-norm recipe but replace or add:

- **RoPE** instead of learned absolute positions, to support context extension and relative structure.
- **RMSNorm** for cheaper normalization, with QK normalization if depth exceeds ~10 layers.
- **SwiGLU or a gated FFN** instead of plain GELU, for better quality per parameter.
- **SDPA/FlashAttention** instead of manual score materialization.
- **Grouped-query attention (GQA)** if the model grows enough that KV cache and attention memory matter.
- **Gradient checkpointing** as a first-class training option for larger models on 8 GB.
- **A larger tokenizer vocabulary** only after a data review, because it invalidates v0.1.0 compatibility.

These are quality and efficiency changes, not a claim that scale alone will fix the baseline. The data and objective problems in §3 and §5 are larger.

---

## 3. Training data investigation

### 3.1 Corpus composition and token totals

Sources: `data/manifests/polycodegpt_corpus_manifest.csv`, `polycodegpt_corpus_validation.json`, `data/processed/tokenized_v1_1/manifest.json`, and the per-language index files under `data/processed/phase12b_work/`.

| Split | Python records | JavaScript records | Total records | Tokens |
|---|---:|---:|---:|---:|
| Train | 290,463 | 102,169 | 392,632 | 133,192,957 |
| Validation | 14,671 | 6,294 | 20,965 | 7,757,005 |

The training split also includes **500 Python documentation files** (2,847,020 serialized tokens). Code records contribute 130,345,937 tokens; documentation contributes 2.85M tokens. The tokenizer manifest records 524,981,058 serialized characters for code and 11,112,983 for documentation, giving roughly 0.25 tokens per character in both streams.

Unique repositories: 24,517 in the training partition, 27,221 across the full corpus. The corpus was partitioned by normalized repository with `TEST > VALID > TRAIN`, and the validation report records **zero partition overlaps**, zero missing fields, zero non-permissive records, and zero incomplete provenance.

### 3.2 Language balance

| Measure | Python | JavaScript | Ratio |
|---|---:|---:|---:|
| Training records | 290,463 | 102,169 | 2.84 : 1 |
| Training code tokens | 98,807,136 | 31,538,801 | 3.13 : 1 |
| Validation records | 14,671 | 6,294 | 2.33 : 1 |

The baseline is strongly Python-heavy. Exp C changes the *sampled* code-token budget to approximately 1:1 (65,172,968 Python + 65,172,969 JavaScript), but this is achieved almost entirely by repetition, not by new JavaScript data.

### 3.3 Token-length distributions

Measured from the per-language tokenized record indexes (serialized record length in tokens, including markers and EOS):

| Corpus | Records | Mean | Median | p90 | p95 | p99 | Max | >512 tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Python train | 290,463 | 340.2 | 218 | 683 | 978 | 1,989 | 37,998 | 16.57% |
| JavaScript train (original) | 102,169 | 308.7 | 178 | 554 | 816 | 1,931 | 624,879 | 11.39% |
| JavaScript curated | 35,658 | 183.5 | 154 | 338 | 402 | 481 | 512 | 0.00% |
| Python documentation | 500 | 5,694.0 | 2,576 | 15,699 | 22,258 | 32,533 | 53,563 | 85.00% |

The important facts:

- The median code record is short (154–218 tokens), so many training examples fit entirely inside a 512-token block.
- A nontrivial minority of code (11–17%) exceeds 512 tokens and is therefore split across blocks.
- Documentation is almost entirely longer than 512 tokens (85%), so documentation examples are split across many blocks and the model rarely sees a whole document.
- The curated JavaScript set is capped at 512 tokens, which makes it convenient for block training but removes long functions and modules from the JavaScript distribution.

### 3.4 Programming-construct coverage

There is no AST-level construct census in the existing reports. The available evidence is indirect:

- CodeSearchNet is **function-level**: each record is a function or method with a docstring. The corpus is not a balanced sample of complete programs.
- JavaScript structural audit: 102,167 of 102,169 records are `function_decl`; only 2 are “other.” The JavaScript corpus is therefore overwhelmingly function declarations, not classes, modules, arrow functions, async functions, or scripts.
- Python records are also functions/methods, but no equivalent structural census was recorded. Class bodies, module-level code, imports, decorators, and multi-function files are under-represented by construction.
- The curated JavaScript filter explicitly removes `stateful_or_method` records (11,253) and `depends_on_ecosystem` records (6,679), which removes object-oriented and module-dependent code. That improves standalone-ness but narrows construct coverage further.

**Conclusion:** construct coverage is narrow and not measured. The model sees many small, self-contained functions and relatively few classes, modules, async patterns, error-handling patterns, or multi-function programs. This is a plausible contributor to weak control-flow and data-structure performance.

### 3.5 Function and class coverage

- Function coverage: high in count (392,632 training records), but concentrated in library helper functions.
- Class coverage: not measured; CodeSearchNet’s function-level extraction means class definitions are not the unit of training. Any class code is incidental inside function bodies or method snippets.
- Method coverage: present in Python, but the metadata does not distinguish methods from free functions. JavaScript methods are deliberately filtered out by the `stateful_or_method` rule.
- Multi-function composition: not represented as a supervised unit. The model never sees “here are three functions that call each other” as a single example, except accidentally through block concatenation.

### 3.6 Algorithmic diversity

No algorithmic-diversity metric exists in the repository. Qualitatively, the corpus is dominated by real project code: utility functions, string/list manipulation, framework glue, API wrappers, and library internals. That is useful for learning syntax and naming, but it is not a curated algorithms corpus. The benchmark’s tasks (factorial, fibonacci, binary search, sorting, etc.) are simple algorithm exercises; the training corpus may contain such functions, but they are a small and unmeasured fraction. There is no guarantee that the model sees balanced coverage of recursion, iteration, dynamic programming, graph traversal, arithmetic, or data-structure manipulation.

### 3.7 Duplicate and near-duplicate risk

Measured evidence:

- All training and validation records have unique `record_id` and unique `code_sha256`; there are **0 exact code duplicates across train and validation**.
- Phase 7 reports **21,676 extra occurrences across 21,676 duplicate docstring groups**. The most common repeated docstring is `"Auto Generated Code"` (1,779 occurrences), followed by `"Return a json dictionary representing this model."` (286). These are weak description/code pairs.
- Phase 7 heuristic repetition indicators: 6,931 Python and 22,507 JavaScript records have repeated non-blank line ratio > 0.2; 2,716 Python and 1,971 JavaScript records have a 4-gram repeat count > 5.
- JavaScript curation applied MinHash near-duplicate detection and rejected 72 near-duplicates, but Python has **no near-duplicate pass**.
- There are 1,387 Python and 2,184 JavaScript bundle/module-marker records, and 720 Python / 281 JavaScript minified-like records. These are retained.

**Risk:** exact duplication is controlled; near-duplication in Python is not. More importantly, Exp C’s JavaScript oversampling creates *training-time* duplication: the same 35,658 curated records are repeated up to 10 times. That is a deliberate re-weighting, but it is also a memorization risk and a reason not to interpret Exp C’s JavaScript gains as generalization.

### 3.8 Documentation-to-code ratio

Documentation is 2,847,020 of 133,192,957 training tokens, or **2.14%**. There are 500 documentation files versus 392,632 code records. Documentation is Python-only; there is no JavaScript documentation corpus in training.

This ratio is low for learning description-to-code mapping. Most natural-language supervision comes from short code docstrings (median 33 tokens, mean 65.9 tokens in Phase 7). Those docstrings are often terse, generic, or auto-generated. The model therefore gets little explicit instruction in “read a requirement, write a program.”

### 3.9 Curated JavaScript corpus coverage and limitations

`data/curated/js_v1` contains **35,658 records**, 8,341 distinct repositories, and **6,542,851 tokens**. It was derived from the 102,169 original JavaScript training records; 66,511 were rejected.

| Rejection reason | Count |
|---|---:|
| `parse_invalid` | 32,830 |
| `description_fragment` | 12,309 |
| `stateful_or_method` | 11,253 |
| `depends_on_ecosystem` | 6,679 |
| `code_too_long` | 2,399 |
| `not_standalone_function` | 497 |
| `description_boilerplate` | 367 |
| `name_mismatch` | 105 |
| `near_duplicate` | 72 |
| `description_missing` | 0 |
| `duplicate` | 0 |
| `code_too_short` | 0 |

Strengths: every retained record parses, is a standalone function, has a non-fragment description, and fits within 512 serialized tokens. This is a meaningful quality improvement over the raw JavaScript.

Limitations:

- Only 34.9% of original JavaScript survives. The curated set is small: 6.54M tokens.
- It removes methods, stateful code, ecosystem-dependent code, and long functions. Those are exactly the constructs a general JavaScript model needs.
- It is still function-level, not program-level.
- The description categories are mostly `plain_text` / line comments; only 7 records have JSDoc blocks in the original audit. The model is not learning idiomatic JSDoc.
- Exp C repeats this set **9.96 times** to fill its JavaScript budget. Exp A repeats it 4.82 times. This is the single clearest data-side weakness of the selected candidate.

### 3.10 Distribution mismatch between training and intended tasks

| Dimension | Training distribution | Benchmark / intended task distribution |
|---|---|---|
| Unit | Real project function with docstring | Standalone algorithm function with docstring and stub |
| Context | Project-specific imports, helpers, globals | Self-contained, no imports |
| Length | Median 154–218 tokens; long tail to 624k | Short prompts, 64 new tokens in evaluation |
| Description | Terse or generic docstrings; 2.14% docs | Clear English requirement |
| Language balance | 3.13:1 Python:JavaScript | 1:1 tasks |
| Constructs | Library helpers, framework glue | Recursion, loops, arithmetic, data structures |
| Correctness signal | Next-token likelihood | Execution against hidden cases |

The benchmark is not asking the model to reproduce repository code. It is asking for correct algorithmic implementations. The training data teaches the surface statistics of real code much better than it teaches algorithmic problem solving. This mismatch is more important than the raw token count.

---

## 4. Serialization and tokenizer investigation

### 4.1 Tokenizer training

Source: `src/polycodegpt/tokenization/train.py`, `serializer.py`, `data/tokenizer_io.py`, and `tokenizer/artifacts/tokenizer_metadata_v1_1.json`.

| Property | Value |
|---|---|
| Algorithm | Byte-level BPE (`tokenizers.models.BPE` + ByteLevel pre-tokenizer/decoder) |
| Vocabulary | 16,000 |
| Merges | 15,738 |
| Added special tokens | 11 |
| Minimum frequency | 2 |
| Normalizer | None |
| Lowercasing | No |
| Byte fallback | No |
| Training inputs | 392,632 code records + 500 Python docs |
| Training characters | 524,981,058 code + 11,112,983 docs |
| Fit time | 33.44 s |
| Peak RAM | 1,051.4 MB |
| Artifact SHA256 | `0459f8556a6f79ca1ba1c3dc3f76c26ae2e55653ebbf97beaa9bd23467358a03` |

The tokenizer is trained **only on the training partition**. `valid.jsonl` and `test.jsonl` are explicitly rejected by `_assert_train_only`. Special-token IDs are fixed:

| Token | ID |
|---|---:|
| `<|pad|>` | 0 |
| `<|unk|>` | 1 |
| `<|eos|>` | 2 |
| `<|code|>` | 3 |
| `<|docstring|>` | 4 |
| `<|text|>` | 5 |
| `<|lang:python|>` | 6 |
| `<|lang:javascript|>` | 7 |
| `<|lang:en|>` | 8 |
| `<|lang:fr|>` | 9 |
| `<|lang:ar|>` | 10 |

No normalizer and no lowercasing are correct for code: whitespace, indentation, case-sensitive identifiers, and punctuation all survive. Byte-level BPE also guarantees that any input is representable. The 16k vocabulary is small for code; common identifiers and framework names are split into several tokens. That increases sequence length and makes long-range dependencies harder.

### 4.2 Dataset serialization

The serializer v1.1.0 emits three record forms:

```text
<|lang:python|><|docstring|>{docstring}<|code|>{code}<|eos|>
<|lang:javascript|><|docstring|>{docstring}<|code|>{code}<|eos|>
<|lang:en|><|text|>{rst}<|eos|>
```

Content is escaped reversibly: backslashes are doubled, and literal control-token strings are broken (`<|eos|>` becomes `<\<|eos|>` in escaped form). Structural markers are inserted after escaping, so content can never be confused with a marker. Every serialized example starts with a language marker and ends with exactly one EOS. `prepare.py` enforces this contract for every record before it is written to the token stream.

This is a strong design. It gives the model an explicit language switch, an explicit description boundary, an explicit code boundary, and an explicit end-of-example signal. It also makes the stream parseable without escape-awareness.

### 4.3 Training block construction

The token stream is a flat little-endian `uint16` array. Blocks are windows of `context_length + 1 = 513` tokens with stride 512. For block `i`, `input_ids = window[:-1]` and `targets = window[1:]`, so every target is the next token. Consecutive blocks overlap by exactly one token, which ensures that no next-token target is lost at a block boundary: the shared token is the last target of block `i` and the first input of block `i+1`.

Measured geometry:

| Split | Tokens | Blocks | Dropped trailing tokens |
|---|---:|---:|---:|
| Train | 133,192,957 | 260,142 | 252 |
| Validation | 7,757,005 | 15,150 | 204 |

The manifest says `examples_never_split: true` and also `blocks_may_cross_example_boundaries: true`. These two statements are in tension. Because examples are concatenated into one flat stream and blocks step by 512 tokens, a block can contain the tail of one example and the head of the next. Examples are therefore *not* kept intact at block level. The practical consequence is that the model sees EOS followed by a new language marker inside the same 512-token window, which may teach separators but also creates unnatural transitions. This should be made explicit in the next data design rather than hidden behind a misleading field name.

### 4.4 Generation prefix and EOS handling

`scripts/evaluate_phase10_benchmark.py` builds the generation prefix as:

```text
<|lang:X|><|docstring|>{docstring}<|code|>{code_stub}
```

It deliberately does **not** append EOS. The model is expected to continue the code and emit EOS when finished. The training sequence includes EOS; the generation prefix is the same prefix without the final EOS. This is consistent: the model is trained to predict code tokens and then EOS, and at inference it is given the prefix and asked to continue.

The generation path:

- Tokenizes the prefix with `add_special_tokens=False`.
- Asserts the prefix contains no EOS.
- Calls `model.generate` with temperature 0.0 (greedy), `max_new_tokens=64`, and EOS ID 2.
- Extracts only the generated tokens after the prompt.
- Removes EOS from the candidate used for syntax checking.
- Does not execute code outside the sandbox.

One interface difference matters: `GPT.forward` raises if `T > context_length`, but `GPT._generate_loop` slices the context to the last 512 tokens. A prompt longer than 512 tokens is therefore silently left-truncated during generation, while the same overlong sequence would raise in training. This is a real consistency gap. The benchmark prompts are short, so it did not surface, but future long prompts need explicit truncation policy and logging.

### 4.5 Does the serialization teach the intended skills?

| Skill | Design support | Assessment |
|---|---|---|
| Description-to-code generation | `<|docstring|>` then `<|code|>`; docstring is always before code | **Supported in form, weak in supervision.** Docstrings are short and often generic; only 2.14% of tokens are documentation. |
| Function signature continuation | Code stub is placed after `<|code|>`; model continues from the stub | **Supported.** The benchmark stubs are function signatures; training examples include full function definitions with signatures. |
| Correct identifier usage | Full code including identifiers is in the target | **Partially supported.** The model sees identifiers, but it is not trained to copy a provided signature into the body with a consistency objective. Failures show invented attributes and methods. |
| Indentation and syntax | Byte-level BPE, no normalizer, no lowercasing; code is emitted verbatim | **Supported.** Python syntax validity is much higher than JavaScript, consistent with the Python-heavy corpus. |
| Multi-line function completion | Full functions are targets; EOS marks the end | **Supported for short functions.** Blocks can split long functions; the model rarely sees a complete long function in one window. |
| Python/JavaScript separation | Explicit language marker at the start of every example | **Supported in form, imbalanced in data.** The model can condition on language, but JavaScript has far less unique data and much lower syntax validity. |

**Conclusion:** the serializer and prompt format are consistent and well designed. The weakness is not the format; it is the content and balance of what is being serialized, plus the block-level splitting of examples.

---

## 5. Training-system investigation

### 5.1 Actual training loop

Source: `src/polycodegpt/training/engine.py`.

- One optimizer step consumes `gradient_accumulation_steps = 8` micro-batches.
- Micro-batch size is 4 sequences, each 512 tokens.
- Effective batch is 32 sequences, or **16,384 supervised tokens per optimizer step**.
- Each micro-batch loss is weighted by its share of the group’s sequences, so the update is the gradient of the mean loss over exactly the sequences in the group. The final partial group of an epoch is handled explicitly.
- The loop refuses to run a partial group unless `partial_final_update` is enabled.
- Training refuses to fall back to CPU; CUDA is required for non-FP32 precision.
- Non-finite loss or gradient norm raises immediately.
- The scheduler advances only after an optimizer step that actually updated parameters.

This is a careful implementation. The token-weighted accumulation and the partial-final-update handling are better than the usual “divide by accumulation steps” shortcut.

### 5.2 Optimizer and parameter groups

- AdamW with `beta1=0.9`, `beta2=0.999`, `eps=1e-8`.
- Learning rate 3e-4.
- Weight decay 0.1 on matrices; zero on biases, LayerNorm weights/biases, and all 1-D parameters.
- Tied parameters appear once in `named_parameters` and are not duplicated across groups.
- No fused or foreach optimizer is enabled; the project deliberately leaves optimizer defaults.

The decay policy is standard and sensible. One nuance: token embeddings are decayed because they are 2-D and not named as norm/bias. That is common, but some code models exclude embeddings from decay. It is not a bug; it is a choice to document.

### 5.3 Learning-rate schedule

- Linear warmup over `warmup_fraction = 0.02`.
- Cosine decay to `min_lr_ratio = 0.1` of the base rate.
- `total_steps = 8,129` in `configs/training.yaml`.
- Observed runs use **8,130** optimizer updates per stage. The one-step difference is a configuration/reporting inconsistency that should be reconciled before the next generation. It is not numerically important, but it undermines exact reproducibility claims.
- Warmup is 163 steps at 8,129 total steps.
- Continuation stages use `fresh_schedule: true`, rebuilding the schedule at the stage’s configured base rate while keeping the global step counter.

### 5.4 Gradient accumulation and clipping

- Accumulation 8 × micro-batch 4 = 32 sequences.
- Gradient clipping is global norm clipping at 1.0.
- Observed gradient norms in the original first pass: min 0.667, max 4.772, mean 0.968. In the final run segment: min 0.775, max 1.251, mean 0.850. No skipped steps.
- Clipping is applied after unscaling the FP16 scaler when FP16 is used. In BF16, no scaler is used.

### 5.5 Mixed precision

- `precision: auto` resolves to **BF16** on the RTX 5060 Laptop GPU because BF16 is supported.
- FP16 would use a `GradScaler`; BF16 does not.
- CPU evaluation is FP32; requesting BF16/FP16 on CPU raises.
- Master weights remain FP32; autocast handles compute precision.
- BF16 matmul and embedding-gradient atomics are not bitwise reproducible on CUDA. The determinism report says this explicitly.

Measured peak allocated VRAM for the 17M model was **1,150.93 MiB** in the first full pass. The pilot memory probe recorded 949.11 MiB peak at the same micro-batch/accumulation settings. These are measured, not estimates.

### 5.6 Checkpoint saving and resumption

- Training checkpoints are plain dictionaries of tensors and primitives, loaded with `weights_only=True`.
- Writes are atomic: temporary file plus `os.replace`.
- Checkpoints carry model, optimizer, scheduler, scaler, step/token counters, model and training config, tokenizer hash, data manifest hash, RNG states, and metrics history.
- Loader state is tracked so a resumed run replays the same permutation and position.
- `reset_loader_state` exists for continuation stages that switch to a fresh dataset.
- The release package uses a model-only state dict checkpoint, not the full training checkpoint, because the package’s inference interface does not need optimizer state.

The checkpoint design is strong. The one caveat is that the package’s selected checkpoint is a full training checkpoint copied as-is in the Phase 15 build (to preserve the expected hash); it therefore includes optimizer state and is larger than a pure inference checkpoint. That is a packaging trade-off, not a training defect.

### 5.7 Validation procedure

- A fixed validation subset of **64 blocks** (32,768 tokens) is reused at every evaluation.
- Validation runs every 100 steps in the first pass, plus a full validation at the end.
- Full validation is 7,756,800 tokens, 15,150 sequences, 1,894 batches.
- There is no early stopping. The best checkpoint is whichever validation loss is lowest; for Exp C, the best step was the final step 16,260.
- The fixed 64-block subset is only 0.42% of the validation stream. It is cheap but noisy; using it for checkpoint selection can overfit to 64 blocks.
- The full validation at the end is a better estimate, but it is not used for model selection during training.

### 5.8 Data loader behavior

- `TokenBlockDataset` memory-maps the `.bin` file; it never loads the full stream into RAM.
- `num_workers=0` by default. On Windows this avoids worker re-import overhead; memory-mapped reads in the main process are fast enough at this scale.
- `shuffle=True` for training, with a seeded `torch.Generator`.
- `drop_last=False`; the final partial batch is handled by the trainer.
- `pin_memory` defaults to true when CUDA is available.
- Blocks overlap by one token by design.

### 5.9 Determinism

- Python, NumPy, Torch CPU, and CUDA seeds are set.
- cuDNN deterministic is true; benchmark is false.
- `torch.use_deterministic_algorithms(True, warn_only=True)` is used because some CUDA ops (embedding backward atomics) have no deterministic kernel.
- `CUBLAS_WORKSPACE_CONFIG=:4096:8` and `PYTHONHASHSEED` are set.
- BF16/FP16 matmul and atomics remain non-bitwise-reproducible.

This is an honest determinism posture. Identical seeds produce statistically identical, not byte-identical, runs.

### 5.10 Reconstructed training exposure

| Run | Optimizer steps | Sequences | Tokens seen | Unique training tokens available | Notes |
|---|---:|---:|---:|---:|---|
| Original first pass | 8,130 | 260,142 | 133,192,704 | 133,192,957 | One pass over the original train split |
| Exp C stage 2 | 8,130 | 260,142 | 133,192,704 | 74,562,809 unique tokens in the stage | Re-weighting of the same corpus; JavaScript repeated 9.96× |
| Exp C total | 16,260 | 520,284 | 266,385,408 | 133,192,957 across both stages | No new unique records are introduced |

The distinction matters:

- **Total tokens seen** for Exp C is 266,385,408.
- **Unique dataset tokens** available across the whole training history is 133,192,957.
- **Stage tokens** are 133,192,704 for the original pass and 133,192,704 for the Exp C continuation.
- Exp C’s stage dataset has 65,172,938 unique Python tokens, 6,542,851 unique JavaScript tokens, and 2,847,020 documentation tokens. The JavaScript budget is filled by repeating the 6.54M unique tokens 9.96 times; the Python budget uses 66.1% of the available Python records.
- Because the curated JavaScript set is a subset of the original JavaScript training records, and the Python subset is a subset of the original Python records, **Exp C introduces no new unique tokens**. It is a re-weighting experiment, not a data-expansion experiment.

### 5.11 Is the schedule matched to capacity and data?

The model has 16.99M parameters and 133.19M unique training tokens, or **7.8 tokens per parameter**. A Chinchilla-style 20 tokens/parameter heuristic would suggest ~340M unique tokens for this model size. Exp C sees 266M tokens, but only 133M are unique. By that heuristic, the model is undertrained on unique data even before considering that code models often benefit from more than 20 tokens/parameter.

At the same time, the model is not obviously *capacity-starved* in the sense that a larger model would automatically fix the observed errors. The errors are dominated by syntax and semantic consistency, and the data is narrow. A larger model trained on the same 133M tokens and the same 6.5M unique JavaScript tokens would likely memorize more, not generalize better. The schedule is therefore reasonably matched to the current pilot, but **not matched to a serious next-generation model**. The next generation needs more unique, more diverse, and better-balanced data before or alongside more capacity.

---

## 6. Baseline performance diagnosis

This section uses only aggregate historical reports: Phase 10 generation summary, Phase 11 functional results, Phase 12 experiment audits, Phase 13 comparison, Phase 14 final audit, and Phase 15 packaging report. No individual `independent_v3` prompts, cases, or outputs were inspected.

### 6.1 Headline aggregate results

| Evaluation | Checkpoint | Syntax valid | EOS hit | Unique-token ratio | Repeated-line ratio | Complete tasks | Cases passed |
|---|---|---:|---:|---:|---:|---:|---:|
| Phase 10 development | Exp C, step 16,260 | 56/100 | 45/100 | 0.4623 | 0.1307 | — | — |
| Phase 11 development | Original, step 8,130 | Python 82%, JS 2% | — | — | — | 0/100 | 13/420 |
| Phase 13 development | Exp C, step 16,260 | 56/100 | 45/100 | 0.4623 | 0.1307 | 1/100 | 34/420 |
| Phase 14 independent | Exp C, step 16,260 | 63/100 | 59/100 | 0.5121 | 0.0995 | 4/100 | 39/328 |
| Phase 14 independent | Exp B, step 16,260 | 57/100 | 50/100 | 0.4708 | 0.1144 | 2/100 | 20/328 |

Phase 14’s independent set has 328 cases (164 per language), while the development set has 420 (210 per language). The numbers are therefore not directly comparable across phases. The consistent signal is that functional success is very low in both settings.

### 6.2 Syntax generation

- Exp C produces 63/100 syntax-valid independent generations: 40/50 Python and 23/50 JavaScript.
- JavaScript syntax is the dominant failure: in Phase 13, JavaScript syntax failures were 35–49 of 50 tasks across checkpoints; Exp C was best at 35.
- Python syntax is much stronger, but still imperfect: 9–15 of 50 tasks fail syntax in the development reports, depending on checkpoint.
- The original model’s JavaScript syntax validity was 1/50; training on JavaScript improved this, but not to a useful level.

**Diagnosis:** the model has learned Python syntax well enough to produce parseable short functions most of the time. It has not learned JavaScript syntax reliably. The gap is consistent with the 3.13:1 Python:JavaScript token ratio and the much smaller unique JavaScript corpus.

### 6.3 Identifier consistency

Observed error patterns in the aggregate reports:

- `UnboundLocalError` where a local variable is referenced before assignment.
- `AttributeError` for invented methods/attributes such as `list.is_empty`, `int.get_gcd`, `int.exp`, `list.is_all`, and `list.get`.
- `NameError` for undefined names such as `string_types`.
- `TypeError` from calling methods with the wrong arity or on the wrong type.
- Declared function names disagreeing with the corpus name in the JavaScript audit.

**Diagnosis:** the model does not maintain an identifier environment. It reproduces plausible-looking attribute and method names from training code without checking whether they exist on the receiver. The serialization format gives it the signature stub, but there is no objective that rewards copying the signature’s identifiers or tracking local bindings. This is a core weakness, not a cosmetic one.

### 6.4 Function signature completion

- The benchmark supplies a code stub, and the model continues from it.
- Many failures are not “cannot continue the signature” but “continues with a body that ignores the required return contract.”
- Examples: factorial returns `n` instead of the product; `is_even` returns `None`; `sum_list` returns the input list; `bubble_sort` returns the bound method `list.sort`; `my_len` returns the string.
- JavaScript failures are often earlier: the continuation does not parse at all.

**Diagnosis:** signature continuation is partially learned. The model can often produce a syntactically plausible body, but it does not reliably implement the described postcondition.

### 6.5 Control flow

Aggregate failure patterns include off-by-one loops, missing accumulators, wrong branch conditions, and returns inside the wrong scope. The benchmark’s simple tasks (factorial, fibonacci, binary search, sorting) expose these because a single control-flow mistake fails all cases.

**Diagnosis:** the model has local fluency but weak global control-flow planning. It is better at emitting idiomatic-looking lines than at composing a correct loop/recursion for a stated goal.

### 6.6 Arithmetic and algorithmic reasoning

- Simple arithmetic tasks fail frequently: factorial, fibonacci, nth prime, gcd, power, modulo, absolute value, temperature conversion.
- The failures are not primarily “the model cannot do arithmetic tokens”; they are “the model writes a function that does not implement the operation.”
- There is no evidence of multi-step reasoning or algorithm selection. The model appears to pattern-match on function names and docstrings.

**Diagnosis:** algorithmic reasoning is the weakest capability. The corpus is real-world library code, not a curated algorithm corpus, and the training objective is next-token prediction without execution feedback.

### 6.7 Data structures

- List operations (flatten, intersection, union, deduplication, filtering) fail often.
- Some failures are type errors (`'list' object has no attribute ...`), some are wrong return shapes, some are incorrect element-wise logic.
- JavaScript array tasks fail mostly at syntax before semantics.

**Diagnosis:** the model has seen many list operations, but it has not learned a reusable data-structure abstraction. It mimics common patterns without tracking element types or return shapes.

### 6.8 Return values

A recurring pattern is a function that computes or prints something but returns the wrong object:

- returns an input instead of a derived value,
- returns `None`,
- returns a method object,
- returns a string where a boolean/number is expected,
- returns a list where a scalar is expected.

**Diagnosis:** return-contract adherence is weak. The training objective does not distinguish “the function returns the right type/value” from “the next token is likely.” Execution-based evaluation exposes this immediately.

### 6.9 Python versus JavaScript competence

| Dimension | Python | JavaScript |
|---|---|---|
| Syntax validity (Exp C, independent) | 40/50 | 23/50 |
| Complete tasks (Exp C, independent) | 3/50 | 1/50 |
| Cases passed (Exp C, independent) | 27/164 | 12/164 |
| Dominant failure | Runtime errors and incorrect outputs | Syntax failures |
| Unique training tokens | 98.8M | 31.5M original; 6.5M curated |
| Curated data | None | 35,658 records, repeated 9.96× in Exp C |

**Diagnosis:** Python is the stronger language by every measure, but “stronger” is relative: 3/50 complete tasks is still weak. JavaScript is data-starved. Exp C’s JavaScript improvement over Exp B is real but likely partly memorization of the 35,658 curated functions.

### 6.10 Generation termination and repetition

- Exp C independent: 59/100 EOS hits, 41/100 generations that did not terminate within 64 tokens.
- Development Exp C: 45/100 EOS hits, 55/100 max-token truncations.
- Repetition is moderate, not catastrophic: mean repeated-line ratio 0.0995–0.1307, repeated 4-gram ratio 0.1107–0.1108.
- Unique-token ratio 0.4623–0.5121, which is low but not degenerate.
- No empty generations were recorded in the Phase 10 report.

**Diagnosis:** termination is unreliable. The model often continues past the natural end of a function, or fails to emit EOS because it has not learned a strong end-of-function signal for the prompt distribution. Repetition is present but secondary to syntax and semantic errors.

### 6.11 What the aggregate pattern means

The model has learned:

- Python lexical and syntactic form to a useful degree.
- Short function-shaped output.
- Common identifier and API-looking fragments.
- Some docstring-to-code association.

It has not learned:

- Reliable JavaScript syntax.
- Identifier/type consistency.
- Return-contract adherence.
- Control-flow planning for algorithms.
- When to stop.
- Generalization from library code to standalone tasks.

This is exactly the profile of a small next-token model trained on a narrow function-level corpus without execution feedback or instruction tuning. More parameters alone would not change the profile; better data, better objectives, and better evaluation would.

---

## 7. Scaling proposals

All proposals are **from-scratch** models for their generation. None is a fine-tune of Exp C. Parameter counts below are analytical and exact for the stated configuration; VRAM and time are **estimates**, clearly labeled, and must be profiled before commitment.

### 7.1 Summary table

| | A — Conservative | B — Intermediate | C — Ambitious |
|---|---|---|---|
| Parameters | 33,674,240 | 65,251,840 | 110,418,432 |
| d_model | 512 | 640 | 768 |
| Layers | 8 | 10 | 12 |
| Heads | 8 | 10 | 12 |
| Head dim | 64 | 64 | 64 |
| FFN dim | 2,048 | 2,560 | 3,072 |
| Vocabulary | 16,000 | 24,000 | 32,000 |
| Context length | 512 | 1,024 | 1,024 (2,048 optional) |
| Approx. VRAM (training) | 2–3 GB | 4–6 GB | 5–7 GB |
| Unique training tokens | 150–250M | 300–600M | 800M–1.5B |
| Est. time on RTX 5060 Laptop | 1.5–3.5 h | 6–17 h | 30–80 h |
| Main constraint | Data quality | JavaScript data + VRAM | Data, time, thermals, risk |

### 7.2 Proposal A — Conservative improvement

**Configuration**

- 8 layers, 512 model dimension, 8 heads, 2,048 FFN, 16k vocabulary, 512 context.
- 33.67M parameters, about 2.0× the baseline.
- Keep tied embeddings and pre-LN.
- Add RoPE and SDPA if possible without changing the data pipeline; otherwise keep learned positions for a strict control.

**Data**

- 150–250M unique tokens, prioritizing:
  - a larger, deduplicated Python function corpus;
  - the existing curated JavaScript plus additional permissive JavaScript;
  - a small increase in Python documentation;
  - a construct-balanced sample (recursion, loops, lists/dicts, strings, arithmetic).
- Keep the 512 context for this proposal; it is a capacity/data experiment, not a long-context experiment.

**Training**

- Effective batch 32–64 sequences.
- 1–1.5 passes over the unique corpus.
- BF16, gradient accumulation, gradient clipping 1.0.
- Same warmup/cosine recipe, retuned total steps.

**Estimated resources**

- VRAM: 2–3 GB with micro-batch 8–16 and accumulation 4–8, no gradient checkpointing. This is an estimate based on the measured 1.12 GiB baseline and a ~2× parameter/activation increase.
- Throughput: roughly 20,000–30,000 tokens/s, down from the measured 44,268 tokens/s for the 17M model. Estimate, not measured.
- Time: 150M tokens at 25k tokens/s ≈ 1.7 h; 250M at 20k ≈ 3.5 h. Range 1.5–3.5 h.

**Risks and trade-offs**

- Lowest risk, fastest iteration.
- Still limited by 512 context and function-level data.
- May improve syntax and short-function quality without improving algorithmic reasoning.
- Best used as a controlled ablation, not as the final generation.

### 7.3 Proposal B — Intermediate model

**Configuration**

- 10 layers, 640 model dimension, 10 heads, 2,560 FFN, 24k vocabulary, 1,024 context.
- 65.25M parameters, about 3.8× the baseline.
- RoPE, RMSNorm, SwiGLU, SDPA, QK norm if stable; tied embeddings optional (untied adds 15.36M parameters, which is likely not worth it at this scale).
- Gradient checkpointing available.

**Data**

- 300–600M unique tokens.
- This requires a material JavaScript expansion: the current curated set is only 6.54M tokens. A 1:1 code budget at 300M tokens needs ~150M JavaScript tokens, or ~23× the current curated corpus. That is not available without new sources and licensing review.
- Python can scale from 98.8M to ~150–200M with additional permissive repositories, but deduplication and quality filtering are mandatory.
- Add JavaScript documentation/examples if licensing permits; the current training has none.

**Training**

- Effective batch 32–64 sequences at 1,024 context; micro-batch 8–16, accumulation 4–8.
- 1–1.5 passes over unique tokens.
- BF16; gradient checkpointing if VRAM exceeds ~6 GB.
- Longer warmup, cosine to 10%.

**Estimated resources**

- VRAM: 4–6 GB with micro-batch 8–16 and gradient checkpointing. Estimate only. At 1,024 context, attention score memory alone is 4× the 512-token case for the same batch; without checkpointing, 8 GB is tight.
- Throughput: roughly 9,000–14,000 tokens/s, extrapolated from the baseline. Estimate.
- Time: 300M at 12k ≈ 6.9 h; 600M at 10k ≈ 16.7 h; at the slow end (9k) 600M ≈ 18.5 h. Range 6–18 h. Add evaluation and checkpoint overhead.

**Risks and trade-offs**

- Best quality-per-risk if the data expansion succeeds.
- JavaScript data collection and licensing are the critical path, not the model code.
- 8 GB VRAM is workable only with careful microbatch and checkpointing; must be profiled before a full run.
- A tokenizer change to 24k invalidates v0.1.0 compatibility; that is acceptable for a new generation but must be versioned.

### 7.4 Proposal C — Ambitious model

**Configuration**

- 12 layers, 768 model dimension, 12 heads, 3,072 FFN, 32k vocabulary, 1,024 context (2,048 optional).
- 110.42M parameters, about 6.5× the baseline.
- RoPE, RMSNorm, SwiGLU, SDPA/FlashAttention, GQA (e.g., 12 query heads / 4 KV heads), gradient checkpointing.
- Untied embeddings would add 24.58M parameters; likely keep tying at this scale.

**Data**

- 800M–1.5B unique tokens.
- This is not achievable with the current corpus. It requires:
  - a substantially larger permissive Python corpus;
  - a substantially larger permissive JavaScript corpus;
  - deduplication and quality filtering at scale;
  - construct/algorithm balancing;
  - a new tokenizer and a new frozen evaluation suite.
- The current 133M unique tokens would be repeated 6–11×, which is not a healthy substitute for diversity.

**Training**

- Effective batch 32–64 sequences at 1,024 context; micro-batch 4–8, accumulation 8–16.
- Gradient checkpointing every layer.
- BF16; possibly FP16 with scaler if BF16 is unstable.
- 1–1.5 passes over unique tokens.
- Long run with frequent checkpointing and resume tests.

**Estimated resources**

- VRAM: 5–7 GB with micro-batch 4–8, gradient checkpointing, and SDPA. Estimate only. At 2,048 context, attention memory grows further and 8 GB may be insufficient even with checkpointing.
- Throughput: roughly 5,000–8,000 tokens/s, extrapolated. Estimate.
- Time: 800M at 7k ≈ 31.7 h; 1.5B at 7k ≈ 59.5 h. At the slow end (5k), 1.5B ≈ 83 h. Range 30–80 h, before evaluation and restarts.
- On a laptop GPU, sustained multi-day training is a thermal and reliability risk. Checkpoint frequently and validate resume behavior.

**Risks and trade-offs**

- Highest potential quality, highest risk.
- Data collection and licensing are the binding constraints.
- 8 GB VRAM is at or beyond the comfortable limit; “it might fit” is not a plan. A measured memory profile at the exact configuration is required before committing.
- Long runs increase the chance of thermal throttling, driver issues, and interrupted checkpoints.
- Not recommended as the first move. It should follow a successful B generation and a demonstrated data pipeline.

### 7.5 Which proposal should be chosen?

**Recommendation: B, conditional on data improvements; A as the fallback; C only after B.**

Reasoning:

- A is safe but does not address the core data and context limitations. It is a good controlled step, not a generation-defining model.
- B is the smallest proposal that meaningfully changes capacity, context, and data coverage. It is still feasible on 8 GB with gradient checkpointing and careful microbatching, and it can finish in hours rather than days.
- C is attractive on paper but is not justified by the current data. Training a 110M model on 133M unique tokens repeated many times would likely produce a more fluent memorizer, not a better programmer. It also pushes the laptop GPU into a regime where fit and sustained throughput are uncertain.

The decision should be gated: **do not start B until the new data manifest, tokenizer decision, and new frozen benchmark are ready.** Do not start C until B has a complete evaluation and the data corpus is at least 800M unique tokens.

---

## 8. Hardware feasibility

### 8.1 Measured baseline

| Metric | Measured value |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB class |
| PyTorch | 2.15.0+cu132 |
| CUDA | 13.2 |
| Precision | BF16 |
| Peak allocated VRAM (17M model) | 1,150.93 MiB |
| Effective batch | 32 sequences |
| Tokens per optimizer step | 16,384 |
| Stage 2 throughput | 44,268 tokens/s |
| One 133.19M-token pass | 3,008.8 s (≈50.1 min) |
| Full validation | 355.7 s for 7.76M tokens |

This is a real, useful baseline: the current model is comfortable on the GPU and fast enough to iterate.

### 8.2 Estimated scaling

The estimates below assume roughly linear parameter/activation growth, BF16, no gradient checkpointing unless stated, and the same effective batch. They are **estimates**, not measurements. They should be replaced by a short profiling run before any full training commitment.

| Model | Params | Context | Est. peak VRAM | Est. throughput | Est. time for 1 pass |
|---|---:|---:|---:|---:|---:|
| Baseline | 17M | 512 | 1.12 GiB (measured) | 44.3k tok/s (measured) | 50 min for 133M |
| A | 33.7M | 512 | 2–3 GB | 20–30k tok/s | 1.5–3.5 h for 150–250M |
| B | 65.3M | 1,024 | 4–6 GB | 9–14k tok/s | 6–18 h for 300–600M |
| C | 110.4M | 1,024 | 5–7 GB | 5–8k tok/s | 30–80 h for 800M–1.5B |

### 8.3 Memory-saving strategies

- **BF16 autocast** is already used and should remain the default.
- **Gradient checkpointing** is the single most important addition for B/C. It trades ~20–30% throughput for a large activation-memory reduction.
- **Micro-batch reduction with increased accumulation** keeps the effective batch constant while fitting memory.
- **SDPA/FlashAttention** avoids materializing the full `[B, heads, T, T]` score tensor.
- **GQA** reduces KV cache and attention memory, especially for long context.
- **Sequence packing with correct block-diagonal masking** can improve throughput, but it complicates the data pipeline; do not add it without tests.
- **Optimizer state offload** is possible but adds complexity and host-device traffic; not recommended on a laptop unless necessary.
- **Fused AdamW** may help throughput but must be verified against the existing optimizer semantics; the project currently avoids unverified optimizer optimizations.

### 8.4 Hard constraints and honest limits

- 8 GB VRAM is enough for B only with profiling and checkpointing. It is not guaranteed for C.
- Laptop GPUs throttle under sustained load. A 30–80 hour C run is a reliability experiment as much as a modelling experiment.
- Disk space for checkpoints grows with parameter count: a 65M model’s full training checkpoint is roughly 0.8 GB (model + AdamW moments), and a 110M model’s is roughly 1.3 GB. Multiple checkpoints add up quickly.
- The current code has no gradient checkpointing flag; adding it is a Phase 18 engineering task, not a configuration change.

---

## 9. Proposed Phase 17–22 plan

All future models must be trained **from scratch** for their respective generation. Do not describe fine-tuning Exp C as a new from-scratch model. Preserve v0.1.0 as the immutable baseline.

### Phase 17 — Data foundation v2

**Goal:** build a larger, cleaner, more balanced, construct-aware corpus before changing the model.

- Inventory new permissive Python and JavaScript sources; complete license review before download or use.
- Add JavaScript documentation/examples only if the license permits; the current corpus has none.
- Expand curated JavaScript beyond 35,658 records. Target at least 20–30M unique JavaScript tokens for a B-scale 1:1 budget; more for C.
- Add a Python near-duplicate pass; the current corpus has exact-dedup but no MinHash-style near-dedup.
- Build an AST-level construct census: functions, classes, methods, recursion, loops, comprehensions, exception handling, async, imports, decorators, data structures, arithmetic.
- Measure docstring quality and description/code alignment; down-weight or remove auto-generated and boilerplate docstrings.
- Decide tokenizer v2: keep 16k or move to 24k/32k. If changed, retrain only on the new training partition and freeze a new tokenizer hash.
- Freeze new train/validation manifests and hashes. Do not touch the old test partition.
- Do not use `functional/independent_v3/` for any training or tuning decision.

**Exit criteria:** new manifests, construct census, dedup report, license review, and tokenizer decision are complete and frozen.

### Phase 18 — Architecture v2 implementation

**Goal:** implement the next architecture without training a final model yet.

- Replace learned absolute positions with RoPE.
- Replace LayerNorm with RMSNorm; add QK norm if depth > 10.
- Replace GELU FFN with SwiGLU.
- Replace manual attention with SDPA/FlashAttention.
- Add optional GQA.
- Add gradient checkpointing and a measured memory profiler.
- Add a clean from-scratch config for A/B/C sizes.
- Add unit tests for shape, masking, RoPE extrapolation, checkpoint round-trip, and deterministic forward passes.
- Keep the old model code path available for v0.1.0 compatibility.

**Exit criteria:** unit tests pass; a short profiling run measures VRAM and throughput for A and B at the target microbatch.

### Phase 19 — Controlled small-scale ablations

**Goal:** choose data mix and architecture details before a full B run.

- Train A-size models (or smaller proxies) from scratch for short runs (e.g., 50–100M tokens).
- Ablate: Python/JavaScript ratio, curated vs original JavaScript, docs ratio, RoPE vs learned positions, RMSNorm vs LayerNorm, SwiGLU vs GELU, context 512 vs 1,024.
- Evaluate on validation loss and a new development benchmark, not `independent_v3`.
- Record throughput, VRAM, gradient norms, and failure categories.
- Keep all runs versioned with config, data manifest, tokenizer hash, and checkpoint hash.

**Exit criteria:** a documented data mix and architecture choice for B, with measured evidence.

### Phase 20 — Intermediate model B from scratch

**Goal:** train the 65M-parameter B model on the new corpus.

- 10 layers, 640 d_model, 10 heads, 2,560 FFN, 24k vocab, 1,024 context.
- 300–600M unique tokens, 1–1.5 passes.
- BF16, gradient checkpointing, effective batch 32–64.
- Full validation at intervals, not only at the end.
- Checkpoint every N steps; test resume at least twice.
- No fine-tuning of Exp C; initialize from random.

**Exit criteria:** completed run, final checkpoint, validation report, and reproducible config.

### Phase 21 — Evaluation and failure analysis

**Goal:** evaluate B honestly on a new frozen benchmark.

- New benchmark, frozen before training; not `independent_v3`.
- 100+ tasks per language, balanced difficulty, construct coverage, hidden cases.
- Execution in the existing Docker sandbox; no network; same resource limits.
- Metrics: syntax validity, complete-task success, case pass rate, runtime-error rate, incorrect-output rate, timeout/resource rate, EOS completion, repetition, unique-token ratio, Python/JavaScript parity.
- Failure taxonomy: syntax, identifier/type, signature, control flow, arithmetic, data structure, return value, termination.
- Contamination check against training data; lexical and near-duplicate checks.
- Compare against v0.1.0 on the same new benchmark, not on the old one.

**Exit criteria:** full report with per-language and per-category results, and a clear statement of what improved and what did not.

### Phase 22 — Release candidate and final audit

**Goal:** package the selected model and preserve the baseline.

- Select the model on the new benchmark, not on `independent_v3`.
- Build an inference-only package with model state, tokenizer, config, metadata, hashes, and minimal source.
- Validate CPU and CUDA load, smoke generation, greedy reproducibility, and package integrity.
- Freeze v0.1.0 artifacts and hashes; do not overwrite them.
- Version the new package (e.g., v0.2.0) with a new manifest.
- Write a final audit that distinguishes measured results from estimates and lists all limitations.

**Exit criteria:** verified package, immutable v0.1.0 baseline, and a final report with an explicit verdict.

### Cross-cutting controls

- **One variable at a time** for architecture/data ablations.
- **Frozen evaluation** before training; no tuning on the final benchmark.
- **No `independent_v3` use** for development; it remains a historical artifact.
- **No old test partition access** for training or tuning.
- **Version everything**: config, tokenizer, data manifest, checkpoint, evaluation report, package.
- **Hash everything**: training inputs, tokenizer, checkpoints, package files.
- **Report failures**, not just successes; a skipped or crashed run is data.
- **Do not claim release readiness** from a small benchmark.

---

## 10. Known uncertainties

1. **Construct coverage is unmeasured.** There is no AST census for Python, and the JavaScript census is function-declaration-heavy. The true distribution of algorithms, classes, async code, and error handling is unknown.
2. **Python near-duplication is unmeasured.** Exact duplicates are controlled; MinHash-style near-duplicates were only applied to curated JavaScript.
3. **Contamination with the independent benchmark is not proven absent.** The Phase 14 audit called the contamination assessment preliminary lexical evidence, not proof of semantic independence.
4. **Exp C’s JavaScript gains may be partly memorization.** The curated set is 6.54M unique tokens repeated 9.96×; the evaluation cannot distinguish generalization from recall.
5. **VRAM and throughput estimates are extrapolations.** Only the 17M baseline was measured. A/B/C estimates must be replaced by profiling.
6. **Tokenizer adequacy for a larger corpus is unknown.** The 16k vocabulary may be too small for 24k–32k-scale models; a tokenizer change invalidates v0.1.0 compatibility.
7. **Licensing for new sources is unresolved.** The current corpus is limited to permissive CodeSearchNet repositories and PSF documentation. Expanding JavaScript/Python requires review.
8. **The old test partition was not accessed**, so there is no independent estimate of generalization beyond the consumed `independent_v3` set.
9. **Training logs are summaries, not a full event log.** Exposure was reconstructed from manifests and run summaries; minor discrepancies (e.g., 8,129 vs 8,130 steps) remain.
10. **BF16 training is not bitwise reproducible.** Identical seeds produce statistically identical, not byte-identical, runs.

---

## Final status

The baseline is technically sound but empirically weak. The dominant bottlenecks are data diversity, JavaScript data scarcity, function-level versus task-level distribution mismatch, and the absence of execution-aware or consistency-aware objectives. The architecture should evolve (RoPE, RMSNorm, SwiGLU, SDPA, longer context, gradient checkpointing), but architecture alone is not the fix. The next generation should be data-first, evaluated on a new frozen benchmark, and trained from scratch.

PHASE 16 DIAGNOSIS COMPLETE — READY FOR PHASE 17
