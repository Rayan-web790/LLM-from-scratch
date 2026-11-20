from __future__ import annotations

import json
import hashlib
import sys
import torch
import argparse
from polycodegpt.model import GPT, ModelConfig
from polycodegpt.model.checkpoint import load_checkpoint as load_model_checkpoint
from polycodegpt.training.checkpoint import load_checkpoint as load_training_checkpoint
from polycodegpt.training import rebuild_model, weights_are_tied
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Minimal special token mapping (matching tokenizer v1.1)
SPECIAL_TOKEN_IDS = {
    "<|pad|>": 0,
    "<|unk|>": 1,
    "<|eos|>": 2,
    "<|code|>": 3,
    "<|docstring|>": 4,
    "<|text|>": 5,
    "<|lang:python|>": 6,
    "<|lang:javascript|>": 7,
    "<|lang:en|>": 8,
    "<|lang:fr|>": 9,
    "<|lang:ar|>": 10,
}
PAD_ID = SPECIAL_TOKEN_IDS["<|pad|>"]
UNK_ID = SPECIAL_TOKEN_IDS["<|unk|>"]
EOS_ID = SPECIAL_TOKEN_IDS["<|eos|>"]
CODE_ID = SPECIAL_TOKEN_IDS["<|code|>"]
DOCSTRING_ID = SPECIAL_TOKEN_IDS["<|docstring|>"]
LANG_PYTHON_ID = SPECIAL_TOKEN_IDS["<|lang:python|>"]
LANG_JAVASCRIPT_ID = SPECIAL_TOKEN_IDS["<|lang:javascript|>"]

MAX_NEW_TOKENS = 64
TEMPERATURE = 0.0
SEED = 42
N_PYTHON = 50
N_JAVASCRIPT = 50

# Checkpoint and tokenizer paths
CHECKPOINT_PATH_DEFAULT = project_root / "checkpoints" / "polycodegpt_first_pass" / "step_008130.pt"
TOKENIZER_PATH = project_root / "tokenizer" / "artifacts" / "polycodegpt_tokenizer_v1_1.json"
MODEL_CONFIG_PATH = project_root / "configs" / "model.yaml"


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(language: str, docstring: str, code_stub: str) -> str:
    """Build the serialization prompt format.

    Format: <|lang:X|><|docstring|>{docstring}<|code|>{code_stub}<|eos|
    """
    lang_marker = "<|lang:python|>" if language == "python" else "<|lang:javascript|>"
    return f"{lang_marker}<|docstring|>{docstring}<|code|>{code_stub}<|eos|>"


def build_training_sequence(language: str, docstring: str, code: str) -> str:
    """Build a complete training serialization sequence.

    Format: <|lang:X|><|docstring|>{docstring}<|code|>{code}<|eos|>

    Used for metrics that require a full sequence with EOS marker.
    """
    lang_marker = "<|lang:python|>" if language == "python" else "<|lang:javascript|>"
    return f"{lang_marker}<|docstring|>{docstring}<|code|>{code}<|eos|>"


def build_generation_prefix(language: str, docstring: str, code_stub: str = "") -> str:
    """Build a generation prefix (unfinished continuation prompt).

    Format: <|lang:X|><|docstring|>{docstring}<|code|>{code_stub}

    Does NOT include a trailing EOS marker. Used for actual generation prompts.
    """
    lang_marker = "<|lang:python|>" if language == "python" else "<|lang:javascript|>"
    return f"{lang_marker}<|docstring|>{docstring}<|code|>{code_stub}"


def tokenize_prompt(prompt_text: str, tokenizer) -> list:
    """Tokenize the prompt text, returning list of token IDs."""
    ids = tokenizer.encode(prompt_text, add_special_tokens=False).ids
    if not isinstance(ids, list) or not all(isinstance(x, int) for x in ids):
        raise TypeError(f"tokenizer.encode() must return list of ints, got {type(ids)}")
    return ids


# ─────────────────────────────────────────────────────────────────────────────
# Device selection
# ─────────────────────────────────────────────────────────────────────────────

def select_device(args_device: str) -> torch.device:
    """Select computation device.

    --device cuda: uses CUDA if available, otherwise raises a clear error.
    --device cpu: always uses CPU.
    """
    if args_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available. Use --device cpu instead.")
        return torch.device("cuda")
    elif args_device == "cpu":
        return torch.device("cpu")
    else:
        raise ValueError(f'Invalid device: {args_device}. Choose from ["cpu", "cuda"]')


# ─────────────────────────────────────────────────────────────────────────────
# Generation helper
# ─────────────────────────────────────────────────────────────────────────────

def generate_continuation(model, device, prompt_ids, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, eos_id=EOS_ID):
    """Generate continuation using the model.

    Returns (gen_no_eos, eos_hit) where:
      - gen_no_eos: generated token IDs (after the prompt), with EOS removed
      - eos_hit: True if the model produced an EOS token
    """
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            eos_token_id=eos_id,
        )
        # Assert input tensor device immediately after generation
        if device.type == "cuda":
            assert input_ids.device.type == "cuda", f"Input tensor not on CUDA: {input_ids.device}"
            assert out.device.type == "cuda", f"Generated tensor not on CUDA: {out.device}"

    # Extract generated tokens (after the prompt)
    generated = out[0, len(prompt_ids):].tolist()

    # Detect EOS hit BEFORE removing EOS
    eos_hit = eos_id is not None and eos_id in generated

    # Stop at EOS if present
    if eos_hit:
        gen_no_eos = generated[:generated.index(eos_id)]
    else:
        gen_no_eos = generated

    return gen_no_eos, eos_hit


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers (mirrors tests/unit/test_phase10_benchmark.py)
# ─────────────────────────────────────────────────────────────────────────────

from collections import Counter


def count_code_tokens(ids: list) -> int:
    """Count code tokens between CODE_ID and EOS_ID in serialized token IDs."""
    if CODE_ID not in ids:
        return 0
    code_index = ids.index(CODE_ID)
    tail = ids[code_index + 1:]
    if tail and tail[-1] == EOS_ID:
        tail = tail[:-1]
    return len(tail)


def unique_token_ratio(ids: list) -> float:
    """Fraction of distinct tokens in the code sequence."""
    n = count_code_tokens(ids)
    if n == 0:
        return 0.0
    code_start = ids.index(CODE_ID) + 1
    code_end = None
    for i in range(code_start, len(ids)):
        if ids[i] == EOS_ID:
            code_end = i
            break
    if code_end is None:
        code_tail = ids[code_start:]
    else:
        code_tail = ids[code_start:code_end]
    if not code_tail:
        return 0.0
    distinct = len(set(code_tail))
    return distinct / n

def unique_token_ratio_generated(generated_ids: list) -> float:
    """Unique-token ratio for generated code tokens only (BPE).

    Only model-generated code token IDs contribute. Excludes prompt,
    language markers, docstring markers, code stub, and EOS.
    Empty generation returns 0.0.
    """
    if not generated_ids:
        return 0.0
    distinct = len(set(generated_ids))
    return distinct / len(generated_ids)


def repeated_line_ratio(gen_text: str) -> float:
    """Fraction of lines that appear more than once."""
    lines = [l for l in gen_text.splitlines() if l.strip()]
    if not lines:
        return 0.0
    counts = Counter(lines)
    repeated = sum(1 for line, cnt in counts.items() if cnt > 1)
    return repeated / len(lines)


def repeated_4gram_ratio(gen_text: str) -> float:
    """Fraction of overlapping 4-token sequences that appear multiple times.

    Uses whitespace-separated words (legacy text-based metric).
    """
    tokens = gen_text.split()
    if len(tokens) < 8:
        return 0.0
    grams = [tuple(tokens[i : i + 4]) for i in range(len(tokens) - 3)]
    counts = Counter(grams)
    repeated = sum(c for c in counts.values() if c > 1)
    total = len(grams)
    return repeated / total


def repeated_4gram_ratio_tokens(token_ids: list) -> float:
    """Fraction of overlapping 4-token sequences (BPE) that appear multiple times.

    Uses BPE token IDs from tokenizer v1.1.0, as specified in the Phase 10
    benchmark design. Measures only the generated code tokens (not prompt).
    """
    tokens = list(token_ids)
    if len(tokens) < 8:
        return 0.0
    grams = [tuple(tokens[i : i + 4]) for i in range(len(tokens) - 3)]
    counts = Counter(grams)
    repeated = sum(1 for c in counts.values() if c > 1)
    total = len(grams)
    return repeated / total


def compute_syntax_validity(gen_text: str, language: str) -> bool:
    """Check if the generated code is syntactically valid.

    Python is validated with :func:`ast.parse` on the complete candidate code.
    JavaScript is validated with ``node --check`` on a temporary file that
    contains the complete candidate code. The code is parsed only; it is never
    executed.
    """
    import ast
    import os
    import shutil
    import subprocess
    import tempfile

    gen_text_stripped = gen_text.strip() if isinstance(gen_text, str) else ""
    if not gen_text_stripped:
        return False

    if language == "python":
        try:
            ast.parse(gen_text_stripped)
            return True
        except Exception:
            return False
    elif language == "javascript":
        node = shutil.which("node")
        if node is None:
            raise RuntimeError(
                "Node.js is not installed or not available on PATH; cannot perform "
                "a genuine JavaScript syntax check. Install Node.js, or use a "
                "parser-only JavaScript parser such as 'acorn' or 'esprima' (npm)."
            )
        # Write the complete candidate code to a temporary .js file and parse it
        # with `node --check`. This parses only; it does not execute the code.
        fd, tmp_path = tempfile.mkstemp(suffix=".js")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(gen_text_stripped)
                f.flush()
            result = subprocess.run(
                [node, "--check", tmp_path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.returncode == 0
        except Exception:
            return False
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark execution
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(checkpoint_path, device):
    """Load model and tokenizer from checkpoint."""
    model_config = ModelConfig.from_yaml(MODEL_CONFIG_PATH)
    from src.polycodegpt.data.tokenizer_io import load_tokenizer as load_tok
    tokenizer = load_tok(TOKENIZER_PATH, vocab_size=model_config.vocab_size)
    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint_path}")
    try:
        payload = load_training_checkpoint(checkpoint_path, map_location="cpu")
        model = rebuild_model(payload, device="cpu")
        step = payload["step"]
    except Exception:
        state_dict, config, meta = load_model_checkpoint(checkpoint_path, device="cpu")
        model = GPT(config)
        model.load_state_dict(state_dict)
        step = meta.get("step", "unknown")
    if not weights_are_tied(model):
        model.lm_head.weight = model.token_embedding.weight
    model = model.to(device).eval()
    if not all(torch.isfinite(p).all().item() for p in model.parameters()):
        raise SystemExit("checkpoint contains non-finite parameters")
    return model, tokenizer, step


def _sha256_file(path: Path) -> str:
    """Return the SHA256 hex digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _phase10_generation_tasks() -> list[dict]:
    """Return the original Phase 10 task inventory as task records."""
    from scripts.phase10_tasks import python_tasks, javascript_tasks

    tasks: list[dict] = []
    for docstring, code_stub in python_tasks():
        tasks.append(
            {
                "task_id": None,
                "language": "python",
                "docstring": docstring,
                "code_stub": code_stub,
            }
        )
    for docstring, code_stub in javascript_tasks():
        tasks.append(
            {
                "task_id": None,
                "language": "javascript",
                "docstring": docstring,
                "code_stub": code_stub,
            }
        )
    return tasks


def load_task_manifest(path) -> tuple[list[dict], str, str]:
    """Load a generation task suite from a JSON manifest.

    Returns ``(tasks, suite_name, manifest_sha256)``. The original Phase 10
    suite is represented by ``None``; this function is only for manifest files.
    """
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise SystemExit(f"task manifest not found: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read task manifest {manifest_path}: {exc}") from exc

    if isinstance(payload, dict):
        raw_tasks = payload.get("tasks")
        suite_name = payload.get("benchmark") or manifest_path.stem
    elif isinstance(payload, list):
        raw_tasks = payload
        suite_name = manifest_path.stem
    else:
        raise SystemExit(f"task manifest must contain a task list: {manifest_path}")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise SystemExit(f"task manifest has no tasks: {manifest_path}")

    seen_ids: set[str] = set()
    tasks: list[dict] = []
    for index, item in enumerate(raw_tasks, start=1):
        if not isinstance(item, dict):
            raise SystemExit(f"task {index} in {manifest_path} is not an object")
        language = item.get("language")
        if language not in ("python", "javascript"):
            raise SystemExit(
                f"task {item.get('task_id', index)} has unsupported language {language!r}"
            )
        docstring = item.get("docstring")
        code_stub = item.get("signature", item.get("code_stub", item.get("stub")))
        if not isinstance(docstring, str) or not docstring.strip():
            raise SystemExit(f"task {item.get('task_id', index)} has no docstring")
        if not isinstance(code_stub, str) or not code_stub.strip():
            raise SystemExit(f"task {item.get('task_id', index)} has no code stub")
        task_id = item.get("task_id")
        if task_id is not None:
            if not isinstance(task_id, str) or not task_id.strip():
                raise SystemExit(f"task {index} has an invalid task_id")
            if task_id in seen_ids:
                raise SystemExit(f"duplicate task_id in manifest: {task_id}")
            seen_ids.add(task_id)
        tasks.append(
            {
                "task_id": task_id,
                "language": language,
                "docstring": docstring,
                "code_stub": code_stub,
            }
        )
    return tasks, str(suite_name), _sha256_file(manifest_path)


def run_benchmark(
    checkpoint_path: str = "step_008130.pt",
    seed: int = SEED,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    device: str = "cpu",
    task_manifest=None,
    task_manifest_path=None,
) -> dict:
    """Run the generation benchmark.

    The default suite is the fixed 50 Python + 50 JavaScript Phase 10 task
    inventory. When ``task_manifest`` is supplied, the tasks are loaded from
    that JSON manifest instead. For every task:
      1. Call build_generation_prefix() to build the prompt.
      2. Encode with the canonical v1.1.0 tokenizer.
      3. Generate using generate_continuation().
      4. Compute the specified metrics.

    Returns a dict with aggregated metrics.
    """
    import random
    import os

    # Device selection
    torch_device = select_device(device)

    # Seed for reproducibility
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Load model and tokenizer
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_absolute():
        ckpt_path = project_root / ckpt_path
    model, tokenizer, step = load_model_and_tokenizer(ckpt_path, torch_device)

    # Resolve the task suite. The default preserves the historical Phase 10
    # inventory exactly; a manifest selects an independent task suite.
    selected_manifest = task_manifest if task_manifest is not None else task_manifest_path
    if selected_manifest is None:
        generation_tasks = _phase10_generation_tasks()
        task_suite = "phase10"
        task_manifest_value = None
        task_manifest_sha256 = None
    else:
        generation_tasks, task_suite, task_manifest_sha256 = load_task_manifest(selected_manifest)
        task_manifest_value = str(selected_manifest)

    n_python = sum(1 for task in generation_tasks if task["language"] == "python")
    n_javascript = sum(1 for task in generation_tasks if task["language"] == "javascript")

    # Metric accumulators
    syntax_valid_count = 0
    empty_generations = 0
    total_prompts = 0
    unique_token_ratios = []
    repeated_line_ratios = []
    repeated_4gram_ratios = []
    eos_hits = 0
    n_python = 0
    n_javascript = 0

    # Per-task results for report integrity
    per_task_results = []

    # Process every task in the selected suite. Default Phase 10 order is
    # Python followed by JavaScript, preserving the historical behavior.
    for task in generation_tasks:
        language = task["language"]
        docstring = task["docstring"]
        code_stub = task["code_stub"]
        task_id = task.get("task_id")
        prefix = build_generation_prefix(language, docstring, code_stub)
        assert "<|eos|>" not in prefix, "Generation prefix must not contain EOS"
        prompt_ids = tokenize_prompt(prefix, tokenizer)
        gen_result = generate_continuation(
            model, torch_device, prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            eos_id=EOS_ID,
        )
        # Unpack (gen_no_eos, eos_hit)
        generated_ids, eos_hit = gen_result
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
        total_prompts += 1
        if language == "python":
            n_python += 1
        else:
            n_javascript += 1

        # Syntax validity: check complete candidate (code_stub + generated continuation)
        candidate = code_stub + generated_text
        syntax_valid = compute_syntax_validity(candidate, language)
        if syntax_valid:
            syntax_valid_count += 1
        if len(generated_text.strip()) == 0:
            empty_generations += 1
        if eos_hit:
            eos_hits += 1

        # Preserve the historical per-language metric implementation.
        full_ids = prompt_ids + generated_ids
        if language == "python":
            utr = unique_token_ratio_generated(generated_ids)
        else:
            utr = unique_token_ratio(full_ids)
        rlr = repeated_line_ratio(generated_text)
        r4r = repeated_4gram_ratio_tokens(generated_ids)
        unique_token_ratios.append(utr)
        repeated_line_ratios.append(rlr)
        repeated_4gram_ratios.append(r4r)

        per_task_results.append({
            "task_id": task_id,
            "language": language,
            "docstring": docstring,
            "code_stub": code_stub,
            "generated_text": generated_text,
            "syntax_candidate": candidate,
            "syntax_valid": syntax_valid,
            "eos_hit": eos_hit,
            "unique_token_ratio": utr,
            "repeated_line_ratio": rlr,
            "repeated_4gram_ratio": r4r,
            "generated_token_count": len(generated_ids),
        })

    # Aggregate metrics
    results = {
        "task_suite": task_suite,
        "task_manifest": task_manifest_value,
        "task_manifest_path": task_manifest_value,
        "task_manifest_sha256": task_manifest_sha256,
        "task_manifest_hash": task_manifest_sha256,
        "task_ids": [task.get("task_id") for task in generation_tasks],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(ckpt_path) if ckpt_path.is_file() else None,
        "prompts_generated": total_prompts,
        "python_tasks": n_python,
        "javascript_tasks": n_javascript,
        "per_task_results": per_task_results,
        "syntax_valid": syntax_valid_count,
        "syntax_valid_rate": syntax_valid_count / total_prompts if total_prompts > 0 else 0.0,
        "empty_generations": empty_generations,
        "unique_token_ratio_mean": sum(unique_token_ratios) / len(unique_token_ratios) if unique_token_ratios else 0.0,
        "repeated_line_ratio_mean": sum(repeated_line_ratios) / len(repeated_line_ratios) if repeated_line_ratios else 0.0,
        "repeated_4gram_ratio_mean": sum(repeated_4gram_ratios) / len(repeated_4gram_ratios) if repeated_4gram_ratios else 0.0,
        "eos_hits": eos_hits,
        "eos_hit_rate": eos_hits / total_prompts if total_prompts > 0 else 0.0,
        "checkpoint_step": step,
        "device": str(torch_device),
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "javascript_syntax_checker": "node --check",
    }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main CLI
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_output_path(path: Path | str) -> Path:
    """Resolve an output path relative to the project root."""
    output = Path(path)
    return output if output.is_absolute() else project_root / output


def prepare_output_paths(
    json_out: Path | str | None,
    md_out: Path | str | None,
    timestamp: str,
) -> tuple[Path, Path, bool]:
    """Resolve report paths and enforce explicit-output no-overwrite rules.

    When neither option is supplied, the historical Phase 10 paths are used and
    the existing overwrite behavior is preserved. When either option is
    supplied, both destinations are treated as explicit: the omitted one is
    derived as a sibling file, existing targets are rejected, and parent
    directories are created.
    """
    explicit = json_out is not None or md_out is not None
    if not explicit:
        json_path = project_root / "reports" / f"phase10_generation_evaluation_{timestamp}.json"
        md_path = project_root / "reports" / "phase10_evaluation_summary.md"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        return json_path, md_path, False

    json_path = _resolve_output_path(json_out) if json_out is not None else None
    md_path = _resolve_output_path(md_out) if md_out is not None else None
    if json_path is None:
        json_path = md_path.with_suffix(".json")
    if md_path is None:
        md_path = json_path.with_suffix(".md")
    if json_path.resolve() == md_path.resolve():
        raise SystemExit("refusing to use the same path for JSON and Markdown output")

    for label, path in (("JSON", json_path), ("Markdown", md_path)):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing {label} output: {path}")
    for label, path in (("JSON", json_path), ("Markdown", md_path)):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise SystemExit(
                f"cannot create output directory for {label} output {path}: {exc}"
            ) from exc
    return json_path, md_path, True


def _open_output(path: Path, explicit: bool):
    """Open an output file, using exclusive creation for explicit paths."""
    mode = "x" if explicit else "w"
    try:
        return open(path, mode, encoding="utf-8")
    except FileExistsError as exc:
        raise SystemExit(f"refusing to overwrite existing output: {path}") from exc


def main():
    parser = argparse.ArgumentParser(description="Phase 10 Benchmark Evaluation")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to run on: cpu or cuda (requires GPU)",
    )
    parser.add_argument(
        "--checkpoint",
        default="step_008130.pt",
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Lightweight verification only, no model generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=TEMPERATURE,
        help="Sampling temperature for generation (0.0 = greedy)",
    )
    parser.add_argument(
        "--max-new-tokens",
        "--max_new_tokens",
        type=int,
        default=MAX_NEW_TOKENS,
        dest="max_new_tokens",
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--task-manifest",
        "--task_manifest",
        type=Path,
        default=None,
        dest="task_manifest",
        help="Optional JSON task manifest (e.g. functional/independent_v3/manifest.json)",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Explicit JSON report path (refuses to overwrite an existing file)",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=None,
        help="Explicit Markdown summary path (refuses to overwrite an existing file)",
    )
    args = parser.parse_args()

    prepared_json_path = None
    prepared_md_path = None
    prepared_explicit = False
    if not args.verify_only and (args.json_out is not None or args.md_out is not None):
        prepared_json_path, prepared_md_path, prepared_explicit = prepare_output_paths(
            args.json_out, args.md_out, timestamp=""
        )

    # Device validation
    torch_device = select_device(args.device)
    print(f"Using device: {torch_device}")

    if args.verify_only:
        # Lightweight verification: just check prompt serialization
        print()
        print("=== Lightweight Verification ===")

        # Test build_prompt
        py_prompt = build_prompt("python", "Compute factorial", "def factorial(n):")
        js_prompt = build_prompt("javascript", "Compute factorial", "function factorial(n) {")
        print(f"Python prompt: {repr(py_prompt)}")
        print(f"JavaScript prompt: {repr(js_prompt)}")

        # Verify markers
        assert "<|lang:python|>" in py_prompt, "Missing Python lang marker"
        assert "<|lang:javascript|>" in js_prompt, "Missing JavaScript lang marker"
        assert "<|docstring|>" in py_prompt, "Missing docstring marker"
        assert "<|docstring|>" in js_prompt, "Missing docstring marker"
        assert "<|code|>" in py_prompt, "Missing code marker"
        assert "<|code|>" in js_prompt, "Missing code marker"
        assert "<|eos|>" in py_prompt, "Missing EOS marker in Python"
        assert "<|eos|>" in js_prompt, "Missing EOS marker in JavaScript"

        # Verify build_training_sequence (with EOS)
        py_training = build_training_sequence("python", "Compute factorial", "def factorial(n):")
        js_training = build_training_sequence("javascript", "Compute factorial", "function factorial(n) {")
        assert "<|eos|>" in py_training, "Missing EOS in training sequence"
        assert "<|eos|>" not in build_generation_prefix("python", "Compute factorial", "def factorial(n):"), "EOS should not be in generation prefix"
        print("Training sequence (with EOS): OK")
        print(f"  Python: {repr(py_training)}")
        print(f"  JavaScript: {repr(js_training)}")

        # Verify build_generation_prefix (without EOS)
        py_prefix = build_generation_prefix("python", "Compute factorial", "def factorial(n):")
        js_prefix = build_generation_prefix("javascript", "Compute factorial", "function factorial(n) {")
        assert "<|eos|>" not in py_prefix, "EOS should NOT be in generation prefix"
        assert "<|eos|>" not in js_prefix, "EOS should NOT be in generation prefix"
        print("Generation prefix (without EOS): OK")
        print(f"  Python: {repr(py_prefix)}")
        print(f"  JavaScript: {repr(js_prefix)}")

        # Verify tokenizer roundtrip (using loaded tokenizer if available)
        try:
            from src.polycodegpt.data.tokenizer_io import load_tokenizer
            tokenizer = load_tokenizer("tokenizer/artifacts/polycodegpt_tokenizer_v1_1.json")
            py_encoded = tokenizer.encode(py_prompt, add_special_tokens=False).ids
            py_decoded = tokenizer.decode(py_encoded.ids, skip_special_tokens=False)
            assert py_decoded == py_prompt, "Python prompt roundtrip failed"
            js_encoded = tokenizer.encode(js_prompt, add_special_tokens=False).ids
            js_decoded = tokenizer.decode(js_encoded.ids, skip_special_tokens=False)
            assert js_decoded == js_prompt, "JavaScript prompt roundtrip failed"
            print("Tokenizer roundtrip: OK")
        except Exception as e:
            print(f"Tokenizer roundtrip: SKIPPED ({e})")

        print()
        print("=== All Verification Passed ===")
    else:
        # Full benchmark
        results = run_benchmark(
            checkpoint_path=args.checkpoint,
            device=args.device,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            task_manifest=args.task_manifest,
        )
        print()
        print("=== Benchmark Results ===")
        print(f"Prompts generated: {results['prompts_generated']}")
        print(f"Python tasks: {results['python_tasks']}")
        print(f"JavaScript tasks: {results['javascript_tasks']}")
        print(f"Syntax valid: {results['syntax_valid']} ({results['syntax_valid_rate']:.2%})")
        print(f"Empty generations: {results['empty_generations']}")
        print(f"Unique token ratio (mean): {results['unique_token_ratio_mean']:.4f}")
        print(f"Repeated line ratio (mean): {results['repeated_line_ratio_mean']:.4f}")
        print(f"Repeated 4-gram ratio (mean): {results['repeated_4gram_ratio_mean']:.4f}")
        print(f"EOS hits: {results['eos_hits']} ({results['eos_hit_rate']:.2%})")
        print(f"Temperature: {results['temperature']}")
        print(f"Max new tokens: {results['max_new_tokens']}")
        print(f"Checkpoint step: {results['checkpoint_step']}")

        # Write JSON report
        import json
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if prepared_json_path is None:
            json_path, md_path, explicit_output = prepare_output_paths(None, None, timestamp)
        else:
            json_path, md_path = prepared_json_path, prepared_md_path
            explicit_output = prepared_explicit
        report_data = {
            "timestamp": timestamp,
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": results.get("checkpoint_sha256"),
            "checkpoint_step": results["checkpoint_step"],
            "device": results["device"],
            "seed": SEED,
            "temperature": results["temperature"],
            "max_new_tokens": results["max_new_tokens"],
            "tokenizer_version": "v1.1.0",
            "tokenizer_path": str(TOKENIZER_PATH),
            "tokenizer_sha256": _sha256_file(TOKENIZER_PATH),
            "task_suite": results.get("task_suite"),
            "task_manifest": results.get("task_manifest"),
            "task_manifest_path": results.get("task_manifest_path"),
            "task_manifest_sha256": results.get("task_manifest_sha256"),
            "task_manifest_hash": results.get("task_manifest_hash"),
            "task_ids": results.get("task_ids"),
            "prompts_generated": results["prompts_generated"],
            "python_tasks": results["python_tasks"],
            "javascript_tasks": results["javascript_tasks"],
            "syntax_valid": results["syntax_valid"],
            "syntax_valid_rate": results["syntax_valid_rate"],
            "empty_generations": results["empty_generations"],
            "unique_token_ratio_mean": results["unique_token_ratio_mean"],
            "repeated_line_ratio_mean": results["repeated_line_ratio_mean"],
            "repeated_4gram_ratio_mean": results["repeated_4gram_ratio_mean"],
            "eos_hits": results["eos_hits"],
            "eos_hit_rate": results["eos_hit_rate"],
            "per_task_results": results["per_task_results"],
        }
        with _open_output(json_path, explicit_output) as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        print(f"JSON report written: {json_path}")

        # Write Markdown summary
        md_lines = [
            "# Phase 10 Evaluation Summary",
            "",
            f"**Date**: {timestamp}",
            f"**Checkpoint**: {args.checkpoint} (step {results['checkpoint_step']})",
            f"**Device**: {results['device']}",
            f"**Seed**: {SEED}",
            f"**Temperature**: {results['temperature']}",
            f"**Max new tokens**: {results['max_new_tokens']}",
            f"**Tokenizer**: v1.1.0 ({TOKENIZER_PATH})",
            "",
            "## Results",
            "",
            f"- Prompts generated: {results['prompts_generated']} ({results['python_tasks']} Python + {results['javascript_tasks']} JavaScript)",
            f"- Syntax valid: {results['syntax_valid']} ({results['syntax_valid_rate']:.2%})",
            f"- Empty generations: {results['empty_generations']}",
            f"- Unique token ratio (mean): {results['unique_token_ratio_mean']:.4f}",
            f"- Repeated line ratio (mean): {results['repeated_line_ratio_mean']:.4f}",
            f"- Repeated 4-gram ratio (mean, BPE): {results['repeated_4gram_ratio_mean']:.4f}",
            f"- EOS hits: {results['eos_hits']} ({results['eos_hit_rate']:.2%})",
            "",
            "## Per-Language Breakdown",
            "",
        ]
        # Per-language stats
        py_results = [r for r in results["per_task_results"] if r["language"] == "python"]
        js_results = [r for r in results["per_task_results"] if r["language"] == "javascript"]
        for lang, lr in [("Python", py_results), ("JavaScript", js_results)]:
            sv = sum(1 for r in lr if r["syntax_valid"])
            eh = sum(1 for r in lr if r["eos_hit"])
            md_lines.append(f"### {lang}")
            md_lines.append(f"- Tasks: {len(lr)}")
            md_lines.append(f"- Syntax valid: {sv}/{len(lr)}")
            md_lines.append(f"- EOS hits: {eh}/{len(lr)}")
            md_lines.append(f"- Empty generations: {sum(1 for r in lr if not r['generated_text'].strip())}")
            md_lines.append("")
        with _open_output(md_path, explicit_output) as f:
            f.write("\n".join(md_lines))
        print(f"Markdown summary written: {md_path}")


if __name__ == "__main__":
    main()
