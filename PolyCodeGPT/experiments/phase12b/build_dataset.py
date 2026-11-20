"""Materialize the four Phase 12 experiment datasets and verify them.

Usage::

    python experiments/phase12b/build_dataset.py --experiment control
    python experiments/phase12b/build_dataset.py --all

The per-language tokenizations are cached under ``data/processed/phase12b_work``
and reused, so building the second branch does not re-tokenize the corpus.  Only
``train.bin`` artifacts are produced: the experiments are train-only continuations
and never touch the validation or test splits.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from polycodegpt.data.experiment_builder import (  # noqa: E402
    EXPERIMENTS,
    PROJECT_ROOT as DATA_ROOT,
    build_experiment_dataset,
    build_language_stream,
    save_stream_index,
    stream_cache,
    verify_with_dataloader,
    _doc_items,
    _serialize_items,
)
from polycodegpt.data.stream import stream_token_count  # noqa: E402
from polycodegpt.data.tokenizer_io import TOKENIZER_SHA256, load_tokenizer  # noqa: E402
from polycodegpt.model.config import ModelConfig  # noqa: E402

WORK_DIR = DATA_ROOT / "data" / "processed" / "phase12b_work"
OUT_ROOT = DATA_ROOT / "data" / "processed" / "phase12b_experiments"

CORPUS_TRAIN = DATA_ROOT / "data" / "processed" / "polycodegpt_corpus" / "train.jsonl"
CURATED_JS_TRAIN = DATA_ROOT / "data" / "curated" / "js_v1" / "train.jsonl"

LANGUAGE_SOURCES = {
    # Both code languages come from the authoritative phase-1 training corpus.
    "python": lambda: _serialize_items(CORPUS_TRAIN, language="python"),
    "javascript_original": lambda: _serialize_items(CORPUS_TRAIN, language="javascript"),
    "javascript_curated": lambda: _serialize_items(CURATED_JS_TRAIN, language="javascript"),
    "english_docs": _doc_items,
}


def prepare_streams(config: ModelConfig, force: bool = False) -> dict:
    """Tokenize each language once, reusing the cached index when valid."""
    cached = stream_cache(WORK_DIR)
    tokenizer = load_tokenizer(expected_sha256=TOKENIZER_SHA256, vocab_size=config.vocab_size)
    streams = {}
    for name, source in LANGUAGE_SOURCES.items():
        stream = cached.get(name)
        if stream is not None and not force and stream.path.is_file():
            actual = stream_token_count(stream.path)
            if actual == stream.tokens:
                print(f"  {name}: cached ({stream.tokens:,} tokens, {stream.records:,} records)")
                streams[name] = stream
                continue
        print(f"  {name}: tokenizing ...", flush=True)
        started = time.perf_counter()
        stream = build_language_stream(
            name,
            source(),
            WORK_DIR / f"_{name}.bin",
            tokenizer,
            config.vocab_size,
        )
        save_stream_index(stream, WORK_DIR)
        print(
            f"  {name}: {stream.tokens:,} tokens, {stream.records:,} records, "
            f"{time.perf_counter() - started:.1f}s"
        )
        streams[name] = stream
    return streams


def build_one(experiment_id: str, streams: dict, config: ModelConfig, settings: dict,
              seed: int) -> dict:
    spec = EXPERIMENTS[experiment_id]
    out_dir = OUT_ROOT / experiment_id
    started = time.perf_counter()
    manifest = build_experiment_dataset(
        spec, out_dir, streams, config.vocab_size, config.context_length, seed=seed
    )
    loader_report = verify_with_dataloader(
        Path(manifest["train_bin"]["path"]),
        context_length=config.context_length,
        micro_batch_size=settings["micro_batch_size"],
        gradient_accumulation_steps=settings["gradient_accumulation_steps"],
        vocab_size=config.vocab_size,
        seed=seed,
    )
    manifest["dataloader_verification"] = loader_report
    (out_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    budget = manifest["declared_budget"]["total"]
    materialized = manifest["train_bin"]["tokens"]
    updates = loader_report["total_updates_one_pass"]
    print(f"\n[{experiment_id}] {'-' * 60}")
    print(f"  train.bin:         {manifest['train_bin']['path']}")
    print(f"  sha256:            {manifest['train_bin']['sha256']}")
    print(f"  tokens:            {materialized:,} (declared {budget:,}, delta {materialized - budget:+,})")
    for lang in ("python", "javascript", "english_documentation"):
        entry = manifest["languages"][lang]
        print(
            f"  {lang:<18} tokens={entry['materialized_tokens']:,} "
            f"target={entry['budget_target_tokens']:,} "
            f"unique={entry['unique_records_used']:,}/{entry['available_records']:,} "
            f"epochs={entry['effective_epochs']}"
        )
    print(f"  blocks:            {loader_report['blocks']:,} "
          f"(dropped tail {loader_report['dropped_trailing_tokens']:,})")
    print(f"  first batch:       input={loader_report['first_batch_input_shape']} "
          f"target={loader_report['first_batch_target_shape']}")
    print(f"  updates (1 pass):  {loader_report['full_updates']:,} full + "
          f"{loader_report['final_partial_update_sequences']} sequences final partial "
          f"= {updates:,}")
    print(f"  elapsed:           {time.perf_counter() - started:.1f}s")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Phase 12 experiment datasets")
    ap.add_argument("--experiment", action="append", choices=sorted(EXPERIMENTS), default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force-retokenize", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ids = sorted(EXPERIMENTS) if args.all or not args.experiment else args.experiment
    config = ModelConfig.from_yaml()
    settings = json.loads(
        (Path(__file__).resolve().parent / "runner_settings.json").read_text(encoding="utf-8")
    ) if (Path(__file__).resolve().parent / "runner_settings.json").is_file() else {
        "micro_batch_size": 4,
        "gradient_accumulation_steps": 8,
    }

    print("tokenizing sources ...")
    streams = prepare_streams(config, force=args.force_retokenize)
    for experiment_id in ids:
        build_one(experiment_id, streams, config, settings, args.seed)


if __name__ == "__main__":
    main()
