"""Inspect the prepared token streams and optionally run a forward-only GPU check.

Usage:
    python scripts/inspect_tokenized_data.py [--forward-check] [--samples N]

Verification performed:
  * manifest/file hashes and token counts
  * language-marker counts per split (proves which sources each split holds)
  * blocks read through TokenBlockDataset, decoded for manual inspection
  * optional one-batch forward pass on CUDA (no backward, no optimizer step)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from polycodegpt.data.dataset import TokenBlockDataset, build_dataloader, open_split  # noqa: E402
from polycodegpt.data.integrity import load_manifest, verify_outputs  # noqa: E402
from polycodegpt.data.prepare import DEFAULT_OUTPUT_DIR  # noqa: E402
from polycodegpt.data.stream import STREAM_DTYPE  # noqa: E402
from polycodegpt.data.tokenizer_io import (  # noqa: E402
    EOS_ID,
    LANG_EN_ID,
    LANG_JAVASCRIPT_ID,
    LANG_PYTHON_ID,
    load_tokenizer,
)
from polycodegpt.model.config import ModelConfig  # noqa: E402
from polycodegpt.model.gpt import GPT  # noqa: E402

CHUNK = 4_000_000


def count_ids(path: Path, targets: set[int]) -> dict[int, int]:
    """Count occurrences of specific token IDs by streaming the memmap."""
    mm = np.memmap(path, dtype=STREAM_DTYPE, mode="r")
    counts = {t: 0 for t in targets}
    for start in range(0, mm.shape[0], CHUNK):
        chunk = np.asarray(mm[start : start + CHUNK])
        for target in targets:
            counts[target] += int(np.count_nonzero(chunk == target))
    return counts


def short(text: str, limit: int = 260) -> str:
    """Trim decoded text for display, cutting on a newline so no UTF-8 char is split."""
    text = text.replace("\r", "")
    if len(text) <= limit:
        return text
    cut = text[:limit]
    newline = cut.rfind("\n")
    if newline > limit // 3:
        cut = cut[:newline]
    return cut.rstrip() + " ..."


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--forward-check", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    manifest = load_manifest(output_dir / "manifest.json")
    print(f"dataset:        {manifest['dataset_version']} ({manifest['manifest_version']})")
    print(
        f"context_length: {manifest['context_length']}  "
        f"window: {manifest['window_length']}  "
        f"stride: {manifest['window_stride']}  "
        f"overlap: {manifest['window_overlap']}"
    )
    print(f"dtype:          {manifest['dtype']} ({manifest['byte_order']}-endian)")
    print(f"tokenizer:      v{manifest['tokenizer']['version']} sha {manifest['tokenizer']['sha256'][:16]}...")
    print(f"serializer:     v{manifest['serializer']['version']} sha {manifest['serializer']['sha256'][:16]}...")

    print("\n--- hash verification ---")
    report = verify_outputs(manifest, output_dir)
    for check in report["splits"]:
        print(f"{check['split']:6s} {'OK' if check['ok'] else 'MISMATCH'}  {check['actual_sha256'][:24]}...")
    print(f"all files verified: {report['ok']}")

    print("\n--- provenance by language marker ---")
    marker_ids = {LANG_PYTHON_ID, LANG_JAVASCRIPT_ID, LANG_EN_ID, EOS_ID}
    expected = {
        "train": manifest["splits"]["train"]["records"] + manifest["splits"]["train"]["documents"],
        "valid": manifest["splits"]["valid"]["records"],
        "test": manifest["splits"]["test"]["records"],
    }
    for split, info in manifest["splits"].items():
        counts = count_ids(output_dir / info["file"], marker_ids)
        examples = counts[LANG_PYTHON_ID] + counts[LANG_JAVASCRIPT_ID] + counts[LANG_EN_ID]
        eos_ok = counts[EOS_ID] == examples
        print(
            f"{split:6s} python={counts[LANG_PYTHON_ID]:>8,} js={counts[LANG_JAVASCRIPT_ID]:>8,} "
            f"en={counts[LANG_EN_ID]:>4,} eos={counts[EOS_ID]:>8,} "
            f"examples={examples:,} (manifest {expected[split]:,}) "
            f"EOS==examples: {eos_ok}"
        )

    tokenizer = load_tokenizer(vocab_size=manifest["tokenizer"]["vocab_size"])

    print("\n--- decoded samples ---")
    for split, info in manifest["splits"].items():
        dataset = open_split(output_dir, split, manifest)
        print(
            f"\n[{split}] {len(dataset):,} blocks of {dataset.window_length} tokens "
            f"(stride {dataset.window_stride}, overlap {dataset.window_overlap}, "
            f"tail {dataset.dropped_trailing_tokens:,})"
        )
        for i in range(min(args.samples, len(dataset))):
            input_ids, targets = dataset[i]
            tokens = input_ids.tolist() + [int(targets[-1])]
            text = tokenizer.decode(tokens, skip_special_tokens=False)
            # shift check on the very sample we print
            assert targets[:-1].tolist() == input_ids[1:].tolist(), "shift mismatch"
            assert int(targets[-1]) == tokens[-1]
            print(f"  block {i}: input {tuple(input_ids.shape)} target {tuple(targets.shape)} "
                  f"dtype {input_ids.dtype}")
            print(f"    {short(text)}")
        dataset.close()

    print("\n--- window geometry ---")
    stride = int(manifest["window_stride"])
    length = int(manifest["window_length"])
    for split, info in manifest["splits"].items():
        dataset = open_split(output_dir, split, manifest)
        if len(dataset) >= 2:
            first, second = dataset.block(0), dataset.block(1)
            assert first.tolist() == dataset.mm[:length].tolist(), "first window is not tokens[0:L+1]"
            assert second.tolist() == dataset.mm[stride : stride + length].tolist(), (
                "second window is not tokens[stride:stride+L+1]"
            )
            shared = int(np.array_equal(first[-1:], second[:1]))
            assert shared == 1, "consecutive windows do not overlap by exactly one token"
            coverage_ok = (
                dataset.token_count
                == dataset.num_blocks * stride + 1 + dataset.dropped_trailing_tokens
            )
            assert coverage_ok, "block count/stride/tail do not account for every token"
            print(
                f"{split:6s} blocks={len(dataset):>7,} stride={stride} overlap=1 "
                f"shared_token={shared} coverage_ok={coverage_ok}"
            )
        else:
            print(f"{split:6s} blocks={len(dataset):>7,} (too short for a pairwise check)")
        dataset.close()

    ok = report["ok"] and manifest["completed_successfully"]
    if args.forward_check:
        ok = run_forward_check(output_dir, manifest, ok, args.batch_size)

    print("\nINSPECTION " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


def run_forward_check(output_dir: Path, manifest: dict, ok: bool, batch_size: int) -> bool:
    """One forward pass through the real model on CUDA. No backward, no optimizer."""
    print("\n--- CUDA forward-only check ---")
    if not torch.cuda.is_available():
        print("CUDA unavailable; skipping forward check")
        return False

    config = ModelConfig.from_dict(manifest["model_config"])
    torch.manual_seed(42)
    model = GPT(config)

    dataset = open_split(output_dir, "valid", manifest, verify=False)
    loader = build_dataloader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda")
    model = model.to(device).eval()
    torch.cuda.reset_peak_memory_stats(device)

    input_ids, targets = next(iter(loader))
    input_ids = input_ids.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)

    with torch.no_grad():
        logits, loss = model(input_ids, targets=targets)

    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated(device) / 1024**2
    peak = torch.cuda.max_memory_allocated(device) / 1024**2

    print(f"device:          {torch.cuda.get_device_name(0)}")
    print(f"input_ids:       {tuple(input_ids.shape)} {input_ids.dtype} on {input_ids.device}")
    print(f"targets:         {tuple(targets.shape)} {targets.dtype}")
    print(f"logits:          {tuple(logits.shape)} {logits.dtype}")
    print(f"forward loss:    {loss.item():.6f} (random init, not a training result)")
    print(f"memory allocated:{allocated:8.2f} MiB")
    print(f"memory peak:     {peak:8.2f} MiB")

    dataset.close()
    return ok and bool(torch.isfinite(loss).item()) and tuple(logits.shape) == (
        batch_size,
        config.context_length,
        config.vocab_size,
    )


if __name__ == "__main__":
    raise SystemExit(main())
