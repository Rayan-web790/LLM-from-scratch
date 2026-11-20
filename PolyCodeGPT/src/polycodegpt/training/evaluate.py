"""Validation: fixed subsets, token-weighted loss, and perplexity.

The validation loss is always a *token-weighted* average: each batch contributes
``loss_batch * valid_tokens_batch``. Averaging batch means would silently weight a
short final batch as heavily as a full one, so the token count is carried
explicitly and combined at the end.
"""

from __future__ import annotations

import contextlib
import math
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

IGNORE_INDEX = -100


@contextlib.contextmanager
def autocast_context(device_type: str, dtype: torch.dtype | None):
    """Autocast for ``dtype``, or a no-op when ``dtype`` is None (FP32)."""
    if dtype is None:
        yield
    else:
        with torch.autocast(device_type=device_type, dtype=dtype):
            yield


def perplexity(loss: float) -> float:
    """``exp(loss)``, saturating to ``inf`` instead of overflowing."""
    if loss != loss:  # NaN
        return float("nan")
    if loss > 700.0:
        return float("inf")
    return math.exp(loss)


def fixed_validation_indices(num_blocks: int, count: int, seed: int = 0) -> list[int]:
    """Deterministic subset of block indices, reused for every evaluation."""
    if num_blocks <= 0:
        raise ValueError("validation dataset has no blocks")
    if count <= 0:
        raise ValueError("count must be positive")
    if count >= num_blocks:
        return list(range(num_blocks))
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    permutation = torch.randperm(num_blocks, generator=generator)
    return sorted(int(i) for i in permutation[:count].tolist())


def build_validation_loader(
    dataset: Dataset,
    indices: Sequence[int] | None = None,
    batch_size: int = 8,
    num_workers: int = 0,
) -> DataLoader:
    """Sequential, unshuffled loader over a fixed subset of validation blocks."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    view: Dataset = dataset if indices is None else Subset(dataset, list(indices))
    return DataLoader(
        view,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    batches: Iterable,
    device: str | torch.device,
    *,
    autocast_dtype: torch.dtype | None = None,
    autocast_device_type: str | None = None,
    max_batches: int | None = None,
) -> dict[str, float | int]:
    """Token-weighted cross-entropy over ``batches``.

    The model is switched to ``eval()`` and its previous mode is restored even if
    evaluation raises, so dropout is disabled here and nowhere else.
    """
    device = torch.device(device)
    device_type = autocast_device_type or device.type
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    sequences = 0
    seen_batches = 0

    try:
        for batch_index, (input_ids, targets) in enumerate(batches):
            if max_batches is not None and batch_index >= max_batches:
                break
            input_ids = input_ids.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with autocast_context(device_type, autocast_dtype):
                _, loss = model(input_ids, targets=targets)
            if loss is None:
                raise RuntimeError("model returned no loss for a batch with targets")
            valid_tokens = int((targets != IGNORE_INDEX).sum().item())
            if valid_tokens == 0:
                continue
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite validation loss: {loss.item()}")
            total_loss += float(loss.item()) * valid_tokens
            total_tokens += valid_tokens
            sequences += int(input_ids.shape[0])
            seen_batches += 1
    finally:
        model.train(was_training)

    if total_tokens == 0:
        raise ValueError("no validation tokens were evaluated")

    mean_loss = total_loss / total_tokens
    return {
        "loss": mean_loss,
        "perplexity": perplexity(mean_loss),
        "tokens": total_tokens,
        "sequences": sequences,
        "batches": seen_batches,
        "loss_sum": total_loss,
    }


def combine_losses(results: Iterable[Mapping[str, float | int]]) -> dict[str, float | int]:
    """Combine partial evaluation results into one token-weighted average."""
    loss_sum = sum(float(r["loss_sum"]) for r in results)
    tokens = sum(int(r["tokens"]) for r in results)
    if tokens == 0:
        raise ValueError("no tokens to combine")
    mean = loss_sum / tokens
    return {
        "loss": mean,
        "perplexity": perplexity(mean),
        "tokens": tokens,
        "batches": sum(int(r["batches"]) for r in results),
        "sequences": sum(int(r["sequences"]) for r in results),
        "loss_sum": loss_sum,
    }
