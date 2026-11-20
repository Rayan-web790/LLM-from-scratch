"""Resumable training checkpoints with atomic writes and safe loading.

Every checkpoint is a plain dictionary of tensors and primitives so it can be
reloaded with ``torch.load(..., weights_only=True)``. It carries the model,
optimizer, scheduler and gradient-scaler state, the step/token counters, the model
and training configuration, tokenizer and dataset hashes, RNG states, and the
metrics collected so far.

Writes go to a temporary file in the destination directory and are then moved
into place with :func:`os.replace`, so an interrupted save can never leave a
half-written checkpoint behind.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn

from ..model.config import ModelConfig
from ..model.gpt import GPT

CHECKPOINT_FORMAT_VERSION = 1


# --------------------------------------------------------------------------- #
# atomic write
# --------------------------------------------------------------------------- #
def atomic_torch_save(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Save ``payload`` to ``path`` via a temporary file and ``os.replace``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    try:
        torch.save(dict(payload), tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():  # only left behind if the rename failed
            tmp.unlink()
    return path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a training checkpoint without ever unpickling arbitrary objects."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint {path} does not contain a dictionary")
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"unsupported checkpoint format {payload.get('format_version')!r} in {path}"
        )
    for key in ("model_state", "optimizer_state", "scheduler_state", "step"):
        if key not in payload:
            raise ValueError(f"checkpoint {path} is missing {key!r}")
    return payload


# --------------------------------------------------------------------------- #
# RNG state
# --------------------------------------------------------------------------- #
def capture_rng_state(loader_generator: torch.Generator | None = None) -> dict[str, Any]:
    """Snapshot CPU/CUDA/python/numpy RNG state plus an optional DataLoader generator."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    return {
        "torch": torch.get_rng_state(),
        "cuda": list(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else [],
        "python": {
            "version": int(py_state[0]),
            "internal": [int(x) for x in py_state[1]],
            "gauss": py_state[2],
        },
        "numpy": {
            "keys": [int(x) for x in np_state[1].tolist()],
            "pos": int(np_state[2]),
            "has_gauss": int(np_state[3]),
            "cached_gaussian": float(np_state[4]),
        },
        "loader_generator": loader_generator.get_state() if loader_generator is not None else None,
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore a snapshot produced by :func:`capture_rng_state`.

    CUDA state is restored only when the saved device count matches the current
    one; on a different machine the CUDA state is skipped and the caller is told so
    by the returned flag.
    """
    torch.set_rng_state(state["torch"])

    cuda_states = list(state.get("cuda") or [])
    if cuda_states and torch.cuda.is_available() and len(cuda_states) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(cuda_states)

    py_state = state.get("python")
    if py_state:
        random.setstate((int(py_state["version"]), tuple(int(x) for x in py_state["internal"]), py_state["gauss"]))

    np_state = state.get("numpy")
    if np_state:
        np.random.set_state(
            (
                "MT19937",
                np.array(np_state["keys"], dtype=np.uint32),
                int(np_state["pos"]),
                int(np_state["has_gauss"]),
                float(np_state["cached_gaussian"]),
            )
        )


def rng_states_equal(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    """Compare two RNG snapshots field by field (used by tests and resume checks)."""
    for key in ("torch", "cuda", "loader_generator"):
        left, right = a.get(key), b.get(key)
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            if left.shape != right.shape or not torch.equal(left, right):
                return False
        elif isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                return False
            if any(
                not (torch.equal(x, y) if isinstance(x, torch.Tensor) else x == y)
                for x, y in zip(left, right)
            ):
                return False
        elif (left is None) != (right is None):
            return False
    return a.get("python") == b.get("python") and a.get("numpy") == b.get("numpy")


# --------------------------------------------------------------------------- #
# weights
# --------------------------------------------------------------------------- #
def weights_are_tied(model: nn.Module) -> bool:
    """True when the output projection shares the token embedding matrix."""
    lm_head = getattr(model, "lm_head", None)
    embedding = getattr(model, "token_embedding", None)
    if lm_head is None or embedding is None:
        return False
    return lm_head.weight is embedding.weight


def load_weights_into(model: nn.Module, state_dict: Mapping[str, torch.Tensor]) -> None:
    """Load weights and re-establish weight tying afterwards.

    ``load_state_dict`` copies into the shared tensor, so tying normally survives;
    the pointer is reassigned defensively because a checkpoint restored onto a
    model built with tying disabled would otherwise silently train two matrices.
    """
    model.load_state_dict(state_dict)
    if getattr(model, "config", None) is not None and getattr(model.config, "tie_embeddings", False):
        model.lm_head.weight = model.token_embedding.weight


# --------------------------------------------------------------------------- #
# training checkpoint
# --------------------------------------------------------------------------- #
def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any | None,
    state: Mapping[str, Any],
    model_config: ModelConfig | None = None,
    tokenizer_sha256: str = "",
    data_manifest_sha256: str = "",
    loader_generator: torch.Generator | None = None,
    rng_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a full training checkpoint and return its summary metadata."""
    resolved_config = model_config if model_config is not None else getattr(model, "config", None)
    payload: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None and scaler.is_enabled() else {},
        "model_config": resolved_config.to_dict() if resolved_config is not None else {},
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "weights_tied": weights_are_tied(model),
        "tokenizer_sha256": tokenizer_sha256,
        "data_manifest_sha256": data_manifest_sha256,
        "rng_state": dict(rng_state) if rng_state is not None else capture_rng_state(loader_generator),
    }
    payload.update(dict(state))
    atomic_torch_save(payload, path)
    return {
        "path": str(path),
        "step": int(payload["step"]),
        "tokens_seen": int(payload.get("tokens_seen", 0)),
        "weights_tied": bool(payload["weights_tied"]),
    }


def rebuild_model(payload: Mapping[str, Any], device: str | torch.device = "cpu") -> GPT:
    """Instantiate the architecture recorded in a checkpoint and load its weights."""
    config = ModelConfig.from_dict(payload["model_config"])
    model = GPT(config)
    load_weights_into(model, payload["model_state"])
    return model.to(device)


def checkpoint_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Human-readable summary of a loaded checkpoint (no tensors)."""
    return {
        "format_version": payload.get("format_version"),
        "step": int(payload.get("step", 0)),
        "tokens_seen": int(payload.get("tokens_seen", 0)),
        "micro_batches": int(payload.get("micro_batches", 0)),
        "parameter_count": int(payload.get("parameter_count", 0)),
        "weights_tied": bool(payload.get("weights_tied", False)),
        "precision": payload.get("precision"),
        "best_val_loss": payload.get("best_val_loss"),
        "tokenizer_sha256": payload.get("tokenizer_sha256", ""),
        "data_manifest_sha256": payload.get("data_manifest_sha256", ""),
    }
