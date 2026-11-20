"""Checkpoint helpers (weights-only, metadata in plain dict)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch

from .config import ModelConfig


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def save_checkpoint(
    path: str | Path,
    model: Any,
    config: ModelConfig,
    tokenizer_version: str = "",
    tokenizer_path: str | Path | None = None,
    init_metadata: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "state_dict": model.state_dict(),
        "config": config.to_dict(),
        "tokenizer_version": tokenizer_version,
        "tokenizer_sha256": _sha256_file(tokenizer_path) if tokenizer_path else "",
        "param_count": sum(p.numel() for p in model.parameters()),
        "init_metadata": dict(
            init_metadata
            if init_metadata is not None
            else (model.initialization_metadata() if hasattr(model, "initialization_metadata") else {})
        ),
    }
    if extra:
        payload["extra"] = dict(extra)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, device: str = "cpu"):
    path = Path(path)
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=device)
    config = ModelConfig.from_dict(payload["config"])
    meta = {k: v for k, v in payload.items() if k not in ("state_dict", "config")}
    return payload["state_dict"], config, meta
