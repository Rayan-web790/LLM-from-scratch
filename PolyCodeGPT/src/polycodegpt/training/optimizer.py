"""AdamW optimizer construction with decay / no-decay parameter groups.

Decay policy (Greg-style): every weight with two or more dimensions is decayed;
biases, LayerNorm parameters and other one-dimensional parameters are not. The
policy lives here rather than at the call site so the pilot, the resume path and
the tests all use the same rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

DECAY_GROUP = "decay"
NO_DECAY_GROUP = "no_decay"


@dataclass(frozen=True)
class OptimizerConfig:
    """Optimizer hyperparameters. Defaults mirror ``configs/training.yaml``."""

    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")
        if not 0.0 <= self.beta1 < 1.0:
            raise ValueError(f"beta1 must be in [0,1), got {self.beta1}")
        if not 0.0 <= self.beta2 < 1.0:
            raise ValueError(f"beta2 must be in [0,1), got {self.beta2}")
        if self.eps <= 0:
            raise ValueError(f"eps must be positive, got {self.eps}")


def should_decay(name: str, param: torch.Tensor) -> bool:
    """Return True for weights that take weight decay.

    Decayed: matrices (embeddings, projections, FFN weights).
    Not decayed: biases, LayerNorm weights/biases, any 1-D parameter.
    """
    if param.dim() < 2:
        return False
    if name.endswith(".bias"):
        return False
    lowered = name.lower()
    if lowered.startswith("ln") or ".ln" in lowered or "layernorm" in lowered or "norm" in lowered:
        return False
    return True


def build_parameter_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """Split trainable parameters into decay and no-decay groups.

    Tied parameters (``lm_head.weight`` is ``token_embedding.weight``) appear once
    in ``named_parameters`` and are therefore never duplicated across groups.
    """
    if weight_decay < 0:
        raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")

    decay: list[torch.Tensor] = []
    no_decay: list[torch.Tensor] = []
    seen: set[int] = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in seen:  # shared storage (weight tying)
            continue
        seen.add(id(param))
        if should_decay(name, param):
            decay.append(param)
        else:
            no_decay.append(param)

    if not decay and not no_decay:
        raise ValueError("model has no trainable parameters")

    groups: list[dict[str, Any]] = []
    if decay:
        groups.append({"params": decay, "weight_decay": float(weight_decay), "group_name": DECAY_GROUP})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0, "group_name": NO_DECAY_GROUP})
    return groups


def build_optimizer(model: nn.Module, config: OptimizerConfig | None = None) -> torch.optim.AdamW:
    """Create AdamW over the model's decay / no-decay groups.

    ``foreach`` and fused kernels are left at their defaults: no unverified
    optimizer optimisations are enabled by this project.
    """
    config = config or OptimizerConfig()
    groups = build_parameter_groups(model, config.weight_decay)
    return torch.optim.AdamW(
        groups,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay,
    )


def parameter_group_report(optimizer: torch.optim.Optimizer) -> list[dict[str, Any]]:
    """Summarise each parameter group for logs and checkpoints."""
    report: list[dict[str, Any]] = []
    for index, group in enumerate(optimizer.param_groups):
        tensors = list(group["params"])
        report.append(
            {
                "index": index,
                "group": group.get("group_name", f"group_{index}"),
                "tensors": len(tensors),
                "parameters": int(sum(t.numel() for t in tensors)),
                "weight_decay": float(group["weight_decay"]),
                "learning_rate": float(group["lr"]),
            }
        )
    return report
