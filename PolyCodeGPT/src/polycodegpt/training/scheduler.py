"""Learning-rate schedule: linear warmup followed by cosine decay to a floor.

The schedule is expressed as a function of the *number of completed optimizer
steps*, so it can be inspected at any step, resumed from a checkpoint, and tested
without running the optimizer. It updates the learning rate of every parameter
group in place (all groups currently share the base rate).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch


def warmup_steps_for(total_steps: int, warmup_fraction: float) -> int:
    """Warmup length for a schedule, at least one step."""
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    if not 0.0 <= warmup_fraction < 1.0:
        raise ValueError(f"warmup_fraction must be in [0,1), got {warmup_fraction}")
    if warmup_fraction == 0.0:
        return 0
    return max(1, int(round(total_steps * warmup_fraction)))


@dataclass(frozen=True)
class ScheduleConfig:
    """Configuration of the warmup + cosine schedule."""

    total_steps: int = 8129
    warmup_fraction: float = 0.02
    min_lr_ratio: float = 0.1

    def __post_init__(self) -> None:
        if self.total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {self.total_steps}")
        if not 0.0 <= self.warmup_fraction < 1.0:
            raise ValueError(f"warmup_fraction must be in [0,1), got {self.warmup_fraction}")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError(f"min_lr_ratio must be in [0,1], got {self.min_lr_ratio}")

    @property
    def warmup_steps(self) -> int:
        return warmup_steps_for(self.total_steps, self.warmup_fraction)


def lr_factor(step: int, total_steps: int, warmup_steps: int, min_lr_ratio: float = 0.1) -> float:
    """Multiplier applied to the base learning rate at ``step``.

    ``step`` is the number of optimizer steps already completed, so the first
    update runs at ``base / warmup_steps``. Warmup is linear; decay is a half
    cosine from the base rate down to ``min_lr_ratio * base`` at the final step.
    """
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError(f"min_lr_ratio must be in [0,1], got {min_lr_ratio}")
    if step < 0:
        raise ValueError(f"step must be >= 0, got {step}")

    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)

    span = total_steps - warmup_steps
    if span <= 0:
        return min_lr_ratio

    progress = min(1.0, max(0.0, float(step - warmup_steps + 1) / float(span)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


class WarmupCosineScheduler:
    """Sets the optimizer learning rate from a warmup + cosine schedule.

    ``step()`` must be called *after* a successful ``optimizer.step()``: the
    schedule describes completed updates, so a skipped update (for example an
    FP16 step discarded because of non-finite gradients) must not advance it.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int = 8129,
        warmup_steps: int | None = None,
        warmup_fraction: float = 0.02,
        min_lr_ratio: float = 0.1,
    ) -> None:
        if not optimizer.param_groups:
            raise ValueError("optimizer has no parameter groups")
        self.optimizer = optimizer
        self.total_steps = int(total_steps)
        self.warmup_steps = (
            warmup_steps_for(self.total_steps, warmup_fraction)
            if warmup_steps is None
            else int(warmup_steps)
        )
        self.min_lr_ratio = float(min_lr_ratio)
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.last_step = 0
        self._apply()

    # ------------------------------------------------------------- schedule
    def factor_at(self, step: int) -> float:
        return lr_factor(step, self.total_steps, self.warmup_steps, self.min_lr_ratio)

    def lr_at(self, step: int) -> list[float]:
        """Learning rate of each parameter group at ``step``."""
        factor = self.factor_at(step)
        return [base * factor for base in self.base_lrs]

    def _apply(self) -> None:
        for group, lr in zip(self.optimizer.param_groups, self.lr_at(self.last_step)):
            group["lr"] = lr

    def set_total_steps(self, total_steps: int, warmup_fraction: float | None = None) -> None:
        """Retarget the schedule while keeping the completed-step progress.

        Used when a run is planned at a slightly different length than the one the
        checkpoint was saved with (for example adding a final partial update that
        covers the blocks left over at the end of an epoch). ``last_step`` is
        untouched, so progress resumes exactly where it was; the rate for the next
        update is recomputed from the new curve.
        """
        total_steps = int(total_steps)
        if total_steps < self.last_step:
            raise ValueError(
                f"cannot retarget the schedule to {total_steps} steps, {self.last_step} are complete"
            )
        self.total_steps = total_steps
        if warmup_fraction is not None:
            self.warmup_steps = warmup_steps_for(total_steps, warmup_fraction)
        else:
            self.warmup_steps = min(self.warmup_steps, total_steps)
        self._apply()

    def step(self) -> None:
        """Advance one completed optimizer step and apply the new rate."""
        if self.last_step < self.total_steps:
            self.last_step += 1
        self._apply()

    def get_last_lr(self) -> list[float]:
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    # ---------------------------------------------------------- checkpointing
    def state_dict(self) -> dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr_ratio": self.min_lr_ratio,
            "base_lrs": list(self.base_lrs),
            "last_step": int(self.last_step),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        stored = int(state["total_steps"])
        if stored > self.total_steps:
            raise ValueError(
                f"cannot shrink scheduler total_steps from {self.total_steps} to {stored} "
                f"({self.last_step} steps already complete)"
            )
        # If extending the schedule, keep the existing progress and adjust
        # warmup_steps to the new total_steps proportionally, but never below
        # the stored value.  last_step stays exactly where it was.
        if stored < self.total_steps and self.last_step > 0:
            # Extend: preserve the completed-step count, recompute warmup.
            stored_warmup = int(state.get("warmup_steps", warmup_steps_for(stored, 0.02)))
            # Keep the ratio between warmup and total roughly proportional so the
            # lr at the resume step is approximately the same.
            if stored_warmup < self.last_step:
                stored_warmup = max(1, self.last_step)
            self.warmup_steps = stored_warmup
        else:
            self.warmup_steps = int(state.get("warmup_steps", warmup_steps_for(stored, 0.02)))
        self.min_lr_ratio = float(state["min_lr_ratio"])
        self.base_lrs = [float(x) for x in state["base_lrs"]]
        if len(self.base_lrs) != len(self.optimizer.param_groups):
            raise ValueError("scheduler base_lrs does not match the optimizer's parameter groups")
        self.last_step = int(state["last_step"])
        self._apply()
