"""Training engine for PolyCodeGPT.

One optimizer step consumes ``gradient_accumulation_steps`` micro-batches; the
reported training loss is the mean of those micro-batch losses and the learning
rate advances only after the optimizer has actually updated the parameters.
Mixed precision follows the device: BF16 when the GPU supports it, FP16 with a
``GradScaler`` otherwise, FP32 only as a documented fallback.
"""

from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..model.config import ModelConfig
from .checkpoint import (
    capture_rng_state,
    load_weights_into,
    restore_rng_state,
    save_checkpoint,
    weights_are_tied,
)
from .evaluate import autocast_context, evaluate_loss
from .optimizer import OptimizerConfig, build_optimizer, parameter_group_report
from .scheduler import WarmupCosineScheduler, warmup_steps_for

#: Columns written to ``logs/training/*.csv`` (required metrics first).
METRIC_COLUMNS = [
    "optimizer_step",
    "epoch_or_pass_fraction",
    "training_loss",
    "validation_loss",
    "learning_rate",
    "gradient_norm",
    "tokens_seen",
    "elapsed_seconds",
    "allocated_vram",
    "peak_allocated_vram",
    "blocks_consumed",
    "effective_batch_size",
    "tokens_per_step",
    "precision",
    "micro_batch_size",
    "gradient_accumulation",
    "max_gpu_memory_mb",
]

PRECISIONS = ("auto", "bf16", "fp16", "fp32")


# --------------------------------------------------------------------------- #
# precision
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PrecisionPlan:
    """How this run computes forward/backward passes."""

    device_type: str
    precision: str
    autocast_dtype: torch.dtype | None
    use_grad_scaler: bool

    @property
    def requires_scaler(self) -> bool:
        return self.use_grad_scaler


def resolve_precision(
    requested: str = "auto",
    device: str | torch.device | None = None,
    bf16_supported: bool | None = None,
) -> PrecisionPlan:
    """Pick the compute precision for ``device``.

    ``auto`` prefers BF16, then FP16 (with a scaler). CPU evaluation is reported as
    FP32; requesting BF16/FP16 on CPU raises instead of pretending to use it.
    """
    requested = (requested or "auto").lower()
    if requested not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {requested!r}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    if device.type != "cuda":
        if requested in ("bf16", "fp16"):
            raise ValueError(f"precision {requested!r} requires CUDA; device is {device.type}")
        return PrecisionPlan(
            device_type=device.type,
            precision="fp32",
            autocast_dtype=None,
            use_grad_scaler=False,
        )

    supported = torch.cuda.is_bf16_supported() if bf16_supported is None else bool(bf16_supported)
    if requested == "auto":
        requested = "bf16" if supported else "fp16"

    if requested == "bf16":
        if not supported:
            raise ValueError("BF16 was requested but this GPU/CUDA build does not support it")
        return PrecisionPlan("cuda", "bf16", torch.bfloat16, use_grad_scaler=False)
    if requested == "fp16":
        return PrecisionPlan("cuda", "fp16", torch.float16, use_grad_scaler=True)
    return PrecisionPlan("cuda", "fp32", None, use_grad_scaler=False)


def require_cuda(device: str | torch.device | None = None) -> torch.device:
    """Resolve the device and refuse to fall back to CPU silently."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; training will not fall back to CPU")
    device = torch.device(device or "cuda")
    if device.type != "cuda":
        raise RuntimeError(f"training requires a CUDA device, got {device}")
    return device


def apply_determinism(settings: TrainSettings) -> dict[str, Any]:
    """Seed Python/NumPy/PyTorch and configure cuDNN for reproducibility.

    Returns a report of what was applied (and any unavoidable nondeterminism).
    """
    import os
    import random

    try:
        import numpy as np
    except Exception:  # pragma: no cover
        np = None

    report: dict[str, Any] = {"applied": [], "unavoidable": []}
    random.seed(int(settings.python_seed))
    report["applied"].append(f"python random seed={settings.python_seed}")
    if np is not None:
        np.random.seed(int(settings.python_seed))
        report["applied"].append(f"numpy seed={settings.python_seed}")
    torch.manual_seed(int(settings.torch_seed))
    report["applied"].append(f"torch cpu seed={settings.torch_seed}")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(settings.cuda_seed))
        report["applied"].append(f"torch cuda seed={settings.cuda_seed}")
    torch.backends.cudnn.deterministic = bool(settings.cudnn_deterministic)
    torch.backends.cudnn.benchmark = bool(settings.cudnn_benchmark)
    report["applied"].append(
        f"cudnn deterministic={settings.cudnn_deterministic} benchmark={settings.cudnn_benchmark}"
    )
    if settings.deterministic:
        # ``warn_only`` on purpose: strict mode raises for ops that have no
        # deterministic CUDA kernel (the embedding backward uses atomicAdd), which
        # would abort a long run at an arbitrary step.  The warning is recorded in
        # the report instead, and the remaining knobs are still enforced.
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            report["applied"].append("torch.use_deterministic_algorithms(True, warn_only=True)")
        except TypeError:  # pragma: no cover - very old torch
            torch.use_deterministic_algorithms(True)
            report["applied"].append("torch.use_deterministic_algorithms(True)")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        report["applied"].append("CUBLAS_WORKSPACE_CONFIG=:4096:8")
        os.environ.setdefault("PYTHONHASHSEED", str(settings.python_seed))
        report["applied"].append(f"PYTHONHASHSEED={settings.python_seed}")
    else:
        report["unavoidable"].append("deterministic mode disabled by settings")
    report["unavoidable"].append(
        "BF16/FP16 matrix-multiply and embedding-gradient atomic accumulation are not "
        "bitwise reproducible on CUDA; identical seeds give statistically identical, "
        "not byte-identical, runs"
    )
    report["unavoidable"].append(
        "dataloader workers are seeded per-epoch through the loader generator; changing "
        "num_workers or the device count changes the sampling order"
    )
    return report


def make_grad_scaler(plan: PrecisionPlan) -> torch.amp.GradScaler:
    """GradScaler for FP16; a disabled scaler otherwise (its API stays uniform)."""
    return torch.amp.GradScaler("cuda", enabled=plan.use_grad_scaler)


def gpu_memory_mb() -> tuple[float, float]:
    """(allocated, peak allocated) MiB for the current CUDA device, else (0, 0)."""
    if not torch.cuda.is_available():
        return 0.0, 0.0
    return (
        torch.cuda.memory_allocated() / 1024**2,
        torch.cuda.max_memory_allocated() / 1024**2,
    )


# --------------------------------------------------------------------------- #
# settings
# --------------------------------------------------------------------------- #
@dataclass
class TrainSettings:
    """Everything the loop needs; all schedule values come from configuration."""

    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_steps: int = 200
    total_steps: int = 8129
    warmup_fraction: float = 0.02
    min_lr_ratio: float = 0.1
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    gradient_clip: float = 1.0
    precision: str = "auto"
    seed: int = 42
    context_length: int = 512
    log_every: int = 10
    eval_steps: tuple[int, ...] = (50, 100, 150, 200)
    checkpoint_steps: tuple[int, ...] = (50, 100, 150, 200)
    eval_every: int = 0
    checkpoint_every: int = 0
    initial_validation: bool = True
    max_skipped_steps: int = 10
    complete_pass: bool = False
    partial_final_update: bool = False
    # continuation-stage accounting (0 initial_global_step => original from-scratch mode)
    initial_global_step: int = 0
    additional_steps: int = 0
    fresh_schedule: bool = False
    deterministic: bool = True
    python_seed: int | None = None
    torch_seed: int | None = None
    cuda_seed: int | None = None
    dataloader_worker_seed: int | None = None
    cudnn_deterministic: bool = True
    cudnn_benchmark: bool = False

    def __post_init__(self) -> None:
        if self.micro_batch_size <= 0:
            raise ValueError("micro_batch_size must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if not 0.0 <= self.warmup_fraction < 1.0:
            raise ValueError("warmup_fraction must be in [0,1)")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in [0,1]")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")
        if self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive")
        if self.precision.lower() not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}")
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if self.micro_batch_size * self.gradient_accumulation_steps <= 0:
            raise ValueError("effective batch size must be positive")
        if self.eval_every < 0 or self.checkpoint_every < 0:
            raise ValueError("eval_every and checkpoint_every must be >= 0")
        if self.complete_pass and self.max_steps <= 0:
            raise ValueError("complete_pass requires a positive max_steps")
        if self.initial_global_step < 0:
            raise ValueError("initial_global_step must be >= 0")
        if self.additional_steps <= 0:
            self.additional_steps = self.max_steps
        if self.additional_steps <= 0:
            raise ValueError("additional_steps must be positive")
        if self.python_seed is None:
            self.python_seed = self.seed
        if self.torch_seed is None:
            self.torch_seed = self.seed
        if self.cuda_seed is None:
            self.cuda_seed = self.seed
        if self.dataloader_worker_seed is None:
            self.dataloader_worker_seed = self.seed

    @property
    def effective_batch_size(self) -> int:
        """Sequences per optimizer update."""
        return self.micro_batch_size * self.gradient_accumulation_steps

    @property
    def tokens_per_step(self) -> int:
        """Supervised tokens per optimizer update (no masking is applied)."""
        return self.effective_batch_size * self.context_length

    @property
    def warmup_steps(self) -> int:
        return warmup_steps_for(self.total_steps, self.warmup_fraction)

    @property
    def expected_final_global_step(self) -> int:
        """Global step reached after this stage's ``additional_steps`` updates."""
        return int(self.initial_global_step) + int(self.additional_steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "micro_batch_size": self.micro_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "effective_batch_size": self.effective_batch_size,
            "tokens_per_step": self.tokens_per_step,
            "max_steps": self.max_steps,
            "total_steps": self.total_steps,
            "warmup_fraction": self.warmup_fraction,
            "warmup_steps": self.warmup_steps,
            "min_lr_ratio": self.min_lr_ratio,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "gradient_clip": self.gradient_clip,
            "precision": self.precision,
            "seed": self.seed,
            "context_length": self.context_length,
            "eval_steps": list(self.eval_steps),
            "checkpoint_steps": list(self.checkpoint_steps),
            "eval_every": self.eval_every,
            "checkpoint_every": self.checkpoint_every,
            "complete_pass": self.complete_pass,
            "partial_final_update": self.partial_final_update,
            "initial_global_step": self.initial_global_step,
            "additional_steps": self.additional_steps,
            "fresh_schedule": self.fresh_schedule,
            "deterministic": self.deterministic,
            "python_seed": self.python_seed,
            "torch_seed": self.torch_seed,
            "cuda_seed": self.cuda_seed,
            "dataloader_worker_seed": self.dataloader_worker_seed,
            "cudnn_deterministic": self.cudnn_deterministic,
            "cudnn_benchmark": self.cudnn_benchmark,
        }


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def write_metrics_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> Path:
    """Write metric rows with the fixed column order (missing values left blank)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=METRIC_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in METRIC_COLUMNS})
    return path


# --------------------------------------------------------------------------- #
# trainer
# --------------------------------------------------------------------------- #
class Trainer:
    """Gradient-accumulation training loop with periodic validation and checkpoints."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        settings: TrainSettings,
        *,
        device: str | torch.device | None = None,
        val_loader: DataLoader | None = None,
        train_tokens: int | None = None,
        checkpoint_dir: str | Path | None = None,
        model_config: ModelConfig | None = None,
        tokenizer_sha256: str = "",
        data_manifest_sha256: str = "",
        log_fn=print,
    ) -> None:
        self.settings = settings
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if settings.precision.lower() != "fp32" and self.device.type != "cuda":
            raise RuntimeError(f"precision {settings.precision!r} requires CUDA, device is {self.device}")
        self.plan = resolve_precision(settings.precision, self.device)

        self.model = model.to(self.device)
        self.model.train()

        self.optimizer = build_optimizer(
            self.model,
            OptimizerConfig(
                learning_rate=settings.learning_rate,
                weight_decay=settings.weight_decay,
            ),
        )
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            total_steps=settings.total_steps,
            warmup_fraction=settings.warmup_fraction,
            min_lr_ratio=settings.min_lr_ratio,
        )
        self.scaler = make_grad_scaler(self.plan)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_tokens = int(train_tokens) if train_tokens else None
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None

        self.model_config = model_config
        self.tokenizer_sha256 = tokenizer_sha256
        self.data_manifest_sha256 = data_manifest_sha256
        self.log = log_fn

        # counters
        self.step_index = int(settings.initial_global_step)
        self.stage_step = 0
        self.tokens_seen = 0
        self.initial_tokens_seen = 0
        self.micro_batches = 0
        self.sequences = 0
        self.passes = 0
        self.skipped_steps = 0
        self.best_val_loss = float("inf")
        self.best_step: int | None = None

        # history
        self.history: list[dict[str, Any]] = []
        self.validation_history: list[dict[str, Any]] = []

        # loader bookkeeping for exact resume
        self._loader_generator = getattr(train_loader, "generator", None)
        self._epoch_start_state: torch.Tensor | None = None
        self._epoch_batches_consumed = 0
        self._pending_skip = 0
        self._start_time = time.perf_counter()

        if settings.deterministic:
            self.determinism_report = apply_determinism(settings)
        else:
            self.determinism_report = {"applied": [], "unavoidable": ["deterministic disabled"]}

        self.log(
            f"trainer: device={self.device.type} precision={self.plan.precision} "
            f"micro_batch={settings.micro_batch_size} accum={settings.gradient_accumulation_steps} "
            f"effective_batch={settings.effective_batch_size} "
            f"tokens_per_step={settings.tokens_per_step} "
            f"warmup_steps={settings.warmup_steps}/{settings.total_steps}"
        )
        for group in parameter_group_report(self.optimizer):
            self.log(
                f"  group {group['group']}: {group['parameters']:,} params in "
                f"{group['tensors']} tensors, weight_decay={group['weight_decay']}"
            )

    # -------------------------------------------------------------- precision
    def _autocast(self):
        return autocast_context(self.plan.device_type, self.plan.autocast_dtype)

    # ----------------------------------------------------------- batch stream
    def _batch_stream(self):
        """Yield micro-batches forever, restarting the loader at each epoch end.

        The generator state at the start of each epoch and the number of batches
        consumed are tracked so that a restored run replays the same permutation
        from the same position instead of jumping to a fresh shuffle.
        """
        skip = self._pending_skip
        self._pending_skip = 0
        while True:
            if self._loader_generator is not None:
                if skip and self._epoch_start_state is not None:
                    self._loader_generator.set_state(self._epoch_start_state)
                self._epoch_start_state = self._loader_generator.get_state().clone()
            self._epoch_batches_consumed = 0
            for batch in self.train_loader:
                self._epoch_batches_consumed += 1
                if skip:
                    skip -= 1
                    continue
                yield batch
            self.passes += 1

    # ----------------------------------------------------------- continuation
    @property
    def initial_global_step(self) -> int:
        return int(self.settings.initial_global_step)

    @property
    def additional_steps(self) -> int:
        return int(self.settings.additional_steps)

    @property
    def expected_final_global_step(self) -> int:
        return self.initial_global_step + self.additional_steps

    # -------------------------------------------------------------- one step
    def train_step(self, batches: Iterable) -> dict[str, Any]:
        """Run one optimizer update weighted by supervised target-token count."""
        settings = self.settings
        group = list(batches)
        if not group:
            raise ValueError("train_step needs at least one micro-batch")

        sizes = [int(input_ids.shape[0]) for input_ids, _ in group]
        total_sequences = sum(sizes)
        if total_sequences <= 0:
            raise ValueError("micro-batch group contains no sequences")

        supervised_counts = [
            int((targets != -100).sum().item())
            for _, targets in group
        ]
        total_supervised = sum(supervised_counts)
        if total_supervised <= 0:
            raise ValueError("micro-batch group contains no supervised tokens")

        if any(count <= 0 for count in supervised_counts):
            raise ValueError("micro-batch contains no supervised target tokens")

        full_group = len(group) == settings.gradient_accumulation_steps and all(
            size == settings.micro_batch_size for size in sizes
        )
        if not full_group and not settings.partial_final_update:
            raise RuntimeError(
                f"accumulated {len(group)} micro-batches ({total_sequences} sequences), expected "
                f"{settings.gradient_accumulation_steps} x {settings.micro_batch_size}"
            )

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        lr_used = self.scheduler.get_last_lr()[0]

        micro = 0
        tokens = 0
        loss_sum = 0.0

        for (input_ids, targets), size, valid_tokens in zip(
            group, sizes, supervised_counts
        ):
            input_ids = input_ids.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with self._autocast():
                _, loss = self.model(input_ids, targets=targets)

            if loss is None:
                raise RuntimeError("model returned no loss for a batch with targets")
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite training loss {loss.item()} before step {self.step_index + 1}"
                )

            loss_sum += float(loss.item()) * valid_tokens
            self.tokens_seen += valid_tokens
            self.micro_batches += 1
            micro += 1
            tokens += valid_tokens

            scaled = loss * (valid_tokens / total_supervised)
            if self.scaler.is_enabled():
                self.scaler.scale(scaled).backward()
            else:
                scaled.backward()

        if self.scaler.is_enabled():
            self.scaler.unscale_(self.optimizer)

        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), settings.gradient_clip
            ).item()
        )
        if not math.isfinite(grad_norm):
            raise FloatingPointError(
                f"non-finite gradient norm {grad_norm} after micro-batch {self.micro_batches}"
            )

        stepped = True
        if self.scaler.is_enabled():
            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            stepped = self.scaler.get_scale() >= scale_before
        else:
            self.optimizer.step()

        if stepped:
            self.scheduler.step()
            self.stage_step += 1
            self.step_index = self.initial_global_step + self.stage_step
            self.sequences += total_sequences

        return {
            "loss": loss_sum / total_supervised,
            "grad_norm": grad_norm,
            "stepped": stepped,
            "micro_batches": micro,
            "lr_used": lr_used,
            "sequences": total_sequences,
            "tokens": tokens,
            "partial": not full_group,
        }

    def validate(self, loader: DataLoader | None = None) -> dict[str, Any]:
        """Evaluate the fixed validation subset (or ``loader``) at the current step."""
        loader = loader if loader is not None else self.val_loader
        if loader is None:
            raise ValueError("no validation loader configured")
        result = evaluate_loss(
            self.model,
            loader,
            self.device,
            autocast_dtype=self.plan.autocast_dtype,
            autocast_device_type=self.plan.device_type,
        )
        record = {"step": self.step_index, **{k: result[k] for k in ("loss", "perplexity", "tokens", "sequences", "batches")}}
        self.validation_history.append(record)
        return record

    # ---------------------------------------------------------- checkpoints
    def checkpoint_state(self) -> dict[str, Any]:
        """Counters and histories stored alongside the weights and optimizer state."""
        return {
            "step": self.step_index,
            "stage_step": self.stage_step,
            "initial_global_step": self.initial_global_step,
            "tokens_seen": self.tokens_seen,
            "initial_tokens_seen": self.initial_tokens_seen,
            "micro_batches": self.micro_batches,
            "sequences": self.sequences,
            "passes": self.passes,
            "skipped_steps": self.skipped_steps,
            "best_val_loss": self.best_val_loss,
            "best_step": self.best_step,
            "precision": self.plan.precision,
            "metrics": list(self.history),
            "validation": list(self.validation_history),
            "settings": self.settings.to_dict(),
            "seed": self.settings.seed,
            "loader_batches_consumed": self._epoch_batches_consumed,
            "loader_epoch_state": self._epoch_start_state,
        }

    def save(self, path: str | Path) -> dict[str, Any]:
        return save_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            state=self.checkpoint_state(),
            model_config=self.model_config,
            tokenizer_sha256=self.tokenizer_sha256,
            data_manifest_sha256=self.data_manifest_sha256,
            loader_generator=self._loader_generator,
        )

    def restore(self, payload: dict[str, Any], reset_loader_state: bool = False) -> None:
        """Restore weights, optimizer, scheduler, scaler, counters and RNG state.

        ``reset_loader_state`` restarts consumption from the beginning of the
        *current* dataset instead of replaying the checkpoint's consumed-batch
        count.  A continuation stage trains on a fresh dataset, so the historical
        ``loader_batches_consumed`` (which describes the phase-1 corpus) must not
        be replayed against it: doing so makes ``complete_pass`` mode believe the
        epoch is already exhausted and produces zero updates.
        """
        load_weights_into(self.model, payload["model_state"])
        self.optimizer.load_state_dict(payload["optimizer_state"])
        if self.settings.fresh_schedule:
            # A continuation stage starts a *new* schedule at stage step 0.  The
            # checkpoint's optimizer state carries the old (almost fully decayed)
            # learning rate in its parameter groups, so the base rate has to be
            # reset to the configured stage rate before the scheduler is rebuilt.
            for group in self.optimizer.param_groups:
                group["lr"] = float(self.settings.learning_rate)
            self.scheduler = WarmupCosineScheduler(
                self.optimizer,
                total_steps=self.settings.total_steps,
                warmup_fraction=self.settings.warmup_fraction,
                min_lr_ratio=self.settings.min_lr_ratio,
            )
        else:
            self.scheduler.load_state_dict(payload["scheduler_state"])
        if self.scaler.is_enabled() and payload.get("scaler_state"):
            self.scaler.load_state_dict(payload["scaler_state"])

        self.step_index = int(payload["step"])
        if self.settings.fresh_schedule and self.step_index != self.initial_global_step:
            raise ValueError(
                f"checkpoint global step {self.step_index} does not match the continuation "
                f"initial_global_step {self.initial_global_step}"
            )
        # Stage-local progress: a continuation stage begins at stage step 0 while
        # the global step counter keeps the historical value.
        self.stage_step = (
            0 if self.settings.fresh_schedule else max(0, self.step_index - self.initial_global_step)
        )
        self.tokens_seen = int(payload.get("tokens_seen", 0))
        self.initial_tokens_seen = int(payload.get("initial_tokens_seen", self.tokens_seen))
        self.micro_batches = int(payload.get("micro_batches", 0))
        self.sequences = int(payload.get("sequences", 0))
        self.passes = int(payload.get("passes", 0))
        self.skipped_steps = int(payload.get("skipped_steps", 0))
        self.best_val_loss = float(payload.get("best_val_loss", float("inf")))
        self.best_step = payload.get("best_step")
        self.history = list(payload.get("metrics", []))
        self.validation_history = list(payload.get("validation", []))

        rng_state = payload.get("rng_state")
        if rng_state:
            restore_rng_state(rng_state)
            saved_generator = rng_state.get("loader_generator")
            if saved_generator is not None and self._loader_generator is not None:
                self._loader_generator.set_state(saved_generator)

        if reset_loader_state:
            self._epoch_start_state = None
            self._epoch_batches_consumed = 0
            self._pending_skip = 0
        else:
            self._epoch_start_state = payload.get("loader_epoch_state")
            self._epoch_batches_consumed = int(payload.get("loader_batches_consumed", 0))
            self._pending_skip = self._epoch_batches_consumed
        self._start_time = time.perf_counter()

    # ------------------------------------------------------------------- run
    def run(self) -> dict[str, Any]:
        """Train until ``additional_steps`` successful optimizer updates exist in this stage."""
        settings = self.settings
        stream = self._batch_stream()
        self.model.train()
        self._start_time = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        if settings.initial_validation and self.val_loader is not None:
            initial = self.validate()
            self.log(
                f"step {self.step_index:4d} | initial validation loss {initial['loss']:.4f} "
                f"(ppl {initial['perplexity']:.2f}, {initial['tokens']:,} tokens)"
            )

        # In complete-pass mode the loader is consumed exactly once: the epoch's
        # batches are grouped into full updates plus (possibly) one shorter final
        # group, and the loop stops at the epoch boundary instead of restarting.
        already_consumed = self._pending_skip if settings.complete_pass else 0
        remaining_batches = (
            len(self.train_loader) - already_consumed if settings.complete_pass else None
        )
        if remaining_batches is not None and remaining_batches < 0:
            raise RuntimeError(
                f"checkpoint says {already_consumed} batches were consumed but the loader "
                f"only has {len(self.train_loader)}"
            )

        self.log(f"training from global step {self.step_index} (stage step {self.stage_step}) "
                 f"to {self.expected_final_global_step} ...")
        while self.stage_step < self.additional_steps:
            size = settings.gradient_accumulation_steps
            if remaining_batches is not None:
                if remaining_batches <= 0:
                    break
                size = min(size, remaining_batches)
            batches = [next(stream) for _ in range(size)]
            result = self.train_step(batches)
            if remaining_batches is not None:
                remaining_batches -= size

            if not result["stepped"]:
                self.skipped_steps += 1
                self.log(f"step {self.step_index + 1}: optimizer update skipped (non-finite grads)")
                if self.skipped_steps > settings.max_skipped_steps:
                    raise RuntimeError(
                        f"aborting: {self.skipped_steps} consecutive skipped optimizer updates"
                    )
                continue

            allocated, peak = gpu_memory_mb()
            row: dict[str, Any] = {
                "optimizer_step": self.step_index,
                "epoch_or_pass_fraction": round(self.tokens_seen / self.train_tokens, 6)
                if self.train_tokens
                else "",
                "training_loss": result["loss"],
                "validation_loss": "",
                "learning_rate": result["lr_used"],
                "gradient_norm": result["grad_norm"],
                "tokens_seen": self.tokens_seen,
                "elapsed_seconds": round(time.perf_counter() - self._start_time, 3),
                "allocated_vram": round(allocated, 2),
                "peak_allocated_vram": round(peak, 2),
                "blocks_consumed": self.sequences,
                "effective_batch_size": result["sequences"],
                "tokens_per_step": result["tokens"],
                "precision": self.plan.precision,
                "micro_batch_size": settings.micro_batch_size,
                "gradient_accumulation": settings.gradient_accumulation_steps,
                "max_gpu_memory_mb": round(peak, 2),
            }

            due_eval = (
                self.step_index in settings.eval_steps
                or (settings.eval_every and self.step_index % settings.eval_every == 0)
                or self.stage_step == self.additional_steps
            )
            if self.val_loader is not None and due_eval:
                val = self.validate()
                row["validation_loss"] = val["loss"]
                self.log(
                    f"step {self.step_index:4d} | val loss {val['loss']:.4f} "
                    f"(ppl {val['perplexity']:.2f})"
                )
                if val["loss"] < self.best_val_loss:
                    self.best_val_loss = float(val["loss"])
                    self.best_step = self.step_index
                    if self.checkpoint_dir is not None:
                        self.save(self.checkpoint_dir / "best.pt")

            self.history.append(row)

            due_checkpoint = (
                self.step_index in settings.checkpoint_steps
                or (settings.checkpoint_every and self.step_index % settings.checkpoint_every == 0)
                or self.stage_step == self.additional_steps
            )
            if due_checkpoint and self.checkpoint_dir is not None:
                path = self.checkpoint_dir / f"step_{self.step_index:06d}.pt"
                self.save(path)
                self.save(self.checkpoint_dir / "latest.pt")
                self.log(f"step {self.step_index:4d} | checkpoint {path.name}")

            if result["partial"]:
                self.log(
                    f"step {self.step_index:4d} | final partial update over "
                    f"{result['sequences']} sequences / {result['tokens']:,} tokens "
                    f"({result['micro_batches']} micro-batches)"
                )

            if self.step_index % settings.log_every == 0 or self.stage_step == self.additional_steps:
                progress = (
                    f"{100.0 * self.tokens_seen / self.train_tokens:.2f}% of one pass"
                    if self.train_tokens
                    else "n/a"
                )
                self.log(
                    f"step {self.step_index:4d} | loss {result['loss']:.4f} "
                    f"| lr {result['lr_used']:.3e} "
                    f"| grad_norm {result['grad_norm']:.3f} "
                    f"| tokens {self.tokens_seen:,} ({progress}) "
                    f"| {row['elapsed_seconds']:.1f}s | peak {peak:.0f} MiB"
                )

        if settings.complete_pass:
            if remaining_batches:
                raise RuntimeError(
                    f"complete-pass run stopped with {remaining_batches} batches unconsumed"
                )
            self.passes += 1

        elapsed = time.perf_counter() - self._start_time
        return {
            "steps": self.step_index,
            "stage_step": self.stage_step,
            "initial_global_step": self.initial_global_step,
            "additional_updates": self.stage_step,
            "expected_final_global_step": self.expected_final_global_step,
            "tokens_seen": self.tokens_seen,
            "initial_tokens_seen": self.initial_tokens_seen,
            "stage_tokens_seen": self.tokens_seen - self.initial_tokens_seen,
            "micro_batches": self.micro_batches,
            "sequences": self.sequences,
            "passes": self.passes,
            "elapsed_seconds": elapsed,
            "tokens_per_second": (self.tokens_seen - self.initial_tokens_seen) / elapsed if elapsed > 0 else 0.0,
            "skipped_steps": self.skipped_steps,
            "best_val_loss": self.best_val_loss,
            "best_step": self.best_step,
            "precision": self.plan.precision,
            "effective_batch_size": settings.effective_batch_size,
            "tokens_per_step": settings.tokens_per_step,
            "weights_tied": weights_are_tied(self.model),
        }

    # ------------------------------------------------------------------ misc
    @property
    def rng_state(self) -> dict[str, Any]:
        return capture_rng_state(self._loader_generator)

    def summary(self) -> dict[str, Any]:
        allocated, peak = gpu_memory_mb()
        return {
            "steps": self.step_index,
            "tokens_seen": self.tokens_seen,
            "micro_batches": self.micro_batches,
            "sequences": self.sequences,
            "passes": self.passes,
            "learning_rate": self.scheduler.get_last_lr()[0],
            "precision": self.plan.precision,
            "device": str(self.device),
            "allocated_vram_mb": round(allocated, 2),
            "peak_allocated_vram_mb": round(peak, 2),
            "best_val_loss": self.best_val_loss,
            "best_step": self.best_step,
            "weights_tied": weights_are_tied(self.model),
        }
