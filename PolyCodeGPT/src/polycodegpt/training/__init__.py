"""PolyCodeGPT training package.

Public entry points:

* :class:`TrainSettings` / :class:`Trainer` — the gradient-accumulation loop
* :func:`resolve_precision` — BF16 / FP16 / FP32 selection from device capability
* :func:`build_optimizer` — AdamW with decay / no-decay groups
* :class:`WarmupCosineScheduler` — warmup + cosine learning-rate schedule
* :func:`save_checkpoint` / :func:`load_checkpoint` — resumable checkpoints
* :func:`evaluate_loss` — token-weighted validation loss and perplexity
"""

from .checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    atomic_torch_save,
    capture_rng_state,
    checkpoint_metadata,
    load_checkpoint,
    load_weights_into,
    rebuild_model,
    restore_rng_state,
    rng_states_equal,
    save_checkpoint,
    weights_are_tied,
)
from .engine import (
    METRIC_COLUMNS,
    PRECISIONS,
    PrecisionPlan,
    Trainer,
    TrainSettings,
    apply_determinism,
    gpu_memory_mb,
    make_grad_scaler,
    require_cuda,
    resolve_precision,
    write_metrics_csv,
)
from .evaluate import (
    autocast_context,
    build_validation_loader,
    combine_losses,
    evaluate_loss,
    fixed_validation_indices,
    perplexity,
)
from .optimizer import (
    DECAY_GROUP,
    NO_DECAY_GROUP,
    OptimizerConfig,
    build_optimizer,
    build_parameter_groups,
    parameter_group_report,
    should_decay,
)
from .scheduler import (
    ScheduleConfig,
    WarmupCosineScheduler,
    lr_factor,
    warmup_steps_for,
)

__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "DECAY_GROUP",
    "METRIC_COLUMNS",
    "NO_DECAY_GROUP",
    "PRECISIONS",
    "OptimizerConfig",
    "PrecisionPlan",
    "ScheduleConfig",
    "TrainSettings",
    "Trainer",
    "WarmupCosineScheduler",
    "atomic_torch_save",
    "autocast_context",
    "build_optimizer",
    "build_parameter_groups",
    "build_validation_loader",
    "capture_rng_state",
    "checkpoint_metadata",
    "combine_losses",
    "evaluate_loss",
    "fixed_validation_indices",
    "gpu_memory_mb",
    "load_checkpoint",
    "load_weights_into",
    "lr_factor",
    "make_grad_scaler",
    "parameter_group_report",
    "perplexity",
    "rebuild_model",
    "require_cuda",
    "resolve_precision",
    "restore_rng_state",
    "rng_states_equal",
    "save_checkpoint",
    "should_decay",
    "warmup_steps_for",
    "weights_are_tied",
    "write_metrics_csv",
]
