"""Phase 12B experiment runner: dataset preparation + engine wiring.

Three modes, in increasing order of what they touch:

``--dry-run``
    Validate the YAML, the checkpoint and the budget arithmetic, and print the
    exact plan.  Nothing is written and no training is started.

``--prepare`` (the default for a non-dry-run invocation)
    Materialize the experiment's ``train.bin`` through
    :mod:`polycodegpt.data.experiment_builder`, verify it with the real
    ``TokenBlockDataset``/``DataLoader``, and write ``dataset_manifest.json``.  The
    command prints ``TRAINING NOT STARTED`` and exits; preparing data is never
    reported as training.

``--train --confirm-training``
    Build the *existing* :class:`polycodegpt.training.Trainer` (no second training
    loop) over the prepared dataset, restore checkpoint state from step 8130, and
    run the 8,130-update continuation stage.  Refuses to run if the dataset
    manifest, checkpoint hash or stage accounting do not verify, and refuses to
    write anywhere except the experiment's own output directory.

``--smoke --confirm-smoke``
    One-update runtime preflight on the real control data with the real model,
    optimizer, precision and 4x8 accumulation, but ``additional_steps=1``, no
    validation pass and no checkpoint writer.  Verifies the checkpoint hash before
    and after, finite loss/gradients, changing parameters, restored-and-advanced
    AdamW state, global step 8130 -> 8131 and stage step 0 -> 1, and reports elapsed
    time plus peak CUDA memory.  Never writes a checkpoint and never starts the
    stage.

Nothing here modifies the checkpoint, the original datasets, the tokenizer, or the
validation/test splits; the old test split is never opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

BASE_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "polycodegpt_first_pass" / "step_008130.pt"
BASE_CHECKPOINT_SHA256 = "5562acc2fd58837f9d4245042b7f0b10de65520aa2ba292999c80045470590bd"
BASE_STEP = 8130
ADDITIONAL_UPDATES = 8130
EXPECTED_FINAL_STEP = 16260
ENGLISH_DOC_TOKENS = 2_847_020
TOTAL_TOKEN_BUDGET = 133_192_957
CONTEXT_LENGTH = 512
MICRO_BATCH = 4
GRAD_ACCUM = 8
SEQUENCES_PER_UPDATE = MICRO_BATCH * GRAD_ACCUM
REQUIRED_BLOCKS = (ADDITIONAL_UPDATES - 1) * SEQUENCES_PER_UPDATE + 1  # 8,129 full + 1 partial
VALID_SPLIT = PROJECT_ROOT / "data" / "processed" / "tokenized_v1_1" / "valid.bin"
SMOKE_UPDATES = 1
SMOKE_EXPERIMENT = "control"
PREPARED_ROOT = PROJECT_ROOT / "data" / "processed" / "phase12b_experiments"
EXPERIMENT_OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "phase12b" / "runs"

EXPERIMENT_FILES = {
    "control": PROJECT_ROOT / "configs" / "experiments" / "control.yaml",
    "exp_a": PROJECT_ROOT / "configs" / "experiments" / "exp_a_curated_js_ratio.yaml",
    "exp_b": PROJECT_ROOT / "configs" / "experiments" / "exp_b_language_balance.yaml",
    "exp_c": PROJECT_ROOT / "configs" / "experiments" / "exp_c_curated_balanced.yaml",
}

ALLOWED_TOP_KEYS = {"experiment", "data", "training", "schedule", "checkpointing", "evaluation", "constraints"}
REQUIRED_TOP_KEYS = {"experiment", "data", "training", "schedule", "checkpointing"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# --------------------------------------------------------------------------- #
# resolved plan
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Resolved:
    """Everything derived from one experiment YAML, with all invariants checked."""

    experiment_id: str
    config_path: Path
    seed: int
    base_checkpoint: Path
    base_checkpoint_sha256: str
    base_step: int

    python_tokens: int
    javascript_tokens: int
    english_tokens: int
    total_tokens: int
    js_source: str
    oversampling: str
    sampling: str

    learning_rate: float
    weight_decay: float
    gradient_clip: float
    micro_batch_size: int
    gradient_accumulation_steps: int
    context_length: int
    precision: str
    deterministic: bool
    config_max_steps: int
    python_seed: int
    torch_seed: int
    cuda_seed: int
    dataloader_worker_seed: int
    dataloader_num_workers: int
    deterministic_algorithms: bool
    cudnn_deterministic: bool
    cudnn_benchmark: bool

    stage_steps: int
    schedule_total_steps: int
    warmup_steps: int
    min_lr_ratio: float
    reset_schedule: bool
    reset_optimizer_step: bool
    checkpoint_steps: tuple[int, ...]
    eval_steps: tuple[int, ...]

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps

    @property
    def tokens_per_step(self) -> int:
        return self.effective_batch_size * self.context_length

    @property
    def output_dir(self) -> Path:
        return EXPERIMENT_OUTPUT_ROOT / self.experiment_id

    @property
    def dataset_dir(self) -> Path:
        return PREPARED_ROOT / self.experiment_id

    @property
    def dataset_manifest_path(self) -> Path:
        return self.dataset_dir / "dataset_manifest.json"

    @property
    def train_bin(self) -> Path:
        return self.dataset_dir / "train.bin"


def lr_at(step: int, base: float, warmup: int, total: int, min_ratio: float) -> float:
    if warmup > 0 and step < warmup:
        return base * (step + 1) / warmup
    span = max(1, total - warmup)
    progress = min(1.0, max(0.0, (step - warmup + 1) / span))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base * (min_ratio + (1.0 - min_ratio) * cosine)


def resolve(config_path: Path) -> Resolved:
    """Load one experiment YAML and verify every invariant it claims."""
    cfg = load_yaml(config_path)
    unknown = set(cfg) - ALLOWED_TOP_KEYS
    if unknown:
        raise SystemExit(f"unknown/unsupported top-level keys: {sorted(unknown)}")
    missing = REQUIRED_TOP_KEYS - set(cfg)
    if missing:
        raise SystemExit(f"missing required top-level keys: {sorted(missing)}")

    exp = cfg["experiment"]
    data = cfg["data"]
    training = cfg["training"]
    schedule = cfg["schedule"]

    exp_id = exp["id"]
    seed = int(exp.get("seed", 42))
    base_ckpt = PROJECT_ROOT / exp["base_checkpoint"]
    base_sha = exp["base_checkpoint_sha256"]
    base_step = int(exp["base_checkpoint_step"])

    if not base_ckpt.is_file():
        raise SystemExit(f"base checkpoint not found: {base_ckpt}")
    actual_sha = sha256_file(base_ckpt)
    if actual_sha != base_sha or actual_sha != BASE_CHECKPOINT_SHA256:
        raise SystemExit(f"checkpoint sha mismatch: {actual_sha}")
    if base_step != BASE_STEP:
        raise SystemExit(f"checkpoint step {base_step} != {BASE_STEP}")

    tb = data["token_budget_per_language"]
    py_tokens = int(tb["python"])
    js_tokens = int(tb["javascript"])
    en_tokens = int(data["english_doc_tokens"])
    total = int(data["token_budget_total"])
    if en_tokens != ENGLISH_DOC_TOKENS:
        raise SystemExit(f"english doc tokens {en_tokens} != {ENGLISH_DOC_TOKENS}")
    if total != TOTAL_TOKEN_BUDGET:
        raise SystemExit(f"total token budget {total} != {TOTAL_TOKEN_BUDGET}")
    if py_tokens + js_tokens + en_tokens != total:
        raise SystemExit(f"budget sum {py_tokens + js_tokens + en_tokens} != {total}")

    lr = float(training["learning_rate"])
    micro_batch = int(training["micro_batch_size"])
    grad_accum = int(training["gradient_accumulation_steps"])
    context = int(training["context_length"])
    if micro_batch != MICRO_BATCH or grad_accum != GRAD_ACCUM or context != CONTEXT_LENGTH:
        raise SystemExit(
            f"batch geometry {micro_batch}x{grad_accum}@{context} != "
            f"{MICRO_BATCH}x{GRAD_ACCUM}@{CONTEXT_LENGTH}"
        )
    if int(training.get("effective_batch_size", SEQUENCES_PER_UPDATE)) != SEQUENCES_PER_UPDATE:
        raise SystemExit("declared effective batch size disagrees with micro x accum")
    if int(training.get("tokens_per_step", SEQUENCES_PER_UPDATE * CONTEXT_LENGTH)) != SEQUENCES_PER_UPDATE * CONTEXT_LENGTH:
        raise SystemExit("declared tokens_per_step disagrees with effective batch x context")

    config_max_steps = int(training["max_steps"])
    additional = int(training["additional_steps"])
    total_sched = int(schedule["total_steps"])
    warmup = int(schedule["warmup_steps"])

    # ``max_steps`` in the experiment YAML is the *final global step*; the engine's
    # own stage length must equal ``additional_steps`` and stay within the schedule.
    if config_max_steps != base_step + additional:
        raise SystemExit(
            f"max_steps {config_max_steps} != base step {base_step} + additional {additional}"
        )
    if additional != ADDITIONAL_UPDATES or config_max_steps != EXPECTED_FINAL_STEP:
        raise SystemExit("additional_steps / max_steps disagree with the frozen stage plan")
    if total_sched != additional:
        raise SystemExit(f"schedule total_steps {total_sched} != additional_steps {additional}")
    if not (0 < additional <= total_sched):
        raise SystemExit("engine constraint max_steps <= total_steps would be violated")
    if int(round(total_sched * float(schedule["warmup_fraction"]))) != warmup:
        raise SystemExit("declared warmup_steps disagrees with warmup_fraction x total_steps")

    js_source = data.get("curated_filter", "none")
    js_source = "original" if js_source in ("none", None) else "curated"

    return Resolved(
        experiment_id=exp_id,
        config_path=config_path,
        seed=seed,
        base_checkpoint=base_ckpt,
        base_checkpoint_sha256=actual_sha,
        base_step=base_step,
        python_tokens=py_tokens,
        javascript_tokens=js_tokens,
        english_tokens=en_tokens,
        total_tokens=total,
        js_source=js_source,
        oversampling=str(data.get("oversampling", data.get("javascript_sampling", "none"))),
        sampling=str(data.get("sampling", data.get("python_sampling", "original_distribution"))),
        learning_rate=lr,
        weight_decay=float(training["weight_decay"]),
        gradient_clip=float(training["gradient_clip"]),
        micro_batch_size=micro_batch,
        gradient_accumulation_steps=grad_accum,
        context_length=context,
        precision=str(training.get("precision", "auto")),
        deterministic=bool(training.get("deterministic", True)),
        config_max_steps=config_max_steps,
        python_seed=int(training.get("python_seed", seed)),
        torch_seed=int(training.get("torch_seed", seed)),
        cuda_seed=int(training.get("cuda_seed", seed)),
        dataloader_worker_seed=int(training.get("dataloader_worker_seed", seed)),
        dataloader_num_workers=int(training.get("dataloader_num_workers", 0)),
        deterministic_algorithms=bool(training.get("deterministic_algorithms", True)),
        cudnn_deterministic=bool(training.get("cudnn_deterministic", True)),
        cudnn_benchmark=bool(training.get("cudnn_benchmark", False)),
        stage_steps=additional,
        schedule_total_steps=total_sched,
        warmup_steps=warmup,
        min_lr_ratio=float(schedule["min_lr_ratio"]),
        reset_schedule=bool(schedule["reset_schedule"]),
        reset_optimizer_step=bool(schedule["reset_optimizer_step"]),
        checkpoint_steps=tuple(int(x) for x in cfg["checkpointing"].get("checkpoint_steps", ())),
        eval_steps=tuple(
            int(x) for x in cfg.get("evaluation", {}).get("eval_steps", ())
        ),
    )


# --------------------------------------------------------------------------- #
# preparation
# --------------------------------------------------------------------------- #
def prepare_dataset(plan: Resolved, seed: int | None = None) -> dict[str, Any]:
    """Materialize and verify this experiment's dataset; return its manifest."""
    from polycodegpt.data.experiment_builder import (
        EXPERIMENTS,
        build_experiment_dataset,
        stream_cache,
        verify_with_dataloader,
    )

    sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "phase12b"))
    from build_dataset import prepare_streams  # noqa: WPS433

    from polycodegpt.model.config import ModelConfig

    config = ModelConfig.from_yaml()
    streams = prepare_streams(config)
    spec = EXPERIMENTS[plan.experiment_id]
    manifest = build_experiment_dataset(
        spec, plan.dataset_dir, streams, config.vocab_size, config.context_length,
        seed=plan.seed if seed is None else seed,
    )
    report = verify_with_dataloader(
        plan.train_bin,
        context_length=plan.context_length,
        micro_batch_size=plan.micro_batch_size,
        gradient_accumulation_steps=plan.gradient_accumulation_steps,
        vocab_size=config.vocab_size,
        seed=plan.seed,
    )
    manifest["dataloader_verification"] = report
    manifest["runner"] = {
        "base_checkpoint": str(plan.base_checkpoint),
        "base_checkpoint_sha256": plan.base_checkpoint_sha256,
        "initial_global_step": plan.base_step,
        "additional_optimizer_updates": plan.stage_steps,
        "expected_final_global_step": plan.base_step + plan.stage_steps,
        "prepared_by": "scripts/run_phase12_experiment.py",
    }
    plan.dataset_manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    # fail closed: the artifact must support exactly the frozen stage plan
    if report["blocks"] < REQUIRED_BLOCKS:
        raise SystemExit(
            f"dataset has {report['blocks']:,} blocks, need at least {REQUIRED_BLOCKS:,} "
            f"for {plan.stage_steps:,} updates"
        )
    if report["total_updates_one_pass"] != plan.stage_steps:
        raise SystemExit(
            f"one pass yields {report['total_updates_one_pass']:,} updates, "
            f"stage plan needs {plan.stage_steps:,}"
        )
    return manifest


def verify_prepared_dataset(plan: Resolved) -> dict[str, Any]:
    """Re-verify an existing prepared dataset before any training attempt."""
    if not plan.dataset_manifest_path.is_file() or not plan.train_bin.is_file():
        raise SystemExit(
            f"dataset not prepared: run --prepare for {plan.experiment_id} first "
            f"({plan.dataset_manifest_path})"
        )
    manifest = json.loads(plan.dataset_manifest_path.read_text(encoding="utf-8"))
    actual = sha256_file(plan.train_bin)
    if actual != manifest["train_bin"]["sha256"]:
        raise SystemExit(f"train.bin sha mismatch: {actual} != {manifest['train_bin']['sha256']}")
    report = manifest["dataloader_verification"]
    if report["total_updates_one_pass"] != plan.stage_steps:
        raise SystemExit("prepared dataset no longer yields the planned update count")
    if report["blocks"] < REQUIRED_BLOCKS:
        raise SystemExit("prepared dataset has too few blocks for the stage")
    return manifest


# --------------------------------------------------------------------------- #
# engine wiring
# --------------------------------------------------------------------------- #
def build_trainer(plan: Resolved, *, device: str | None = None, log_fn=print, smoke: bool = False,
                  reset_loader_state: bool = False):
    """Construct the project's own Trainer for this experiment.

    This is deliberately thin: model reconstruction, optimizer, scheduler, restore,
    validation loader and metrics all come from :mod:`polycodegpt.training`.  No
    training loop is reimplemented here.

    ``smoke=True`` changes only what makes the probe single-step and write-free:
    the stage length becomes one update, the epoch-completeness requirement is
    dropped (one update cannot consume a whole pass) and no checkpoint directory or
    validation loader is attached.  Model, optimizer, precision, batch geometry and
    the 8,130-step schedule shape stay exactly as the real run uses them.
    """
    stage_steps = SMOKE_UPDATES if smoke else plan.stage_steps
    import torch

    from polycodegpt.data import TokenBlockDataset, build_dataloader
    from polycodegpt.model.config import ModelConfig
    from polycodegpt.training import (
        Trainer,
        TrainSettings,
        build_validation_loader,
        fixed_validation_indices,
        load_checkpoint,
        rebuild_model,
    )

    verify_prepared_dataset(plan)

    manifest = load_checkpoint(plan.base_checkpoint, map_location="cpu")
    if sha256_file(plan.base_checkpoint) != BASE_CHECKPOINT_SHA256:
        raise SystemExit("base checkpoint changed during setup; refusing to continue")
    if int(manifest["step"]) != plan.base_step:
        raise SystemExit(f"checkpoint step {manifest['step']} != {plan.base_step}")

    model_config = ModelConfig.from_yaml()
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    precision = plan.precision
    if resolved_device.type != "cuda" and precision.lower() != "fp32":
        log_fn(f"precision {precision!r} is CUDA-only; using fp32 on {resolved_device.type}")
        precision = "fp32"
    model = rebuild_model(manifest, device=resolved_device)

    settings = TrainSettings(
        micro_batch_size=plan.micro_batch_size,
        gradient_accumulation_steps=plan.gradient_accumulation_steps,
        max_steps=stage_steps,
        total_steps=plan.schedule_total_steps,
        warmup_fraction=plan.warmup_steps / plan.schedule_total_steps,
        min_lr_ratio=plan.min_lr_ratio,
        learning_rate=plan.learning_rate,
        weight_decay=plan.weight_decay,
        gradient_clip=plan.gradient_clip,
        precision=precision,
        seed=plan.seed,
        context_length=plan.context_length,
        eval_steps=plan.eval_steps,
        checkpoint_steps=plan.checkpoint_steps,
        initial_validation=False,
        complete_pass=not smoke,
        partial_final_update=True,
        initial_global_step=plan.base_step,
        additional_steps=stage_steps,
        fresh_schedule=plan.reset_schedule,
        deterministic=plan.deterministic and plan.deterministic_algorithms,
        python_seed=plan.python_seed,
        torch_seed=plan.torch_seed,
        cuda_seed=plan.cuda_seed,
        dataloader_worker_seed=plan.dataloader_worker_seed,
        cudnn_deterministic=plan.cudnn_deterministic,
        cudnn_benchmark=plan.cudnn_benchmark,
    )

    train_dataset = TokenBlockDataset(
        plan.train_bin, context_length=plan.context_length, vocab_size=model_config.vocab_size
    )
    train_loader = build_dataloader(
        train_dataset, batch_size=plan.micro_batch_size, shuffle=True,
        num_workers=plan.dataloader_num_workers, seed=plan.dataloader_worker_seed,
        pin_memory=(resolved_device.type == "cuda"),
    )

    val_loader = None
    if not smoke and VALID_SPLIT.is_file():
        valid_dataset = TokenBlockDataset(
            VALID_SPLIT, context_length=plan.context_length, vocab_size=model_config.vocab_size
        )
        indices = fixed_validation_indices(len(valid_dataset), min(512, len(valid_dataset)),
                                          seed=plan.seed)
        val_loader = build_validation_loader(valid_dataset, indices, batch_size=plan.micro_batch_size)

    trainer = Trainer(
        model,
        train_loader,
        settings,
        device=resolved_device,
        val_loader=val_loader,
        train_tokens=train_dataset.token_count,
        checkpoint_dir=None if smoke else plan.output_dir,
        model_config=model_config,
        tokenizer_sha256=manifest.get("tokenizer_sha256", ""),
        data_manifest_sha256=str(manifest.get("data_manifest_sha256", "")),
        log_fn=log_fn,
    )
    trainer.restore(manifest, reset_loader_state=reset_loader_state)
    return trainer, manifest


def _snapshot_parameters(model) -> dict[str, Any]:
    return {name: param.detach().to("cpu").clone() for name, param in model.named_parameters()}


def _optimizer_step_counters(optimizer) -> list[int]:
    state = optimizer.state_dict()["state"]
    return sorted({int(entry["step"]) for entry in state.values()})


def run_smoke_test(plan: Resolved, *, device: str | None = None, log_fn=print) -> dict[str, Any]:
    """One real optimizer update on real data; writes nothing, proves the runtime works."""
    import time

    import torch

    if plan.experiment_id != SMOKE_EXPERIMENT:
        raise SystemExit(
            f"the smoke test uses the real {SMOKE_EXPERIMENT} training data; "
            f"got experiment {plan.experiment_id!r}"
        )

    print_plan(plan, "SMOKE (1 optimizer update)")

    checkpoint_dir = plan.base_checkpoint.parent
    sha_before = sha256_file(plan.base_checkpoint)
    if sha_before != BASE_CHECKPOINT_SHA256:
        raise SystemExit(f"SMOKE FAILED: checkpoint sha256 before run is {sha_before}")
    files_before = sorted(p.name for p in checkpoint_dir.iterdir() if p.is_file())
    output_dir = plan.output_dir
    output_files_before = (
        sorted(p.name for p in output_dir.iterdir() if p.is_file()) if output_dir.is_dir() else []
    )
    print(f"checkpoint sha256 before: {sha_before}")

    trainer, _ = build_trainer(plan, device=device, log_fn=log_fn, smoke=True, reset_loader_state=True)
    if trainer.additional_steps != SMOKE_UPDATES or trainer.stage_step != 0:
        raise SystemExit(
            f"SMOKE FAILED: stage is {trainer.stage_step}/{trainer.additional_steps}, "
            f"expected 0/{SMOKE_UPDATES}"
        )

    params_before = _snapshot_parameters(trainer.model)
    steps_before = _optimizer_step_counters(trainer.optimizer)
    sample_before = {
        key: (
            entry["exp_avg"].detach().cpu().clone(),
            entry["exp_avg_sq"].detach().cpu().clone(),
        )
        for key, entry in list(trainer.optimizer.state_dict()["state"].items())[:4]
    }
    lr_before = trainer.scheduler.get_last_lr()[0]
    # tokens_seen / micro_batches / sequences / history are cumulative for the whole
    # project and are restored from the checkpoint, so the probe compares deltas.
    tokens_before = trainer.tokens_seen
    micro_before = trainer.micro_batches
    sequences_before = trainer.sequences
    history_before = len(trainer.history)
    print(f"restored: global step {trainer.step_index}, stage step {trainer.stage_step}, "
          f"adamw step counters {steps_before}, scheduler last_step {trainer.scheduler.last_step}, "
          f"lr {lr_before:.6e}")
    print(f"restored counters: tokens_seen {tokens_before:,}, micro_batches {micro_before:,}, "
          f"sequences {sequences_before:,}, history rows {history_before:,}")

    cuda = trainer.device.type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    try:
        result = trainer.run()
    except torch.cuda.OutOfMemoryError as exc:
        raise SystemExit(f"SMOKE FAILED: CUDA out of memory: {exc}") from exc
    except FloatingPointError as exc:
        raise SystemExit(f"SMOKE FAILED: non-finite value during the update: {exc}") from exc
    except RuntimeError as exc:
        message = str(exc)
        if "CUDA" in message or "cuda" in message or "device-side" in message:
            raise SystemExit(f"SMOKE FAILED: CUDA error: {message}") from exc
        raise
    if cuda:
        torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - started

    params_after = _snapshot_parameters(trainer.model)
    changed = [
        name
        for name, before in params_before.items()
        if not torch.equal(before, params_after[name])
    ]
    max_delta = 0.0
    for name, before in params_before.items():
        delta = (params_after[name] - before).abs().max().item()
        max_delta = max(max_delta, delta)
    steps_after = _optimizer_step_counters(trainer.optimizer)
    state_after = trainer.optimizer.state_dict()["state"]
    sample_after = {
        key: (state_after[key]["exp_avg"].detach().cpu(),
              state_after[key]["exp_avg_sq"].detach().cpu())
        for key in sample_before
    }
    moments_changed = all(
        not torch.equal(sample_before[key][0], sample_after[key][0])
        or not torch.equal(sample_before[key][1], sample_after[key][1])
        for key in sample_before
    )

    sha_after = sha256_file(plan.base_checkpoint)
    files_after = sorted(p.name for p in checkpoint_dir.iterdir() if p.is_file())
    output_files_after = (
        sorted(p.name for p in output_dir.iterdir() if p.is_file()) if output_dir.is_dir() else []
    )
    row = trainer.history[-1] if trainer.history else {}
    peak_allocated = float(row.get("peak_allocated_vram", 0.0))
    peak_reserved = (
        torch.cuda.max_memory_reserved() / 1024**2 if cuda else 0.0
    )

    tokens_update = trainer.tokens_seen - tokens_before
    micro_update = trainer.micro_batches - micro_before
    sequences_update = trainer.sequences - sequences_before
    history_update = len(trainer.history) - history_before

    checks = {
        "checkpoint_sha_unchanged": sha_before == sha_after == BASE_CHECKPOINT_SHA256,
        "checkpoint_dir_unchanged": files_before == files_after,
        "output_dir_unchanged": output_files_before == output_files_after,
        "exactly_one_update": history_update == SMOKE_UPDATES,
        "loss_finite": bool(row) and math.isfinite(row["training_loss"]) and row["training_loss"] > 0,
        "grad_norm_finite": bool(row) and math.isfinite(row["gradient_norm"]) and row["gradient_norm"] > 0,
        "parameters_updated": len(changed) == len(params_before) and max_delta > 0,
        "optimizer_state_restored": steps_before == [BASE_STEP],
        "optimizer_state_advanced": steps_after == [BASE_STEP + 1],
        "adamw_moments_updated": moments_changed,
        "global_step_advanced": trainer.step_index == BASE_STEP + 1 and result["steps"] == BASE_STEP + 1,
        "stage_step_advanced": trainer.stage_step == 1 and result["stage_step"] == 1,
        "accumulation_used": micro_update == GRAD_ACCUM
        and sequences_update == SEQUENCES_PER_UPDATE
        and tokens_update == SEQUENCES_PER_UPDATE * CONTEXT_LENGTH
        and row.get("effective_batch_size") == SEQUENCES_PER_UPDATE
        and row.get("tokens_per_step") == SEQUENCES_PER_UPDATE * CONTEXT_LENGTH,
        "scheduler_advanced": trainer.scheduler.last_step == 1,
    }
    failed = [name for name, passed in checks.items() if not passed]

    report = {
        "mode": "smoke",
        "experiment": plan.experiment_id,
        "device": str(trainer.device),
        "precision": trainer.plan.precision,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "checkpoint": str(plan.base_checkpoint),
        "checkpoint_sha256_before": sha_before,
        "checkpoint_sha256_after": sha_after,
        "dataset": str(plan.train_bin),
        "dataset_sha256": sha256_file(plan.train_bin),
        "updates": 1,
        "micro_batches_this_update": micro_update,
        "sequences_this_update": sequences_update,
        "tokens_this_update": tokens_update,
        "tokens_seen_cumulative": trainer.tokens_seen,
        "training_loss": row.get("training_loss"),
        "gradient_norm": row.get("gradient_norm"),
        "learning_rate_used": row.get("learning_rate"),
        "parameters_changed": len(changed),
        "parameters_total": len(params_before),
        "max_parameter_delta": max_delta,
        "optimizer_step_before": steps_before,
        "optimizer_step_after": steps_after,
        "global_step_before": BASE_STEP,
        "global_step_after": trainer.step_index,
        "stage_step_after": trainer.stage_step,
        "elapsed_seconds": round(wall_seconds, 3),
        "engine_elapsed_seconds": round(float(result["elapsed_seconds"]), 3),
        "tokens_per_second": round(tokens_update / wall_seconds, 1) if wall_seconds else 0.0,
        "peak_vram_allocated_mb": round(peak_allocated, 1),
        "peak_vram_reserved_mb": round(peak_reserved, 1),
        "output_dir_unchanged": output_files_before == output_files_after,
        "checkpoint_written": False,
        "checks": checks,
        "failed_checks": failed,
        "passed": not failed,
    }

    print("-" * 72)
    print(f"device/precision:   {report['device']} / {report['precision']}  "
          f"({report['gpu'] or 'no CUDA device in use'})")
    print(f"update:             micro_batches={micro_update} "
          f"sequences={sequences_update} tokens={tokens_update:,}")
    print(f"loss / grad_norm:   {report['training_loss']:.6f} / {report['gradient_norm']:.4f}")
    print(f"lr used:            {report['learning_rate_used']:.6e}")
    print(f"parameters changed: {report['parameters_changed']}/{report['parameters_total']} "
          f"(max |delta| {max_delta:.3e})")
    print(f"optimizer step:     {steps_before} -> {steps_after}")
    print(f"global/stage step:  {BASE_STEP} -> {trainer.step_index} / 0 -> {trainer.stage_step}")
    print(f"elapsed:            {wall_seconds:.2f}s  "
          f"({report['tokens_per_second']:,.0f} tokens/s)")
    print(f"peak VRAM:          {peak_allocated:,.1f} MiB allocated, "
          f"{peak_reserved:,.1f} MiB reserved")
    print(f"checkpoint sha:     before {sha_before}")
    print(f"                    after  {sha_after}")
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print("-" * 72)
    if failed:
        raise SystemExit(f"SMOKE FAILED: {failed}")
    print("SMOKE PASSED: one production update verified; no checkpoint written, "
          "full stage NOT started.")
    print("SMOKE REPORT JSON:")
    print(json.dumps(report, indent=2))
    return report


def assert_output_paths_are_safe(plan: Resolved) -> None:
    """Training output must be a fresh directory, never a source location."""
    protected = [
        plan.base_checkpoint.resolve(),
        PROJECT_ROOT / "data" / "processed" / "tokenized_v1_1",
        PROJECT_ROOT / "data" / "processed" / "polycodegpt_corpus",
        PROJECT_ROOT / "data" / "curated",
        PROJECT_ROOT / "tokenizer",
        PROJECT_ROOT / "data" / "processed" / "phase12b_work",
    ]
    out = plan.output_dir.resolve()
    for path in protected:
        path = path.resolve()
        if out == path or path in out.parents:
            raise SystemExit(f"refusing to write training output inside protected path {path}")
    if out == plan.dataset_dir.resolve():
        raise SystemExit("training output must not overwrite the prepared dataset directory")


# --------------------------------------------------------------------------- #
# reporting / CLI
# --------------------------------------------------------------------------- #
def print_plan(plan: Resolved, mode: str) -> None:
    first5 = [lr_at(s, plan.learning_rate, plan.warmup_steps, plan.schedule_total_steps,
                    plan.min_lr_ratio) for s in range(5)]
    last5 = [lr_at(s, plan.learning_rate, plan.warmup_steps, plan.schedule_total_steps,
                   plan.min_lr_ratio) for s in range(plan.schedule_total_steps - 5, plan.schedule_total_steps)]
    print("=" * 72)
    print(f"EXPERIMENT: {plan.experiment_id}   [{mode}]")
    print("=" * 72)
    print(f"config:             {plan.config_path}")
    print(f"base checkpoint:    {plan.base_checkpoint}")
    print(f"                    step={plan.base_step}  sha256={plan.base_checkpoint_sha256[:16]}...")
    print(f"initial global step:{plan.base_step}")
    print(f"stage length:       {plan.stage_steps:,} updates (max_steps <= total_steps: "
          f"{plan.stage_steps} <= {plan.schedule_total_steps})")
    print(f"expected final step:{plan.base_step + plan.stage_steps:,} (global)")
    print(f"schedule:           fresh={plan.reset_schedule} warmup={plan.warmup_steps} "
          f"total={plan.schedule_total_steps} min_lr_ratio={plan.min_lr_ratio}")
    print(f"lr base / first5:   {plan.learning_rate}  {['%.6e' % x for x in first5]}")
    print(f"lr last5:           {['%.6e' % x for x in last5]}")
    print(f"batch:              micro={plan.micro_batch_size} x accum="
          f"{plan.gradient_accumulation_steps} = {plan.effective_batch_size} sequences "
          f"({plan.tokens_per_step:,} tokens/update)")
    print(f"determinism:        seed={plan.seed} deterministic={plan.deterministic} "
          f"algorithms={plan.deterministic_algorithms} "
          f"seeds(py/torch/cuda/loader)="
          f"{plan.python_seed}/{plan.torch_seed}/{plan.cuda_seed}/{plan.dataloader_worker_seed} "
          f"workers={plan.dataloader_num_workers}"
          f" cudnn(det/bench)={plan.cudnn_deterministic}/{plan.cudnn_benchmark}")
    print(f"token budgets:      python={plan.python_tokens:,} javascript={plan.javascript_tokens:,} "
          f"english={plan.english_tokens:,} total={plan.total_tokens:,}")
    print(f"javascript source:  {plan.js_source} (oversampling={plan.oversampling}, "
          f"python sampling={plan.sampling})")
    print(f"dataset dir:        {plan.dataset_dir}")
    print(f"training output:    {plan.output_dir}")
    print("=" * 72)


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 12B experiment runner")
    ap.add_argument("--experiment", choices=sorted(EXPERIMENT_FILES), default=None)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--confirm-training", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="one-update runtime preflight on the control dataset")
    ap.add_argument("--confirm-smoke", action="store_true",
                    help="required together with --smoke")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if args.config is not None:
        config_path = args.config
    elif args.experiment is not None:
        config_path = EXPERIMENT_FILES[args.experiment]
    else:
        raise SystemExit("provide --experiment or --config")
    if not config_path.is_file():
        raise SystemExit(f"config not found: {config_path}")

    plan = resolve(config_path)

    if args.dry_run:
        print_plan(plan, "DRY RUN")
        print("DRY RUN: no files written, no training started.")
        return

    if args.smoke:
        if not args.confirm_smoke:
            raise SystemExit(
                "refusing to run the smoke test: pass --confirm-smoke. It performs exactly "
                "one real optimizer update on GPU and writes no checkpoint."
            )
        if args.train:
            raise SystemExit("choose either --smoke or --train, not both")
        assert_output_paths_are_safe(plan)
        run_smoke_test(plan, device=args.device)
        return

    if args.train:
        if not args.confirm_training:
            raise SystemExit(
                "refusing to train: pass --confirm-training. Preparation is the default "
                "non-dry-run action because Phase 12C runs are not authorized yet."
            )
        assert_output_paths_are_safe(plan)
        print_plan(plan, "TRAIN")
        trainer, payload = build_trainer(plan, device=args.device, reset_loader_state=True)
        print(f"restored: global step {trainer.step_index}, stage step {trainer.stage_step}, "
              f"scheduler last_step {trainer.scheduler.last_step}, "
              f"adamw step counters {sorted({int(s['step']) for s in trainer.optimizer.state_dict()['state'].values()})}")
        plan.output_dir.mkdir(parents=True, exist_ok=True)
        result = trainer.run()
        (plan.output_dir / "run_summary.json").write_text(json.dumps(result, indent=2) + "\n",
                                                          encoding="utf-8")
        print(json.dumps(result, indent=2))
        return

    # default non-dry-run action: prepare data only
    print_plan(plan, "PREPARE")
    manifest = prepare_dataset(plan)
    report = manifest["dataloader_verification"]
    print(f"materialized:       {manifest['train_bin']['tokens']:,} tokens  "
          f"sha256={manifest['train_bin']['sha256'][:16]}...")
    print(f"blocks/updates:     {report['blocks']:,} blocks -> {report['full_updates']:,} full + "
          f"{report['final_partial_update_sequences']} sequences final partial "
          f"= {report['total_updates_one_pass']:,} updates")
    print(f"manifest:           {plan.dataset_manifest_path}")
    print("TRAINING NOT STARTED: data preparation only.")


if __name__ == "__main__":
    main()
