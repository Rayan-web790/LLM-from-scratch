import argparse
import hashlib
import json
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from polycodegpt.data.scenario_c_dataset import (
    ScenarioCCompletionDataset,
    scenario_c_collate,
)
from polycodegpt.model.config import ModelConfig
from polycodegpt.training.checkpoint import load_checkpoint, rebuild_model
from polycodegpt.training.engine import Trainer, TrainSettings


ROOT = Path(__file__).resolve().parents[1]

BASELINE = ROOT / "checkpoints/polycodegpt_phase20_scenario_b/best.pt"
TOKENIZER = ROOT / "tokenizer/artifacts/v2/polycodegpt_tokenizer_v2.json"
DATA_DIR = ROOT / "data/interim/scenario_c"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()



def main():
    parser = argparse.ArgumentParser(
        description="Train or resume PolyCodeGPT Scenario C."
    )
    parser.add_argument("--mode", choices=["fresh", "resume"], default="fresh")
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--steps", type=int, default=1000,
                        help="Total target Scenario C stage updates.")
    parser.add_argument("--micro-batch", type=int, default=16)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--total-steps", type=int, default=1000)
    parser.add_argument("--validation-records", type=int, default=512)
    parser.add_argument("--validation-batch", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument(
        "--output-dir",
        default="checkpoints/polycodegpt_scenario_c_completion",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Scenario C training.")

    if args.mode == "resume" and not args.resume_from:
        raise ValueError("--resume-from is required when --mode resume.")

    if args.mode == "fresh" and args.resume_from:
        raise ValueError("--resume-from can only be used with --mode resume.")

    if min(args.steps, args.micro_batch, args.accumulation,
           args.total_steps) <= 0:
        raise ValueError("Steps, batch sizes and total-steps must be positive.")

    if args.steps > args.total_steps:
        raise ValueError("--steps cannot exceed --total-steps.")

    if args.validation_records <= 0 or args.validation_batch <= 0:
        raise ValueError("Validation records and batch size must be positive.")

    if args.eval_every < 0 or args.checkpoint_every < 0:
        raise ValueError("Evaluation/checkpoint intervals cannot be negative.")

    output_dir = (ROOT / args.output_dir).resolve()
    resume_path = (
        (ROOT / args.resume_from).resolve()
        if args.resume_from else None
    )

    if output_dir.exists():
        if args.mode != "resume" or resume_path.parent != output_dir:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                "For resume, use the same directory as the checkpoint, "
                "or choose a new output directory."
            )

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    print("========== SCENARIO C TRAINING ==========")
    print("Mode:", args.mode)
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM GB:", round(
        torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
    ))

    if args.mode == "fresh":
        print("\nLoading frozen Scenario B checkpoint...")
        payload = load_checkpoint(BASELINE, map_location="cpu")
    else:
        print("\nLoading Scenario C resume checkpoint...")
        payload = load_checkpoint(resume_path, map_location="cpu")

    config = ModelConfig.from_dict(payload["model_config"])
    model = rebuild_model(payload, device="cpu")

    if args.mode == "fresh":
        # Scenario C begins immediately after the frozen Scenario B checkpoint.
        initial_global_step = int(payload["step"])
        restored_stage_step = 0
    else:
        # Preserve the original Scenario C stage origin when resuming.
        initial_global_step = int(
            payload.get("initial_global_step", payload["step"])
        )
        restored_stage_step = int(payload.get("stage_step", 0))

    if int(payload["step"]) != initial_global_step + restored_stage_step:
        raise RuntimeError(
            "Checkpoint step accounting is inconsistent: "
            f"global={payload['step']}, initial={initial_global_step}, "
            f"stage={restored_stage_step}."
        )

    if args.mode == "resume":
        if restored_stage_step >= args.steps:
            raise ValueError(
                f"Checkpoint already completed {restored_stage_step} stage "
                f"updates; requested target is {args.steps}."
            )

        scheduler_state = payload.get("scheduler_state", {})
        saved_total_steps = scheduler_state.get("total_steps")
        if saved_total_steps != args.total_steps:
            raise ValueError(
                f"Scheduler horizon mismatch: checkpoint={saved_total_steps}, "
                f"requested={args.total_steps}. Resume with the original "
                "--total-steps value."
            )

        saved_settings = payload.get("settings", {})
        expected_settings = {
            "micro_batch_size": args.micro_batch,
            "gradient_accumulation_steps": args.accumulation,
            "learning_rate": args.learning_rate,
        }

        for key, requested in expected_settings.items():
            saved = saved_settings.get(key)
            if saved is None or saved != requested:
                raise ValueError(
                    f"Resume setting mismatch for {key}: "
                    f"checkpoint={saved}, requested={requested}."
                )

    print("Checkpoint global step:", payload["step"])
    print("Scenario C stage progress:", restored_stage_step, "/", args.steps)
    print("Stage starting global step:", initial_global_step)
    print("Model parameters:", f"{config.estimate_parameters():,}")

    print("\nVerifying tokenizer...")
    tokenizer_hash = sha256_file(TOKENIZER)

    if tokenizer_hash != payload["tokenizer_sha256"]:
        raise RuntimeError("Tokenizer SHA-256 does not match checkpoint.")

    print("Tokenizer SHA-256:", tokenizer_hash)
    print("Tokenizer matches checkpoint: YES")

    print("\nIndexing Scenario C datasets...")
    train_path = DATA_DIR / "train_functions.jsonl"
    valid_path = DATA_DIR / "valid_functions.jsonl"
    report_path = DATA_DIR / "extraction_report.json"

    train_dataset = ScenarioCCompletionDataset(
        train_path,
        TOKENIZER,
        context_length=config.context_length,
    )
    valid_dataset = ScenarioCCompletionDataset(
        valid_path,
        TOKENIZER,
        context_length=config.context_length,
    )

    config.validate_tokenizer_vocab(
        train_dataset.tokenizer.get_vocab_size()
    )

    # Stable dataset fingerprint based on the extraction report and
    # production JSONL file sizes. This avoids re-hashing several GB
    # of corpus data on every resume.
    manifest = hashlib.sha256()
    for data_path in (train_path, valid_path):
        manifest.update(data_path.name.encode("utf-8"))
        manifest.update(str(data_path.stat().st_size).encode("ascii"))

    if report_path.is_file():
        manifest.update(sha256_file(report_path).encode("ascii"))

    data_hash = manifest.hexdigest()

    saved_data_hash = payload.get("data_manifest_sha256", "")
    if args.mode == "resume":
        if not saved_data_hash or saved_data_hash != data_hash:
            raise RuntimeError(
                "Scenario C dataset fingerprint does not match the "
                "resume checkpoint. Do not resume with changed data."
            )

    print("Dataset fingerprint:", data_hash)
    print("Train records:", len(train_dataset))
    print("Validation records:", len(valid_dataset))
    print("Malformed train records:", train_dataset.skipped_invalid_json)
    print("Oversized train records:", train_dataset.skipped_oversize)

    collate = partial(
        scenario_c_collate,
        pad_id=train_dataset.pad_id,
    )

    generator = torch.Generator()
    generator.manual_seed(42)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.micro_batch,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate,
    )

    validation_count = min(args.validation_records, len(valid_dataset))
    validation_generator = torch.Generator().manual_seed(2026)
    validation_indices = torch.randperm(
        len(valid_dataset),
        generator=validation_generator,
    )[:validation_count].tolist()

    validation_subset = Subset(valid_dataset, validation_indices)

    valid_loader = DataLoader(
        validation_subset,
        batch_size=args.validation_batch,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate,
    )

    print("Validation subset:", len(validation_subset))
    print("Validation batch size:", args.validation_batch)
    print("Validation batches:", len(valid_loader))

    settings = TrainSettings(
        micro_batch_size=args.micro_batch,
        gradient_accumulation_steps=args.accumulation,
        max_steps=args.steps,
        additional_steps=args.steps,
        total_steps=args.total_steps,
        warmup_fraction=0.05,
        min_lr_ratio=0.1,
        learning_rate=args.learning_rate,
        weight_decay=0.1,
        gradient_clip=1.0,
        precision="auto",
        seed=42,
        context_length=config.context_length,
        log_every=1,
        eval_steps=(),
        checkpoint_steps=(),
        eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every,
        initial_validation=False,
        complete_pass=False,
        partial_final_update=False,
        initial_global_step=initial_global_step,
        fresh_schedule=(args.mode == "fresh"),
        deterministic=False,
    )

    print("\n========== TRAINING CONFIGURATION ==========")
    print("Mode:", args.mode)
    print("Micro-batch:", settings.micro_batch_size)
    print("Gradient accumulation:", settings.gradient_accumulation_steps)
    print("Effective batch:", settings.effective_batch_size)
    print("Target stage updates:", settings.additional_steps)
    print("Already completed:", restored_stage_step)
    print("Remaining updates:", settings.additional_steps - restored_stage_step)
    print("Starting global step:", settings.initial_global_step)
    print("Expected final global step:", settings.expected_final_global_step)
    print("Learning rate:", settings.learning_rate)
    print("Scheduler horizon:", settings.total_steps)
    print("Evaluation interval:", settings.eval_every)
    print("Checkpoint interval:", settings.checkpoint_every)
    print("Output:", output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model,
        train_loader,
        settings,
        device="cuda",
        val_loader=valid_loader,
        train_tokens=None,
        checkpoint_dir=output_dir,
        model_config=config,
        tokenizer_sha256=tokenizer_hash,
        data_manifest_sha256=data_hash,
    )

    if args.mode == "resume":
        print("\nRestoring complete Scenario C training state...")
        trainer.restore(payload, reset_loader_state=False)

        if trainer.stage_step != restored_stage_step:
            raise RuntimeError(
                f"Restored stage step mismatch: "
                f"{trainer.stage_step} != {restored_stage_step}"
            )

        if trainer.step_index != int(payload["step"]):
            raise RuntimeError("Restored global step does not match checkpoint.")

        print("Optimizer state: RESTORED")
        print("Scheduler state: RESTORED")
        print("Random states: RESTORED")
        print("DataLoader position: RESTORED")
        print("Global step:", trainer.step_index)
        print("Stage step:", trainer.stage_step)

    print("\n========== STARTING SCENARIO C TRAINING ==========")
    result = trainer.run()

    if trainer.stage_step != args.steps:
        raise RuntimeError(
            f"Training stopped at stage step {trainer.stage_step}; "
            f"target was {args.steps}."
        )

    print("\n========== FINAL RESULT ==========")
    print(json.dumps(result, indent=2))

    print(
        "\nPeak allocated VRAM GB:",
        round(torch.cuda.max_memory_allocated() / 1024**3, 3),
    )

    print("\n========== CHECKPOINT FILES ==========")
    for checkpoint_path in sorted(output_dir.glob("*.pt")):
        print(
            checkpoint_path.name,
            "|",
            round(checkpoint_path.stat().st_size / 1024**2, 2),
            "MB",
        )

    summary_path = output_dir / "scenario_c_training_summary.json"
    summary_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("\nSCENARIO C TRAINING COMPLETED")


if __name__ == "__main__":
    main()
