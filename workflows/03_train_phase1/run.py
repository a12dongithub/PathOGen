#!/usr/bin/env python3
"""Workflow 03: train the Phase-1 H&E domain-adapted diffusion UNet."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from cpathogen.training.config_io import (
    parse_args_with_yaml,
    resolve_model_source,
    resolve_repository_path,
)
from cpathogen.training.phase1 import Phase1TrainingConfig, run_phase1_training

DEFAULT_BASE = REPOSITORY_ROOT / "artifacts/models/pathogen_phase2/checkpoint_30000"
DEFAULT_PHASE1 = REPOSITORY_ROOT / "artifacts/models/pathogen_phase1/checkpoint_30000"
DEFAULT_METADATA = REPOSITORY_ROOT / "data/processed/conditions/metadata.jsonl"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "artifacts/runs/phase1_domain_adapt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a four-channel Stable Diffusion UNet on H&E tiles. The base "
            "model supplies frozen VAE/text components; --initial-unet may point "
            "to an existing Phase-1 model for continued training."
        )
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--base-model", default=str(DEFAULT_BASE))
    parser.add_argument("--initial-unet", default=str(DEFAULT_PHASE1))
    parser.add_argument("--metadata-file", default=str(DEFAULT_METADATA))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-train-steps", type=int, default=100_000)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--lr-scheduler", default="constant_with_warmup")
    parser.add_argument("--lr-warmup-steps", type=int, default=1_000)
    parser.add_argument("--checkpointing-steps", type=int, default=10_000)
    parser.add_argument("--dataloader-workers", type=int, default=4)
    parser.add_argument(
        "--mixed-precision", choices=("no", "fp16", "bf16"), default="no"
    )
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--random-flip", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--use-8bit-adam", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--use-ema", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--prediction-type", choices=("epsilon", "v_prediction"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument(
        "--local-files-only", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--forward-check",
        action="store_true",
        help="Run one real no-gradient model pass, report its loss, and exit.",
    )
    parser.add_argument("--logging-steps", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = parse_args_with_yaml(build_parser(), argv)
    base_model = resolve_model_source(args.base_model, REPOSITORY_ROOT)
    initial_unet = (
        resolve_model_source(args.initial_unet, REPOSITORY_ROOT)
        if args.initial_unet
        else None
    )
    config = Phase1TrainingConfig(
        base_model=base_model,
        initial_unet=initial_unet,
        metadata_file=resolve_repository_path(args.metadata_file, REPOSITORY_ROOT),
        output_dir=resolve_repository_path(args.output_dir, REPOSITORY_ROOT),
        resolution=args.resolution,
        max_samples=args.max_samples,
        max_train_steps=args.max_train_steps,
        train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler=args.lr_scheduler,
        lr_warmup_steps=args.lr_warmup_steps,
        checkpointing_steps=args.checkpointing_steps,
        dataloader_workers=args.dataloader_workers,
        mixed_precision=args.mixed_precision,
        gradient_checkpointing=args.gradient_checkpointing,
        random_flip=args.random_flip,
        use_8bit_adam=args.use_8bit_adam,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
        max_grad_norm=args.max_grad_norm,
        prediction_type=args.prediction_type,
        seed=args.seed,
        resume_from_checkpoint=args.resume_from_checkpoint,
        local_files_only=args.local_files_only,
        validate_only=args.validate_only,
        forward_check=args.forward_check,
        logging_steps=args.logging_steps,
    )
    result = run_phase1_training(config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
