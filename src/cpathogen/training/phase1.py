"""Phase-1 H&E domain-adaptation training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from cpathogen.training.common import (
    checkpoint_step,
    diffusion_batch,
    load_base_components,
    load_unet,
    mse_loss,
    resolve_resume_checkpoint,
    save_base_components,
    set_global_seed,
    write_manifest,
)
from cpathogen.training.data import Phase1TileDataset, make_collate_fn


@dataclass(frozen=True)
class Phase1TrainingConfig:
    """Configuration for continuing or starting Phase-1 training."""

    base_model: str
    initial_unet: str | None
    metadata_file: Path
    output_dir: Path
    resolution: int = 512
    max_samples: int | None = None
    max_train_steps: int = 100_000
    train_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1.0e-5
    lr_scheduler: str = "constant_with_warmup"
    lr_warmup_steps: int = 1_000
    checkpointing_steps: int = 10_000
    dataloader_workers: int = 4
    mixed_precision: str = "fp16"
    gradient_checkpointing: bool = True
    random_flip: bool = True
    use_8bit_adam: bool = False
    use_ema: bool = True
    ema_decay: float = 0.9999
    max_grad_norm: float = 1.0
    prediction_type: str | None = None
    seed: int = 42
    resume_from_checkpoint: str | Path | None = None
    local_files_only: bool = True
    validate_only: bool = False
    forward_check: bool = False
    logging_steps: int = 10


def _validate_config(config: Phase1TrainingConfig) -> None:
    if config.resolution < 64 or config.resolution % 8:
        raise ValueError("resolution must be at least 64 and divisible by eight")
    if config.train_batch_size < 1 or config.gradient_accumulation_steps < 1:
        raise ValueError("batch size and gradient accumulation must be positive")
    if config.max_train_steps < 1:
        raise ValueError("max_train_steps must be positive")
    if config.checkpointing_steps < 1 or config.logging_steps < 1:
        raise ValueError("checkpointing_steps and logging_steps must be positive")
    if config.mixed_precision not in {"no", "fp16", "bf16"}:
        raise ValueError("mixed_precision must be one of: no, fp16, bf16")
    if config.prediction_type not in {None, "epsilon", "v_prediction"}:
        raise ValueError("prediction_type must be epsilon or v_prediction")


def _optimizer(config: Phase1TrainingConfig, parameters):
    if config.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError as error:
            raise ImportError(
                "use_8bit_adam requires bitsandbytes; disable it on unsupported hosts"
            ) from error
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW
    return optimizer_class(parameters, lr=config.learning_rate)


def _weight_dtype(mixed_precision: str) -> torch.dtype:
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def run_phase1_training(config: Phase1TrainingConfig) -> dict[str, object]:
    """Validate, forward-check, or train the Phase-1 four-channel UNet."""
    _validate_config(config)
    set_global_seed(config.seed)
    dataset = Phase1TileDataset(
        config.metadata_file,
        resolution=config.resolution,
        max_samples=config.max_samples,
        random_flip=config.random_flip and not config.validate_only,
    )
    summary: dict[str, object] = {
        "phase": 1,
        "samples": len(dataset),
        "metadata_file": str(dataset.metadata_file),
        "resolution": config.resolution,
    }
    if config.validate_only:
        return summary

    from accelerate import Accelerator
    from accelerate.utils import ProjectConfiguration
    from diffusers.training_utils import EMAModel

    project = ProjectConfiguration(
        project_dir=config.output_dir,
        logging_dir=config.output_dir / "logs",
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        project_config=project,
    )
    tokenizer, text_encoder, vae, noise_scheduler = load_base_components(
        config.base_model,
        local_files_only=config.local_files_only,
    )
    unet = load_unet(
        config.initial_unet or config.base_model,
        local_files_only=config.local_files_only,
    )
    if unet.config.in_channels != 4:
        raise ValueError(
            f"Phase 1 requires a four-channel UNet, found {unet.config.in_channels}"
        )
    if config.prediction_type is not None:
        noise_scheduler.register_to_config(prediction_type=config.prediction_type)

    text_encoder.requires_grad_(False).eval()
    vae.requires_grad_(False).eval()
    unet.requires_grad_(True).train()
    if config.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    train_dataloader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=config.train_batch_size,
        num_workers=config.dataloader_workers,
        persistent_workers=config.dataloader_workers > 0,
        collate_fn=make_collate_fn(tokenizer),
    )
    optimizer = _optimizer(config, unet.parameters())
    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=config.max_train_steps * accelerator.num_processes,
    )
    ema = (
        EMAModel(
            unet.parameters(),
            decay=config.ema_decay,
            model_cls=unet.__class__,
            model_config=unet.config,
        )
        if config.use_ema
        else None
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    weight_dtype = _weight_dtype(config.mixed_precision)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    if ema is not None:
        ema.to(accelerator.device)

    resume = resolve_resume_checkpoint(config.output_dir, config.resume_from_checkpoint)
    global_step = 0
    if resume is not None:
        accelerator.load_state(resume)
        global_step = checkpoint_step(resume)
        ema_path = resume / "unet_ema"
        if ema is not None and ema_path.is_dir():
            loaded_ema = EMAModel.from_pretrained(ema_path, unet.__class__)
            ema.load_state_dict(loaded_ema.state_dict())
            ema.to(accelerator.device)

    if config.forward_check:
        batch = next(iter(train_dataloader))
        with torch.no_grad(), accelerator.autocast():
            noisy, timesteps, hidden, target = diffusion_batch(
                batch=batch,
                vae=vae,
                text_encoder=text_encoder,
                scheduler=noise_scheduler,
                weight_dtype=weight_dtype,
            )
            prediction = unet(noisy, timesteps, hidden, return_dict=False)[0]
            loss = mse_loss(prediction, target)
        if not torch.isfinite(loss):
            raise RuntimeError("Phase-1 forward check produced a non-finite loss")
        summary.update(
            {
                "forward_check": "passed",
                "prediction_shape": list(prediction.shape),
                "loss": float(loss.detach().cpu()),
                "device": str(accelerator.device),
            }
        )
        return summary

    config.output_dir.mkdir(parents=True, exist_ok=True)
    updates_per_epoch = max(
        1, math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    )
    first_epoch = global_step // updates_per_epoch
    progress = tqdm(
        total=config.max_train_steps,
        initial=global_step,
        disable=not accelerator.is_local_main_process,
        desc="Phase 1",
    )
    epoch = first_epoch
    last_loss = float("nan")
    while global_step < config.max_train_steps:
        for batch in train_dataloader:
            with accelerator.accumulate(unet):
                with accelerator.autocast():
                    noisy, timesteps, hidden, target = diffusion_batch(
                        batch=batch,
                        vae=vae,
                        text_encoder=text_encoder,
                        scheduler=noise_scheduler,
                        weight_dtype=weight_dtype,
                    )
                    prediction = unet(noisy, timesteps, hidden, return_dict=False)[0]
                    loss = mse_loss(prediction, target)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), config.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                last_loss = float(
                    accelerator.gather(loss.detach().reshape(1)).mean().cpu()
                )
                if ema is not None:
                    ema.step(accelerator.unwrap_model(unet).parameters())
                progress.update(1)
                if global_step % config.logging_steps == 0:
                    progress.set_postfix(
                        loss=f"{last_loss:.5f}",
                        lr=f"{lr_scheduler.get_last_lr()[0]:.2e}",
                    )
                if global_step % config.checkpointing_steps == 0:
                    checkpoint = (
                        config.output_dir / "checkpoints" / f"checkpoint-{global_step}"
                    )
                    accelerator.save_state(checkpoint)
                    if accelerator.is_main_process:
                        if ema is not None:
                            ema.save_pretrained(checkpoint / "unet_ema")
                        write_manifest(
                            checkpoint / "training_manifest.json",
                            {"config": config, "global_step": global_step},
                        )
                if global_step >= config.max_train_steps:
                    break
        epoch += 1
    progress.close()
    accelerator.wait_for_everyone()

    final_dir = config.output_dir / "models" / "final"
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(unet)
        if ema is not None:
            ema.store(unwrapped.parameters())
            ema.copy_to(unwrapped.parameters())
        unwrapped.save_pretrained(final_dir / "unet", safe_serialization=True)
        if ema is not None:
            ema.restore(unwrapped.parameters())
        save_base_components(
            final_dir,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            scheduler=noise_scheduler,
        )
        write_manifest(
            final_dir / "training_manifest.json",
            {
                "config": config,
                "global_step": global_step,
                "last_loss": last_loss,
                "architecture": "four_channel_text_conditioned_unet",
            },
        )
    accelerator.wait_for_everyone()
    summary.update(
        {
            "global_step": global_step,
            "last_loss": last_loss,
            "final_model": str(final_dir),
        }
    )
    return summary
