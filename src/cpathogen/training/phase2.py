"""Phase-2 spatial-concat and morphology-FiLM training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from diffusers.optimization import get_scheduler
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from cpathogen.generation.conditioning import (
    SpatialCondEncoder,
    film_condition,
    inject_film_into_unet,
    set_film_condition,
)
from cpathogen.training.common import (
    checkpoint_step,
    diffusion_batch,
    expand_unet_input,
    load_base_components,
    load_unet,
    mse_loss,
    resolve_resume_checkpoint,
    save_base_components,
    set_global_seed,
    write_manifest,
)
from cpathogen.training.data import Phase2ConditionDataset, make_collate_fn


@dataclass(frozen=True)
class Phase2TrainingConfig:
    """Configuration for the two-path structured-conditioning trainer."""

    base_model: str
    phase1_model: str
    tiles_dir: Path
    spatial_maps_dir: Path
    morphology_table: Path
    output_dir: Path
    resolution: int = 512
    prompt: str = "he"
    max_samples: int | None = None
    max_train_steps: int = 50_000
    train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 8.0e-6
    unet_learning_rate_scale: float = 0.3
    spatial_learning_rate_scale: float = 1.0
    minimum_learning_rate: float = 1.0e-7
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 1_000
    checkpointing_steps: int = 5_000
    dataloader_workers: int = 4
    mixed_precision: str = "fp16"
    gradient_checkpointing: bool = True
    random_flip: bool = True
    use_8bit_adam: bool = False
    max_grad_norm: float = 1.0
    prediction_type: str | None = None
    seed: int = 42
    resume_from_checkpoint: str | Path | None = None
    local_files_only: bool = True
    validate_only: bool = False
    forward_check: bool = False
    logging_steps: int = 10


def _validate_config(config: Phase2TrainingConfig) -> None:
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
    for name, value in (
        ("learning_rate", config.learning_rate),
        ("unet_learning_rate_scale", config.unet_learning_rate_scale),
        ("spatial_learning_rate_scale", config.spatial_learning_rate_scale),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")


def _optimizer(config: Phase2TrainingConfig, unet, spatial_encoder, film_mlps):
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

    film_parameter_ids = {id(parameter) for parameter in film_mlps.parameters()}
    base_parameters = [
        parameter
        for parameter in unet.parameters()
        if id(parameter) not in film_parameter_ids
    ]
    return optimizer_class(
        [
            {
                "params": base_parameters,
                "lr": config.learning_rate * config.unet_learning_rate_scale,
            },
            {"params": film_mlps.parameters(), "lr": config.learning_rate},
            {
                "params": spatial_encoder.parameters(),
                "lr": config.learning_rate * config.spatial_learning_rate_scale,
            },
        ]
    )


def _weight_dtype(mixed_precision: str) -> torch.dtype:
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def _learning_rate_scheduler(config, optimizer, process_count: int):
    total_steps = config.max_train_steps * process_count
    warmup_steps = config.lr_warmup_steps * process_count
    if config.lr_scheduler != "cosine":
        return get_scheduler(
            config.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
    floor = min(1.0, config.minimum_learning_rate / config.learning_rate)

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(float(step) / float(warmup_steps), 1.0e-8)
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return floor + (1.0 - floor) * cosine

    return LambdaLR(optimizer, multiplier)


def run_phase2_training(config: Phase2TrainingConfig) -> dict[str, object]:
    """Validate, forward-check, or train the Phase-2 conditioned UNet."""
    _validate_config(config)
    set_global_seed(config.seed)
    dataset = Phase2ConditionDataset(
        config.tiles_dir,
        config.spatial_maps_dir,
        config.morphology_table,
        resolution=config.resolution,
        prompt=config.prompt,
        max_samples=config.max_samples,
        random_flip=config.random_flip and not config.validate_only,
    )
    summary: dict[str, object] = {
        "phase": 2,
        "samples": len(dataset),
        "tiles_dir": str(dataset.tiles_dir),
        "spatial_maps_dir": str(dataset.spatial_maps_dir),
        "morphology_table": str(dataset.morphology_table),
        "morphology_features": list(dataset.feature_names),
        "resolution": config.resolution,
    }
    if config.validate_only:
        return summary

    from accelerate import Accelerator
    from accelerate.utils import ProjectConfiguration

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        project_config=ProjectConfiguration(
            project_dir=config.output_dir,
            logging_dir=config.output_dir / "logs",
        ),
    )
    tokenizer, text_encoder, vae, noise_scheduler = load_base_components(
        config.base_model,
        local_files_only=config.local_files_only,
    )
    unet = load_unet(
        config.phase1_model,
        local_files_only=config.local_files_only,
    )
    expand_unet_input(unet)
    film_mlps = inject_film_into_unet(unet, film_dim=16)
    spatial_encoder = SpatialCondEncoder()
    if config.prediction_type is not None:
        noise_scheduler.register_to_config(prediction_type=config.prediction_type)

    text_encoder.requires_grad_(False).eval()
    vae.requires_grad_(False).eval()
    unet.requires_grad_(True).train()
    spatial_encoder.requires_grad_(True).train()
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
    optimizer = _optimizer(config, unet, spatial_encoder, film_mlps)
    lr_scheduler = _learning_rate_scheduler(
        config, optimizer, accelerator.num_processes
    )
    unet, spatial_encoder, optimizer, train_dataloader, lr_scheduler = (
        accelerator.prepare(
            unet,
            spatial_encoder,
            optimizer,
            train_dataloader,
            lr_scheduler,
        )
    )
    weight_dtype = _weight_dtype(config.mixed_precision)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    resume = resolve_resume_checkpoint(config.output_dir, config.resume_from_checkpoint)
    global_step = 0
    if resume is not None:
        accelerator.load_state(resume)
        global_step = checkpoint_step(resume)

    def prepare_prediction(batch: dict[str, object]):
        noisy, timesteps, hidden, target = diffusion_batch(
            batch=batch,
            vae=vae,
            text_encoder=text_encoder,
            scheduler=noise_scheduler,
            weight_dtype=weight_dtype,
        )
        spatial = batch["spatial_maps"].to(dtype=weight_dtype)
        morphology = batch["morphology"].to(dtype=weight_dtype)
        spatial_features = spatial_encoder(spatial)
        model_input = torch.cat([noisy, spatial_features.to(dtype=noisy.dtype)], dim=1)
        return model_input, timesteps, hidden, target, morphology, spatial_features

    def predict_once(batch: dict[str, object]):
        model_input, timesteps, hidden, target, morphology, spatial_features = (
            prepare_prediction(batch)
        )
        with film_condition(unet, morphology):
            prediction = unet(model_input, timesteps, hidden, return_dict=False)[0]
        return prediction, target, spatial_features

    if config.forward_check:
        batch = next(iter(train_dataloader))
        with torch.no_grad(), accelerator.autocast():
            prediction, target, spatial_features = predict_once(batch)
            loss = mse_loss(prediction, target)
        if not torch.isfinite(loss):
            raise RuntimeError("Phase-2 forward check produced a non-finite loss")
        summary.update(
            {
                "forward_check": "passed",
                "prediction_shape": list(prediction.shape),
                "spatial_feature_shape": list(spatial_features.shape),
                "film_blocks": len(film_mlps),
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
        desc="Phase 2",
    )
    epoch = first_epoch
    last_loss = float("nan")
    while global_step < config.max_train_steps:
        for batch in train_dataloader:
            with accelerator.accumulate(unet, spatial_encoder):
                with accelerator.autocast():
                    (
                        model_input,
                        timesteps,
                        hidden,
                        target,
                        morphology,
                        _,
                    ) = prepare_prediction(batch)
                    # Keep the condition attached until backward finishes. This
                    # is required because gradient checkpointing recomputes the
                    # ResNet blocks during backward.
                    set_film_condition(unet, morphology)
                    prediction = unet(
                        model_input, timesteps, hidden, return_dict=False
                    )[0]
                    loss = mse_loss(prediction, target)
                try:
                    accelerator.backward(loss)
                finally:
                    set_film_condition(unet, None)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        list(unet.parameters()) + list(spatial_encoder.parameters()),
                        config.max_grad_norm,
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                last_loss = float(
                    accelerator.gather(loss.detach().reshape(1)).mean().cpu()
                )
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
        unwrapped_unet = accelerator.unwrap_model(unet)
        unwrapped_spatial = accelerator.unwrap_model(spatial_encoder)
        unwrapped_unet.save_pretrained(final_dir / "unet", safe_serialization=True)
        final_dir.mkdir(parents=True, exist_ok=True)
        torch.save(unwrapped_spatial.state_dict(), final_dir / "spatial_encoder.pt")
        torch.save(film_mlps.state_dict(), final_dir / "film_mlps.pt")
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
                "architecture": "direct_concat_spatial_encoder_plus_film",
                "film_blocks": len(film_mlps),
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
