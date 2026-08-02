"""Model loading, checkpointing, and diffusion-loss helpers for training."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from torch import Tensor, nn
from torch.nn import functional


def source_is_local(source: str | Path) -> bool:
    return Path(source).expanduser().exists()


def pretrained_kwargs(source: str | Path, local_files_only: bool) -> dict[str, object]:
    return {"local_files_only": local_files_only or source_is_local(source)}


def load_unet(
    source: str | Path,
    *,
    local_files_only: bool,
    torch_dtype: torch.dtype | None = None,
) -> UNet2DConditionModel:
    path = Path(source).expanduser()
    kwargs: dict[str, object] = pretrained_kwargs(source, local_files_only)
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    if path.is_dir() and (path / "config.json").is_file():
        return UNet2DConditionModel.from_pretrained(path, **kwargs)
    return UNet2DConditionModel.from_pretrained(source, subfolder="unet", **kwargs)


def load_base_components(
    source: str | Path,
    *,
    local_files_only: bool,
    torch_dtype: torch.dtype | None = None,
) -> tuple[object, nn.Module, AutoencoderKL, DDPMScheduler]:
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

    kwargs = pretrained_kwargs(source, local_files_only)
    model_kwargs = dict(kwargs)
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    tokenizer = CLIPTokenizer.from_pretrained(source, subfolder="tokenizer", **kwargs)
    text_config = CLIPTextConfig.from_pretrained(
        source, subfolder="text_encoder", **kwargs
    )
    text_encoder = CLIPTextModel.from_pretrained(
        source,
        subfolder="text_encoder",
        config=text_config,
        **model_kwargs,
    )
    vae = AutoencoderKL.from_pretrained(source, subfolder="vae", **model_kwargs)
    scheduler = DDPMScheduler.from_pretrained(source, subfolder="scheduler", **kwargs)
    return tokenizer, text_encoder, vae, scheduler


def diffusion_target(
    scheduler: DDPMScheduler,
    latents: Tensor,
    noise: Tensor,
    timesteps: Tensor,
) -> Tensor:
    prediction_type = scheduler.config.prediction_type
    if prediction_type == "epsilon":
        return noise
    if prediction_type == "v_prediction":
        return scheduler.get_velocity(latents, noise, timesteps)
    raise ValueError(f"Unsupported scheduler prediction_type: {prediction_type}")


def diffusion_batch(
    *,
    batch: dict[str, object],
    vae: AutoencoderKL,
    text_encoder: nn.Module,
    scheduler: DDPMScheduler,
    weight_dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    pixels = batch["pixel_values"].to(dtype=weight_dtype)
    input_ids = batch["input_ids"]
    with torch.no_grad():
        latents = vae.encode(pixels).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        hidden_states = text_encoder(input_ids, return_dict=False)[0]
    noise = torch.randn_like(latents)
    timesteps = torch.randint(
        0,
        scheduler.config.num_train_timesteps,
        (latents.shape[0],),
        device=latents.device,
        dtype=torch.long,
    )
    noisy_latents = scheduler.add_noise(latents, noise, timesteps)
    target = diffusion_target(scheduler, latents, noise, timesteps)
    return noisy_latents, timesteps, hidden_states, target


def mse_loss(prediction: Tensor, target: Tensor) -> Tensor:
    return functional.mse_loss(prediction.float(), target.float(), reduction="mean")


def expand_unet_input(unet: UNet2DConditionModel) -> None:
    """Expand a Phase-1 UNet from four to eight inputs with a neutral new path."""
    old = unet.conv_in
    if old.in_channels != 4:
        raise ValueError(
            f"Phase-2 initialization requires a four-channel UNet, found {old.in_channels}"
        )
    expanded = nn.Conv2d(
        8,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        groups=old.groups,
        bias=old.bias is not None,
        padding_mode=old.padding_mode,
        device=old.weight.device,
        dtype=old.weight.dtype,
    )
    with torch.no_grad():
        expanded.weight[:, :4].copy_(old.weight)
        expanded.weight[:, 4:].zero_()
        if old.bias is not None:
            expanded.bias.copy_(old.bias)
    unet.conv_in = expanded
    unet.register_to_config(in_channels=8)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints_dir = output_dir / "checkpoints"
    candidates: list[tuple[int, Path]] = []
    if checkpoints_dir.is_dir():
        for path in checkpoints_dir.glob("checkpoint-*"):
            try:
                candidates.append((int(path.name.rsplit("-", 1)[1]), path))
            except ValueError:
                continue
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def resolve_resume_checkpoint(
    output_dir: Path, resume: str | Path | None
) -> Path | None:
    if resume is None:
        return None
    if str(resume) == "latest":
        checkpoint = latest_checkpoint(output_dir)
        if checkpoint is None:
            raise FileNotFoundError(
                f"No checkpoints found under {output_dir / 'checkpoints'}"
            )
        return checkpoint
    checkpoint = Path(resume).expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint}")
    return checkpoint


def checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.rsplit("-", 1)[1])
    except (IndexError, ValueError) as error:
        raise ValueError(
            f"Checkpoint directory must end in '-<step>': {path}"
        ) from error


def save_base_components(
    destination: Path,
    *,
    tokenizer: object,
    text_encoder: nn.Module,
    vae: AutoencoderKL,
    scheduler: DDPMScheduler,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(destination / "tokenizer")
    text_encoder.save_pretrained(destination / "text_encoder", safe_serialization=True)
    vae.save_pretrained(destination / "vae", safe_serialization=True)
    scheduler.save_pretrained(destination / "scheduler")


def json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return json_ready(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_manifest(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
