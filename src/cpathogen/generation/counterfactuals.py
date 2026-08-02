"""Matched-noise Phase-2 generation for baseline and intervened controls."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from diffusers import DDIMScheduler
from PIL import Image

from cpathogen.counterfactuals import ConditionBundle
from cpathogen.generation.checkpoints import Phase2GenerationModels
from cpathogen.generation.conditioning import film_condition


def _ddim_scheduler(models: Phase2GenerationModels) -> DDIMScheduler:
    config = models.noise_scheduler.config
    return DDIMScheduler(
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        beta_schedule=config.beta_schedule,
        num_train_timesteps=config.num_train_timesteps,
        prediction_type=config.prediction_type,
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
        timestep_spacing="leading",
    )


def _initial_latents(
    models: Phase2GenerationModels, batch_size: int, seed: int
) -> torch.Tensor:
    # Generate once on CPU, then clone across conditions. This provides matched
    # noise even when CUDA/MPS random-number implementations differ.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    sample_size = models.unet.config.sample_size
    if isinstance(sample_size, int):
        height = width = sample_size
    else:
        height, width = sample_size
    channels = int(models.unet.config.out_channels)
    latent = torch.randn(
        (1, channels, height, width), generator=generator, dtype=torch.float32
    )
    return latent.expand(batch_size, -1, -1, -1).clone().to(
        device=models.device, dtype=models.dtype
    )


def _to_pil(decoded: torch.Tensor) -> list[Image.Image]:
    images = (decoded / 2 + 0.5).clamp(0, 1)
    images = images.detach().float().cpu().permute(0, 2, 3, 1).numpy()
    arrays = (images * 255.0).round().astype(np.uint8)
    return [Image.fromarray(array, mode="RGB") for array in arrays]


@torch.inference_mode()
def generate_matched_conditions(
    models: Phase2GenerationModels,
    conditions: Sequence[ConditionBundle],
    *,
    seed: int,
    prompt: str = "he",
    num_inference_steps: int = 20,
) -> list[Image.Image]:
    """Generate conditions from the exact same initial latent noise tensor."""
    if not conditions:
        return []
    for condition in conditions:
        condition.validate()
    spatial_shapes = {tuple(condition.spatial.shape) for condition in conditions}
    if len(spatial_shapes) != 1:
        raise ValueError(f"All spatial maps in a batch must have one shape: {spatial_shapes}")

    batch_size = len(conditions)
    scheduler = _ddim_scheduler(models)
    scheduler.set_timesteps(num_inference_steps, device=models.device)

    tokenized = models.tokenizer(
        [prompt] * batch_size,
        max_length=models.tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = models.text_encoder(
        tokenized.input_ids.to(models.device), return_dict=False
    )[0]
    spatial = torch.stack([item.spatial for item in conditions]).to(
        device=models.device, dtype=models.dtype
    )
    morphology = torch.stack([item.morphology for item in conditions]).to(
        device=models.device, dtype=models.dtype
    )
    spatial_features = models.spatial_encoder(spatial)
    latents = _initial_latents(models, batch_size, seed) * scheduler.init_noise_sigma

    with film_condition(models.unet, morphology):
        for timestep in scheduler.timesteps:
            latent_input = scheduler.scale_model_input(latents, timestep)
            model_input = torch.cat([latent_input, spatial_features], dim=1)
            predicted_noise = models.unet(
                model_input,
                timestep,
                encoder_hidden_states=text_embeddings,
                return_dict=False,
            )[0]
            latents = scheduler.step(
                predicted_noise, timestep, latents, return_dict=False
            )[0]

    scaling_factor = float(models.vae.config.scaling_factor)
    decoded = models.vae.decode(latents / scaling_factor, return_dict=False)[0]
    return _to_pil(decoded)
