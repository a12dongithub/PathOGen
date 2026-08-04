"""PathOGen inference backend with guidance extension points."""

from __future__ import annotations

import contextlib
import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, UNet2DConditionModel
from PIL import Image
from safetensors.torch import load_file as load_safetensors
from transformers import CLIPTextModel, CLIPTokenizer

from .guidance import (
    CandidateDecision,
    GenerationContext,
    GuidanceHook,
    NoOpGuidance,
    apply_retry_feedback,
)
from .model_components import SpatialCondEncoder, inject_film_into_unet

BASE_MODEL = "Manojb/stable-diffusion-2-1-base"


@dataclass(frozen=True)
class GenerationResult:
    image: Image.Image
    context: GenerationContext
    decision: CandidateDecision
    seconds: float


def _load_torch_state(path: Path) -> dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


class PathOGenGenerator:
    """Single-image generator designed for paired fidelity experiments."""

    def __init__(
        self,
        checkpoint_dir: Path,
        precision: str = "auto",
        low_vram: bool | None = None,
        base_model: str = BASE_MODEL,
    ):
        self.checkpoint_dir = checkpoint_dir.resolve()
        self.base_model = base_model
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for PathOGen experiment generation")
        self.device = torch.device("cuda")
        self.gpu_name = torch.cuda.get_device_name(0)
        self.gpu_memory_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if precision not in {"auto", "fp16", "fp32"}:
            raise ValueError("precision must be auto, fp16, or fp32")
        if precision == "auto":
            precision = "fp32" if "GTX 16" in self.gpu_name.upper() else "fp16"
        self.precision = precision
        self.dtype = torch.float16 if precision == "fp16" else torch.float32
        self.low_vram = (
            self.gpu_memory_gib < 12 or self.dtype == torch.float32
            if low_vram is None
            else bool(low_vram)
        )
        self.unet: UNet2DConditionModel | None = None
        self.vae: AutoencoderKL | None = None
        self.spatial_encoder: SpatialCondEncoder | None = None
        self.noise_scheduler: DDPMScheduler | None = None
        self.text_embeddings: torch.Tensor | None = None

    @property
    def loaded(self) -> bool:
        return all(
            item is not None
            for item in (
                self.unet,
                self.vae,
                self.spatial_encoder,
                self.noise_scheduler,
                self.text_embeddings,
            )
        )

    def describe(self) -> dict[str, Any]:
        return {
            "checkpoint": str(self.checkpoint_dir),
            "base_model": self.base_model,
            "gpu": self.gpu_name,
            "gpu_memory_gib": round(self.gpu_memory_gib, 2),
            "precision": self.precision,
            "low_vram": self.low_vram,
        }

    def load(self) -> None:
        if self.loaded:
            return
        unet_dir = self.checkpoint_dir / "unet"
        for required in (
            unet_dir / "config.json",
            unet_dir / "diffusion_pytorch_model.safetensors",
            self.checkpoint_dir / "film_mlps.pt",
            self.checkpoint_dir / "spatial_encoder.pt",
        ):
            if not required.is_file():
                raise FileNotFoundError(f"Required checkpoint artifact missing: {required}")

        tokenizer = CLIPTokenizer.from_pretrained(self.base_model, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(
            self.base_model, subfolder="text_encoder", torch_dtype=self.dtype
        ).to(self.device)
        tokens = tokenizer(
            ["he"],
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        with torch.inference_mode():
            self.text_embeddings = text_encoder(
                tokens.input_ids.to(self.device), return_dict=False
            )[0].detach()
        if self.low_vram:
            self.text_embeddings = self.text_embeddings.cpu()
        del tokenizer, text_encoder
        gc.collect()
        torch.cuda.empty_cache()

        config = UNet2DConditionModel.load_config(str(unet_dir))
        unet = UNet2DConditionModel.from_config(config)
        film_mlps = inject_film_into_unet(unet, film_dim=16)
        state = load_safetensors(
            str(unet_dir / "diffusion_pytorch_model.safetensors"), device="cpu"
        )
        missing, unexpected = unet.load_state_dict(state, strict=False)
        non_film_missing = [key for key in missing if ".film_mlp." not in key]
        if non_film_missing or unexpected:
            raise RuntimeError(
                "Checkpoint/UNet mismatch: "
                f"{len(non_film_missing)} non-FiLM missing, {len(unexpected)} unexpected"
            )
        del state
        film_mlps.load_state_dict(_load_torch_state(self.checkpoint_dir / "film_mlps.pt"))
        unet.set_attention_slice("max")
        self.unet = unet.to(
            device="cpu" if self.low_vram else self.device, dtype=self.dtype
        ).eval()

        spatial_encoder = SpatialCondEncoder()
        spatial_encoder.load_state_dict(
            _load_torch_state(self.checkpoint_dir / "spatial_encoder.pt")
        )
        self.spatial_encoder = spatial_encoder.to(
            device="cpu" if self.low_vram else self.device, dtype=self.dtype
        ).eval()

        vae_dir = self.checkpoint_dir / "vae"
        if vae_dir.is_dir():
            vae = AutoencoderKL.from_pretrained(
                str(self.checkpoint_dir), subfolder="vae", torch_dtype=torch.float32
            )
        else:
            vae = AutoencoderKL.from_pretrained(
                self.base_model, subfolder="vae", torch_dtype=torch.float32
            )
        vae.enable_slicing()
        vae.enable_tiling()
        self.vae = vae.to("cpu" if self.low_vram else self.device).eval()
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.base_model, subfolder="scheduler"
        )

    def unload(self) -> None:
        self.unet = None
        self.vae = None
        self.spatial_encoder = None
        self.noise_scheduler = None
        self.text_embeddings = None
        gc.collect()
        torch.cuda.empty_cache()

    def _component_to_gpu(self, component: torch.nn.Module) -> None:
        if self.low_vram:
            component.to(self.device)

    def _component_to_cpu(self, component: torch.nn.Module) -> None:
        if self.low_vram:
            component.to("cpu")
            torch.cuda.empty_cache()

    def _generate_once(
        self,
        context: GenerationContext,
        steps: int,
        spatial_strength: float,
        hook: GuidanceHook,
    ) -> Image.Image:
        self.load()
        assert self.unet is not None
        assert self.vae is not None
        assert self.spatial_encoder is not None
        assert self.noise_scheduler is not None
        assert self.text_embeddings is not None
        if context.spatial_map.shape != (512, 512, 5):
            raise ValueError(f"Expected spatial map 512x512x5, got {context.spatial_map.shape}")
        if context.morphology.shape != (16,):
            raise ValueError(f"Expected morphology vector length 16, got {context.morphology.shape}")

        use_autocast = self.dtype == torch.float16
        autocast = (
            lambda: torch.autocast(device_type="cuda", dtype=torch.float16)
        ) if use_autocast else contextlib.nullcontext

        spatial = torch.from_numpy(context.spatial_map.astype(np.float32) / 255.0)
        spatial = spatial.permute(2, 0, 1).unsqueeze(0).to(dtype=self.dtype)
        morphology = torch.from_numpy(context.morphology.astype(np.float32)).unsqueeze(0)
        morphology = morphology.to(self.device, dtype=self.dtype)
        effective_spatial_strength = float(
            context.metadata.get("guidance_spatial_scale", spatial_strength)
        )

        self._component_to_gpu(self.spatial_encoder)
        with torch.inference_mode(), autocast():
            spatial_features = self.spatial_encoder(spatial.to(self.device))
            spatial_features = spatial_features * effective_spatial_strength
        self._component_to_cpu(self.spatial_encoder)

        scheduler = DDIMScheduler(
            beta_start=self.noise_scheduler.config.beta_start,
            beta_end=self.noise_scheduler.config.beta_end,
            beta_schedule=self.noise_scheduler.config.beta_schedule,
            num_train_timesteps=self.noise_scheduler.config.num_train_timesteps,
            prediction_type=self.noise_scheduler.config.prediction_type,
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
            timestep_spacing="leading",
        )
        scheduler.set_timesteps(int(steps), device=self.device)
        generator = torch.Generator(device=self.device).manual_seed(int(context.seed))
        latents = torch.randn(
            (1, 4, 64, 64),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        ) * scheduler.init_noise_sigma

        self._component_to_gpu(self.unet)
        embeddings = self.text_embeddings.to(self.device, dtype=self.dtype)
        for module in self.unet.modules():
            if hasattr(module, "film_mlp"):
                module.current_morph16 = morphology
        try:
            for step_index, timestep in enumerate(scheduler.timesteps):
                latent_input = scheduler.scale_model_input(latents, timestep)
                model_input = torch.cat([latent_input, spatial_features], dim=1)
                with torch.inference_mode(), autocast():
                    noise = self.unet(
                        model_input,
                        timestep,
                        encoder_hidden_states=embeddings,
                        return_dict=False,
                    )[0]
                    latents = scheduler.step(noise, timestep, latents, return_dict=False)[0]
                latents = hook.on_denoising_step(
                    context, step_index, timestep, latents
                )
                if not torch.isfinite(latents).all():
                    raise RuntimeError(
                        f"Non-finite latents at denoising step {step_index + 1}/{steps}"
                    )
        finally:
            for module in self.unet.modules():
                if hasattr(module, "film_mlp"):
                    module.current_morph16 = None
        self._component_to_cpu(self.unet)

        self._component_to_gpu(self.vae)
        scaled = latents.to(dtype=torch.float32) / self.vae.config.scaling_factor
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
            decoded = self.vae.decode(scaled, return_dict=False)[0]
        self._component_to_cpu(self.vae)
        decoded = (decoded / 2 + 0.5).clamp(0, 1)
        array = decoded[0].permute(1, 2, 0).detach().cpu().numpy()
        return Image.fromarray((array * 255.0).round().astype(np.uint8))

    def generate(
        self,
        context: GenerationContext,
        steps: int = 20,
        spatial_strength: float = 1.0,
        hook: GuidanceHook | None = None,
        max_attempts: int = 1,
    ) -> GenerationResult:
        if steps < 1 or max_attempts < 1:
            raise ValueError("steps and max_attempts must be positive")
        hook = hook or NoOpGuidance()
        started = time.perf_counter()
        active = context
        for attempt in range(max_attempts):
            active = active.clone(attempt=attempt)
            active = hook.adjust_conditions(active)
            image = self._generate_once(active, steps, spatial_strength, hook)
            decision = hook.evaluate_candidate(image, active)
            if decision.accept:
                return GenerationResult(
                    image=image,
                    context=active,
                    decision=decision,
                    seconds=time.perf_counter() - started,
                )
            if attempt + 1 < max_attempts:
                active = apply_retry_feedback(active, decision, active.seed + 1)
        return GenerationResult(
            image=image,
            context=active,
            decision=decision,
            seconds=time.perf_counter() - started,
        )
