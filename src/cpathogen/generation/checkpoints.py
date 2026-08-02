"""Loading contracts for trained PathOGen Phase-2 checkpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.utils import logging as diffusers_logging
from torch import nn

from cpathogen.generation.conditioning import (
    SpatialCondEncoder,
    inject_film_into_unet,
)

REQUIRED_PHASE2_PATHS = (
    Path("unet/config.json"),
    Path("unet/diffusion_pytorch_model.safetensors"),
    Path("spatial_encoder.pt"),
    Path("film_mlps.pt"),
)


@dataclass(frozen=True)
class Phase2ConditioningModels:
    """Frozen model components required for a conditioned UNet forward pass."""

    checkpoint_dir: Path
    base_model: str
    unet: UNet2DConditionModel
    spatial_encoder: SpatialCondEncoder
    film_mlps: nn.ModuleList
    device: torch.device
    dtype: torch.dtype


@dataclass(frozen=True)
class Phase2GenerationModels:
    """All components required for full Phase-2 image generation."""

    conditioning: Phase2ConditioningModels
    vae: AutoencoderKL
    text_encoder: nn.Module
    tokenizer: object
    noise_scheduler: DDPMScheduler
    base_model: str
    revision: str | None

    @property
    def unet(self) -> UNet2DConditionModel:
        return self.conditioning.unet

    @property
    def spatial_encoder(self) -> SpatialCondEncoder:
        return self.conditioning.spatial_encoder

    @property
    def device(self) -> torch.device:
        return self.conditioning.device

    @property
    def dtype(self) -> torch.dtype:
        return self.conditioning.dtype


def validate_phase2_checkpoint(checkpoint_dir: str | Path) -> Path:
    """Return a resolved checkpoint directory after validating its contract."""
    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(
            f"Phase-2 checkpoint directory not found: {checkpoint_dir}"
        )

    missing = [
        str(relative_path)
        for relative_path in REQUIRED_PHASE2_PATHS
        if not (checkpoint_dir / relative_path).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Incomplete Phase-2 checkpoint at {checkpoint_dir}; missing: {missing}"
        )
    return checkpoint_dir


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve ``auto`` to CUDA, Apple MPS, or CPU in that order."""
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(requested: str, device: torch.device) -> torch.dtype:
    """Resolve a user dtype while avoiding unsupported CPU half precision."""
    if requested == "auto":
        return torch.float16 if device.type in {"cuda", "mps"} else torch.float32
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    try:
        dtype = mapping[requested]
    except KeyError as error:
        raise ValueError(f"Unsupported dtype: {requested}") from error
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("float16 CPU inference is unsupported; use float32")
    return dtype


def _load_weights(path: Path) -> dict[str, torch.Tensor]:
    """Load a tensor-only PyTorch state dictionary without unpickling objects."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch before the weights_only argument was introduced.
        return torch.load(path, map_location="cpu")


def load_phase2_conditioning_models(
    checkpoint_dir: str | Path,
    *,
    device: str = "auto",
    dtype: str = "auto",
) -> Phase2ConditioningModels:
    """Load the UNet and both conditioning paths from a local checkpoint."""
    checkpoint_dir = validate_phase2_checkpoint(checkpoint_dir)
    resolved_device = resolve_device(device)
    resolved_dtype = resolve_dtype(dtype, resolved_device)

    with (checkpoint_dir / "unet" / "config.json").open(
        "r", encoding="utf-8"
    ) as handle:
        unet_config = json.load(handle)
    if unet_config.get("in_channels") != 8:
        raise ValueError(
            "Expected an eight-channel direct-concat UNet checkpoint, found "
            f"in_channels={unet_config.get('in_channels')}"
        )
    if unet_config.get("cross_attention_dim") != 1024:
        raise ValueError(
            "Expected SD 2.1 cross_attention_dim=1024, found "
            f"{unet_config.get('cross_attention_dim')}"
        )

    # This checkpoint embeds FiLM keys in the UNet safetensors file and also
    # stores their canonical state in film_mlps.pt. Diffusers cannot construct
    # those custom modules itself, so suppress only its expected "unused keys"
    # warning here; the separate state is loaded strictly immediately below.
    previous_verbosity = diffusers_logging.get_verbosity()
    diffusers_logging.set_verbosity_error()
    try:
        unet = UNet2DConditionModel.from_pretrained(
            checkpoint_dir / "unet",
            torch_dtype=resolved_dtype,
            local_files_only=True,
            use_safetensors=True,
        )
    finally:
        diffusers_logging.set_verbosity(previous_verbosity)
    film_mlps = inject_film_into_unet(unet, film_dim=16)
    film_state = _load_weights(checkpoint_dir / "film_mlps.pt")
    film_mlps.load_state_dict(film_state, strict=True)

    spatial_encoder = SpatialCondEncoder()
    spatial_state = _load_weights(checkpoint_dir / "spatial_encoder.pt")
    spatial_encoder.load_state_dict(spatial_state, strict=True)

    unet.to(device=resolved_device, dtype=resolved_dtype).eval()
    spatial_encoder.to(device=resolved_device, dtype=resolved_dtype).eval()
    film_mlps.to(device=resolved_device, dtype=resolved_dtype).eval()
    for module in (unet, spatial_encoder, film_mlps):
        module.requires_grad_(False)

    return Phase2ConditioningModels(
        checkpoint_dir=checkpoint_dir,
        base_model=unet_config.get("_name_or_path", "Manojb/stable-diffusion-2-1-base"),
        unet=unet,
        spatial_encoder=spatial_encoder,
        film_mlps=film_mlps,
        device=resolved_device,
        dtype=resolved_dtype,
    )


def load_phase2_generation_models(
    checkpoint_dir: str | Path,
    *,
    base_model: str | None = None,
    revision: str | None = None,
    device: str = "auto",
    dtype: str = "auto",
    local_files_only: bool = False,
) -> Phase2GenerationModels:
    """Load a local Phase-2 checkpoint plus immutable SD 2.1 text/scheduler parts."""
    conditioning = load_phase2_conditioning_models(
        checkpoint_dir, device=device, dtype=dtype
    )
    checkpoint_dir = conditioning.checkpoint_dir
    vae_dir = checkpoint_dir / "vae"
    if not (vae_dir / "config.json").is_file():
        raise FileNotFoundError(f"Checkpoint VAE config not found: {vae_dir}")
    if not any(
        (vae_dir / filename).is_file()
        for filename in (
            "diffusion_pytorch_model.safetensors",
            "diffusion_pytorch_model.bin",
        )
    ):
        raise FileNotFoundError(f"Checkpoint VAE weights not found: {vae_dir}")

    bundled_base_components = all(
        (checkpoint_dir / component).is_dir()
        for component in ("tokenizer", "text_encoder", "scheduler")
    )
    if base_model is None and bundled_base_components:
        resolved_base_model = str(checkpoint_dir)
        local_files_only = True
        revision = None
    else:
        resolved_base_model = base_model or conditioning.base_model
    common_kwargs: dict[str, object] = {
        "local_files_only": local_files_only,
    }
    if revision is not None:
        common_kwargs["revision"] = revision

    # Imported lazily so checkpoint-only validation does not require Transformers.
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(
        resolved_base_model, subfolder="tokenizer", **common_kwargs
    )
    text_config = CLIPTextConfig.from_pretrained(
        resolved_base_model, subfolder="text_encoder", **common_kwargs
    )
    text_encoder = CLIPTextModel.from_pretrained(
        resolved_base_model,
        subfolder="text_encoder",
        config=text_config,
        torch_dtype=conditioning.dtype,
        **common_kwargs,
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        resolved_base_model, subfolder="scheduler", **common_kwargs
    )
    vae = AutoencoderKL.from_pretrained(
        vae_dir,
        torch_dtype=conditioning.dtype,
        local_files_only=True,
        use_safetensors=True,
    )

    text_encoder.to(device=conditioning.device, dtype=conditioning.dtype).eval()
    vae.to(device=conditioning.device, dtype=conditioning.dtype).eval()
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    return Phase2GenerationModels(
        conditioning=conditioning,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        noise_scheduler=noise_scheduler,
        base_model=resolved_base_model,
        revision=revision,
    )
