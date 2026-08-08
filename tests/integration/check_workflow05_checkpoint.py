"""Manual integration check for the Workflow 05 Phase-2 checkpoint.

This is deliberately not part of the production generation workflow. It uses
deterministic synthetic latents and text embeddings to verify that a saved
UNet, spatial encoder, and FiLM modules load and respond to both conditions.
It does not generate an image or evaluate model quality.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cpathogen.generation.checkpoints import load_phase2_conditioning_models
from cpathogen.generation.conditioning import film_condition
from cpathogen.utils.paths import MODEL_ROOT, MORPHOLOGY_STATS, SPATIAL_MAPS

DEFAULT_CHECKPOINT = (
    MODEL_ROOT
    / "pathogen_phase2"
    / "checkpoint_30000"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--spatial-maps-dir", default=str(SPATIAL_MAPS))
    parser.add_argument(
        "--morphology-table",
        default=str(MORPHOLOGY_STATS),
    )
    parser.add_argument("--stem", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timestep", type=int, default=500)
    parser.add_argument("--morphology-delta", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional JSON report path; no report is written by default.",
    )
    return parser.parse_args()


def load_condition(args: argparse.Namespace) -> tuple[str, np.ndarray, np.ndarray]:
    map_dir = Path(args.spatial_maps_dir).expanduser().resolve()
    morphology_path = Path(args.morphology_table).expanduser().resolve()
    if not map_dir.is_dir():
        raise FileNotFoundError(f"Spatial-map directory not found: {map_dir}")
    if not morphology_path.is_file():
        raise FileNotFoundError(f"Morphology table not found: {morphology_path}")

    morphology_frame = pd.read_parquet(morphology_path)
    morphology_frame.index = morphology_frame.index.astype(str)
    if morphology_frame.shape[1] != 16:
        raise ValueError(
            f"Expected 16 morphology columns, found {morphology_frame.shape[1]}"
        )
    matches = sorted(
        {path.stem for path in map_dir.glob("*.npz")} & set(morphology_frame.index)
    )
    if args.stem is not None:
        if args.stem not in matches:
            raise ValueError(f"No matched condition found for stem: {args.stem}")
        stem = args.stem
    elif matches:
        stem = matches[0]
    else:
        raise ValueError("No matching spatial-map and morphology-table stems found")

    with np.load(map_dir / f"{stem}.npz", allow_pickle=False) as payload:
        spatial_map = payload["map"]
    morphology = morphology_frame.loc[stem].to_numpy(dtype=np.float32, copy=True)
    if spatial_map.shape != (512, 512, 5):
        raise ValueError(f"Unexpected spatial-map shape: {spatial_map.shape}")
    if morphology.shape != (16,):
        raise ValueError(f"Unexpected morphology shape: {morphology.shape}")
    return stem, spatial_map, morphology


def seeded_noise(shape: tuple[int, ...], seed: int, device, dtype) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=torch.float32).to(
        device=device, dtype=dtype
    )


def predict(
    models,
    noisy_latents: torch.Tensor,
    spatial_features: torch.Tensor,
    morphology: torch.Tensor,
    text_embeddings: torch.Tensor,
    timestep: torch.Tensor,
) -> torch.Tensor:
    with film_condition(models.unet, morphology):
        return models.unet(
            torch.cat([noisy_latents, spatial_features], dim=1),
            timestep,
            encoder_hidden_states=text_embeddings,
            return_dict=False,
        )[0]


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    stem, spatial_map, morphology = load_condition(args)
    models = load_phase2_conditioning_models(
        args.checkpoint,
        device=args.device,
        dtype=args.dtype,
    )

    spatial = (
        torch.from_numpy(spatial_map.astype(np.float32) / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=models.device, dtype=models.dtype)
    )
    morphology_tensor = torch.from_numpy(morphology).unsqueeze(0).to(
        device=models.device, dtype=models.dtype
    )
    noisy_latents = seeded_noise(
        (1, 4, 64, 64), args.seed, models.device, models.dtype
    )
    text_embeddings = seeded_noise(
        (1, 77, 1024), args.seed + 1, models.device, models.dtype
    )
    timestep = torch.tensor([args.timestep], device=models.device, dtype=torch.long)

    spatial_features = models.spatial_encoder(spatial)
    zero_spatial_features = models.spatial_encoder(torch.zeros_like(spatial))
    baseline = predict(
        models,
        noisy_latents,
        spatial_features,
        morphology_tensor,
        text_embeddings,
        timestep,
    )
    without_spatial = predict(
        models,
        noisy_latents,
        zero_spatial_features,
        morphology_tensor,
        text_embeddings,
        timestep,
    )
    shifted_morphology = predict(
        models,
        noisy_latents,
        spatial_features,
        morphology_tensor + args.morphology_delta,
        text_embeddings,
        timestep,
    )

    spatial_delta = float((baseline - without_spatial).abs().mean().item())
    morphology_delta = float((baseline - shifted_morphology).abs().mean().item())
    finite = bool(
        torch.isfinite(baseline).all()
        and torch.isfinite(without_spatial).all()
        and torch.isfinite(shifted_morphology).all()
    )
    if tuple(baseline.shape) != (1, 4, 64, 64):
        raise RuntimeError(f"Unexpected prediction shape: {tuple(baseline.shape)}")
    if not finite:
        raise RuntimeError("Checkpoint produced NaN or infinite values")
    if spatial_delta <= 1e-7 or morphology_delta <= 1e-7:
        raise RuntimeError("One or more conditioning paths did not affect the output")

    report = {
        "status": "pass",
        "checkpoint": str(models.checkpoint_dir),
        "tile_stem": stem,
        "device": str(models.device),
        "dtype": str(models.dtype).removeprefix("torch."),
        "prediction_shape": list(baseline.shape),
        "all_values_finite": finite,
        "film_block_count": sum(
            hasattr(module, "film_mlp") for module in models.unet.modules()
        ),
        "spatial_ablation_mean_absolute_delta": spatial_delta,
        "morphology_ablation_mean_absolute_delta": morphology_delta,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
