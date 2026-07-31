#!/usr/bin/env python
"""Download PathOGen assets and run one conditional 512x512 inference smoke test."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Iterable

import gdown
import numpy as np
import pandas as pd
import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from PIL import Image
from safetensors.torch import load_file as load_safetensors
from transformers import CLIPTextModel, CLIPTokenizer


DATA_ADDY = "https://drive.google.com/file/d/1sBc4-CexT3S2cw1LZysrX4BLjVjN6BPt/view?usp=sharing"
MODEL_ADDY = "https://drive.google.com/file/d/1QLymjt0qnjM2FM-oR5vRYB0B1URcg5wq/view?usp=sharing"
GIT_ADDY = "https://github.com/a12dongithub/PathOGen"

BASE_MODEL = "Manojb/stable-diffusion-2-1-base"
EXPECTED_DATA_GIB = 24.5
EXPECTED_MODEL_GIB = 5.6
MIN_DATA_PREP_FREE_GIB = 48.0
MIN_MODEL_PREP_FREE_GIB = 12.0
MIN_BOTH_PREP_FREE_GIB = 55.0

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = REPO_ROOT / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from compare_checkpoints import add_label, spatial_map_to_rgb_with_legend  # noqa: E402
from train_pathogen import SpatialCondEncoder, inject_film_into_unet  # noqa: E402
from validation_utils import generate_concat_conditioned  # noqa: E402


def gib(value: int) -> float:
    return value / (1024**3)


def free_gib(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    return gib(shutil.disk_usage(path).free)


def download_drive_zip(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"[assets] Resuming/reusing {destination} ({gib(destination.stat().st_size):.2f} GiB)")
    result = gdown.download(url=url, output=str(destination), quiet=False, fuzzy=True, resume=True)
    if not result or not destination.exists() or destination.stat().st_size == 0:
        raise RuntimeError(
            f"Google Drive download failed for {url}. Confirm that link sharing permits downloads."
        )
    print(f"[assets] Downloaded {destination.name}: {gib(destination.stat().st_size):.2f} GiB")
    return destination


def safe_extract_zip(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    marker = destination / f".{archive.stem}.extracted"
    if marker.exists():
        print(f"[assets] Extraction marker found; reusing {destination}")
        return

    print(f"[assets] Extracting {archive} -> {destination}")
    with zipfile.ZipFile(archive) as handle:
        members = handle.infolist()
        for member in members:
            target = (destination / member.filename).resolve()
            if target != destination_root and destination_root not in target.parents:
                raise RuntimeError(f"Unsafe ZIP member path: {member.filename}")
        for index, member in enumerate(members, start=1):
            handle.extract(member, destination)
            if index % 10_000 == 0 or index == len(members):
                print(f"[assets] Extracted {index:,}/{len(members):,} entries")
    marker.write_text("ok\n", encoding="utf-8")


def find_checkpoint_dir(root: Path) -> Path:
    candidates: Iterable[Path] = (root, *root.rglob("checkpoint-*"))
    for candidate in candidates:
        if (
            candidate.is_dir()
            and (candidate / "unet").is_dir()
            and (candidate / "film_mlps.pt").is_file()
            and (candidate / "spatial_encoder.pt").is_file()
        ):
            return candidate
    raise FileNotFoundError(f"Could not find a PathOGen checkpoint under {root}")


def morphology_file(root: Path) -> Path | None:
    options = (
        root / "morphology_stats.parquet",
        root / "morphology_features" / "morphology_stats.parquet",
    )
    return next((path for path in options if path.is_file()), None)


def find_data_dir(root: Path) -> Path:
    candidates: Iterable[Path] = (root, *root.rglob("512_final_dataset"))
    for candidate in candidates:
        if (
            candidate.is_dir()
            and (candidate / "images").is_dir()
            and (candidate / "spatial_maps").is_dir()
            and morphology_file(candidate) is not None
        ):
            return candidate

    for images_dir in root.rglob("images"):
        candidate = images_dir.parent
        if (candidate / "spatial_maps").is_dir() and morphology_file(candidate) is not None:
            return candidate
    raise FileNotFoundError(f"Could not find 512_final_dataset contents under {root}")


def prepare_assets(args: argparse.Namespace) -> tuple[Path, Path]:
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    data_search_root = args.data_root.resolve() if args.data_root else work_dir / "data"
    model_search_root = args.model_root.resolve() if args.model_root else work_dir / "model"

    data_dir = None
    checkpoint_dir = None
    if args.data_root:
        data_dir = find_data_dir(data_search_root)
    else:
        try:
            data_dir = find_data_dir(data_search_root)
            print(f"[assets] Reusing extracted data; download/extraction skipped: {data_dir}")
        except FileNotFoundError:
            pass

    if args.model_root:
        checkpoint_dir = find_checkpoint_dir(model_search_root)
    else:
        try:
            checkpoint_dir = find_checkpoint_dir(model_search_root)
            print(
                "[assets] Reusing extracted checkpoint; download/extraction skipped: "
                f"{checkpoint_dir}"
            )
        except FileNotFoundError:
            pass

    available = free_gib(work_dir)
    print(f"[disk] Initial free space: {available:.2f} GiB")
    if data_dir is None and checkpoint_dir is None:
        required = MIN_BOTH_PREP_FREE_GIB
    elif data_dir is None:
        required = MIN_DATA_PREP_FREE_GIB
    elif checkpoint_dir is None:
        required = MIN_MODEL_PREP_FREE_GIB
    else:
        required = 0.0

    if available < required:
        raise RuntimeError(
            f"Need at least {required:.0f} GiB free to prepare the missing assets; "
            f"only {available:.2f} GiB is available."
        )

    if data_dir is None:
        archive = download_drive_zip(args.data_url, work_dir / "downloads" / "data.zip")
        safe_extract_zip(archive, data_search_root)
        if not args.keep_archives:
            archive.unlink(missing_ok=True)
            print("[disk] Deleted data.zip after extraction")
        data_dir = find_data_dir(data_search_root)

    if checkpoint_dir is None:
        archive = download_drive_zip(args.model_url, work_dir / "downloads" / "model.zip")
        safe_extract_zip(archive, model_search_root)
        if not args.keep_archives:
            archive.unlink(missing_ok=True)
            print("[disk] Deleted model.zip after extraction")
        checkpoint_dir = find_checkpoint_dir(model_search_root)

    print(f"[assets] Dataset: {data_dir}")
    print(f"[assets] Checkpoint: {checkpoint_dir}")
    print(f"[disk] Free after extraction: {free_gib(work_dir):.2f} GiB")
    return data_dir, checkpoint_dir


def collect_samples(data_dir: Path, count: int, seed: int):
    morph_path = morphology_file(data_dir)
    if morph_path is None:
        raise FileNotFoundError("Morphology parquet was not found")
    morph_df = pd.read_parquet(morph_path)
    morph_df.index = morph_df.index.astype(str)

    images_dir = data_dir / "images"
    spatial_dir = data_dir / "spatial_maps"
    valid = []
    for image_path in sorted(images_dir.glob("*.png")):
        stem = image_path.stem
        spatial_path = spatial_dir / f"{stem}.npz"
        if spatial_path.is_file() and stem in morph_df.index:
            valid.append((image_path, spatial_path, stem))
    if not valid:
        raise RuntimeError("No aligned image/spatial-map/morphology samples were found")

    rng = random.Random(seed)
    selected = rng.sample(valid, min(count, len(valid)))
    real_images, spatial_maps, morphologies, stems = [], [], [], []
    for image_path, spatial_path, stem in selected:
        real_images.append(Image.open(image_path).convert("RGB"))
        spatial = np.load(spatial_path)["map"]
        if spatial.shape != (512, 512, 5):
            raise ValueError(f"Unexpected spatial map shape for {stem}: {spatial.shape}")
        spatial_maps.append(spatial)
        vector = morph_df.loc[stem].to_numpy(dtype=np.float32)
        if vector.shape != (16,) or not np.isfinite(vector).all():
            raise ValueError(f"Invalid morphology vector for {stem}: shape={vector.shape}")
        morphologies.append(torch.from_numpy(vector))
        stems.append(stem)
    return real_images, spatial_maps, morphologies, stems


def load_state_dict(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_models(checkpoint_dir: Path, device: torch.device, dtype: torch.dtype):
    print(f"[model] Loading base model: {BASE_MODEL}")
    tokenizer = CLIPTokenizer.from_pretrained(BASE_MODEL, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        BASE_MODEL, subfolder="text_encoder", torch_dtype=dtype
    ).to(device)
    vae = AutoencoderKL.from_pretrained(BASE_MODEL, subfolder="vae", torch_dtype=dtype).to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(BASE_MODEL, subfolder="scheduler")
    unet = UNet2DConditionModel.from_pretrained(BASE_MODEL, subfolder="unet", torch_dtype=dtype)

    old_conv = unet.conv_in
    new_conv = torch.nn.Conv2d(
        8,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
    ).to(dtype=dtype)
    with torch.no_grad():
        new_conv.weight[:, :4].copy_(old_conv.weight)
        new_conv.weight[:, 4:].zero_()
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)
    unet.conv_in = new_conv
    unet.config["in_channels"] = 8

    film_mlps = inject_film_into_unet(unet, film_dim=16)
    unet_weights = checkpoint_dir / "unet" / "diffusion_pytorch_model.safetensors"
    if not unet_weights.is_file():
        raise FileNotFoundError(f"Checkpoint UNet weights not found: {unet_weights}")

    # Load directly into the FiLM-injected architecture. Constructing a stock
    # UNet from this checkpoint makes Diffusers discard its custom FiLM keys.
    trained_state = load_safetensors(str(unet_weights), device="cpu")
    film_keys = {key for key in trained_state if ".film_mlp." in key}
    core_state = {key: value for key, value in trained_state.items() if key not in film_keys}
    missing, unexpected = unet.load_state_dict(core_state, strict=False)
    non_film_missing = [key for key in missing if ".film_mlp." not in key]
    if non_film_missing or unexpected:
        raise RuntimeError(
            "Checkpoint does not match the inference UNet: "
            f"{len(non_film_missing)} non-FiLM keys missing, "
            f"{len(unexpected)} unexpected keys"
        )
    del trained_state, core_state
    print(
        f"[model] UNet core loaded; {len(film_keys)} FiLM tensors deferred "
        "to film_mlps.pt"
    )

    spatial_encoder = SpatialCondEncoder().to(device=device, dtype=dtype)
    spatial_encoder.load_state_dict(load_state_dict(checkpoint_dir / "spatial_encoder.pt", device))
    film_mlps.load_state_dict(load_state_dict(checkpoint_dir / "film_mlps.pt", device))
    print("[model] Spatial encoder and FiLM weights loaded")

    unet.to(device=device, dtype=dtype)
    film_mlps.to(device=device, dtype=dtype)
    text_encoder.eval()
    vae.eval()
    return tokenizer, text_encoder, vae, noise_scheduler, unet, spatial_encoder


def save_results(
    output_dir: Path,
    stems: list[str],
    real_images: list[Image.Image],
    spatial_maps: list[np.ndarray],
    generated: list[Image.Image],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for stem, real, spatial, generated_image in zip(
        stems, real_images, spatial_maps, generated, strict=True
    ):
        real = real.resize((512, 512))
        generated_image = generated_image.resize((512, 512))
        spatial_for_display = spatial.astype(np.float32)
        if spatial_for_display.max() > 1.0:
            spatial_for_display /= 255.0
        spatial_rgb = spatial_map_to_rgb_with_legend(spatial_for_display).resize(
            (512, 512), Image.NEAREST
        )

        real.save(output_dir / f"{stem}_real.png")
        generated_image.save(output_dir / f"{stem}_generated.png")
        spatial_rgb.save(output_dir / f"{stem}_spatial.png")

        grid = Image.new("RGB", (1536, 512))
        grid.paste(add_label(spatial_rgb.copy(), "Spatial map"), (0, 0))
        grid.paste(add_label(real.copy(), "Observed H&E"), (512, 0))
        grid.paste(add_label(generated_image.copy(), "Generated H&E"), (1024, 0))
        grid.save(output_dir / f"{stem}_comparison.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-url", default=DATA_ADDY)
    parser.add_argument("--model-url", default=MODEL_ADDY)
    parser.add_argument("--work-dir", type=Path, default=Path("/content/pathogen_assets"))
    parser.add_argument("--output-dir", type=Path, default=Path("/content/pathogen_outputs"))
    parser.add_argument("--data-root", type=Path, help="Use an already-extracted local dataset")
    parser.add_argument("--model-root", type=Path, help="Use an already-extracted local checkpoint")
    parser.add_argument("--num-images", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-archives", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Validate extraction and aligned samples without loading the diffusion model",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_images < 1:
        raise ValueError("--num-images must be positive")

    started = time.perf_counter()
    data_dir, checkpoint_dir = prepare_assets(args)
    real_images, spatial_maps, morphologies, stems = collect_samples(
        data_dir, args.num_images, args.seed
    )
    print(f"[data] Selected {len(stems)} aligned sample(s): {stems}")
    if args.prepare_only:
        print("[done] Asset and sample validation passed")
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. In Colab choose Runtime > Change runtime type > GPU.")
    device = torch.device("cuda")
    dtype = torch.float16
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = gib(torch.cuda.get_device_properties(0).total_memory)
    print(f"[gpu] {gpu_name}, {gpu_memory:.1f} GiB")
    if "L4" not in gpu_name:
        print("[gpu] L4 is recommended for this smoke test; continuing on the assigned GPU.")

    tokenizer, text_encoder, vae, scheduler, unet, spatial_encoder = load_models(
        checkpoint_dir, device, dtype
    )
    generation_started = time.perf_counter()
    generated = generate_concat_conditioned(
        unet,
        vae,
        spatial_encoder,
        text_encoder,
        tokenizer,
        scheduler,
        spatial_maps,
        morphologies,
        device,
        dtype,
        num_inference_steps=args.steps,
        seed=args.seed,
    )
    generation_seconds = time.perf_counter() - generation_started
    save_results(args.output_dir, stems, real_images, spatial_maps, generated)

    manifest = {
        "experiment": "01_inference_smoke",
        "git_repository": GIT_ADDY,
        "gpu": gpu_name,
        "gpu_memory_gib": round(gpu_memory, 2),
        "checkpoint_dir": str(checkpoint_dir),
        "data_dir": str(data_dir),
        "stems": stems,
        "seed": args.seed,
        "steps": args.steps,
        "generation_seconds": round(generation_seconds, 3),
        "total_seconds": round(time.perf_counter() - started, 3),
        "torch": torch.__version__,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    print(f"[done] Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
