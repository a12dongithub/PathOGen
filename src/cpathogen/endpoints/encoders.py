"""Frozen pathology encoder registry and resumable embedding extraction."""

from __future__ import annotations

import gc
import hashlib
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


@dataclass
class EncoderBundle:
    name: str
    model: nn.Module
    transform: Callable[[Image.Image], torch.Tensor]
    forward: Callable[[torch.Tensor], torch.Tensor]
    feature_dim: int
    model_id: str


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _prepare_model(model: nn.Module, device: torch.device) -> nn.Module:
    model.eval().requires_grad_(False).to(device)
    return model


def build_encoder(
    name: str,
    *,
    device: torch.device,
    ctranspath_checkpoint: Path | None = None,
) -> EncoderBundle:
    """Build one frozen encoder using its released preprocessing contract."""
    key = name.lower().replace("-", "").replace("_", "")
    if key == "debugrgb":

        class ChannelMean(nn.Module):
            def forward(self, images: torch.Tensor) -> torch.Tensor:
                return images.mean(dim=(2, 3))

        model = _prepare_model(ChannelMean(), device)
        transform = transforms.Compose(
            [transforms.Resize((32, 32)), transforms.ToTensor()]
        )
        return EncoderBundle(
            "debug_rgb", model, transform, model, 3, "internal/smoke-test-channel-mean"
        )
    if key == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
        model.fc = nn.Identity()
        model = _prepare_model(model, device)
        return EncoderBundle(
            "resnet50",
            model,
            weights.transforms(),
            model,
            2048,
            "torchvision/resnet50-imagenet1k-v2",
        )

    if key == "resnet50clam":
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        backbone = models.resnet50(weights=weights)
        model = nn.Sequential(
            *list(backbone.children())[:-3], nn.AdaptiveAvgPool2d(1), nn.Flatten()
        )
        model = _prepare_model(model, device)
        return EncoderBundle(
            "resnet50_clam",
            model,
            weights.transforms(),
            model,
            1024,
            "torchvision/resnet50-imagenet1k-v1-through-layer3",
        )

    if key == "ctranspath":
        if ctranspath_checkpoint is None or not ctranspath_checkpoint.is_file():
            raise FileNotFoundError(
                "--ctranspath-checkpoint is required for CTransPath"
            )
        from cpathogen.probes.ctranspath import build_encoder as build_ctranspath
        from cpathogen.probes.ctranspath import image_transform

        model = _prepare_model(build_ctranspath(ctranspath_checkpoint, device), device)
        return EncoderBundle(
            "ctranspath",
            model,
            image_transform(),
            model,
            768,
            "jamesdolezal/CTransPath",
        )

    import timm
    from timm.data import create_transform, resolve_model_data_config

    if key == "uni2h":
        model = timm.create_model(
            "hf_hub:MahmoodLab/UNI2-h",
            pretrained=True,
            img_size=224,
            patch_size=14,
            depth=24,
            num_heads=24,
            init_values=1e-5,
            embed_dim=1536,
            mlp_ratio=2.66667 * 2,
            num_classes=0,
            no_embed_class=True,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
            reg_tokens=8,
            dynamic_img_size=True,
        )
        model = _prepare_model(model, device)
        transform = create_transform(
            **resolve_model_data_config(model), is_training=False
        )
        return EncoderBundle(
            "uni2h", model, transform, model, 1536, "MahmoodLab/UNI2-h"
        )

    if key == "virchow2":
        model = timm.create_model(
            "hf_hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        model = _prepare_model(model, device)
        transform = create_transform(
            **resolve_model_data_config(model), is_training=False
        )

        def virchow_forward(images: torch.Tensor) -> torch.Tensor:
            tokens = model(images)
            return torch.cat((tokens[:, 0], tokens[:, 5:].mean(dim=1)), dim=-1)

        return EncoderBundle(
            "virchow2",
            model,
            transform,
            virchow_forward,
            2560,
            "paige-ai/Virchow2",
        )

    if key == "uni":
        from huggingface_hub import hf_hub_download

        model = timm.create_model(
            "vit_large_patch16_224",
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
        )
        checkpoint = hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(payload, strict=True)
        model = _prepare_model(model, device)
        transform = create_transform(
            **resolve_model_data_config(model), is_training=False
        )
        return EncoderBundle("uni", model, transform, model, 1024, "MahmoodLab/UNI")

    if key == "conch":
        from conch.open_clip_custom import create_model_from_pretrained

        token = os.environ.get("HF_TOKEN")
        model, preprocess = create_model_from_pretrained(
            "conch_ViT-B-16",
            checkpoint_path="hf_hub:MahmoodLab/CONCH",
            hf_auth_token=token,
        )
        model = _prepare_model(model, device)

        def conch_forward(images: torch.Tensor) -> torch.Tensor:
            return model.encode_image(images, proj_contrast=False, normalize=False)

        return EncoderBundle(
            "conch", model, preprocess, conch_forward, 512, "MahmoodLab/CONCH"
        )
    raise ValueError(f"Unknown encoder: {name}")


class TileDataset(Dataset):
    def __init__(
        self,
        paths: list[Path],
        transform: Callable[[Image.Image], torch.Tensor],
        augmentation_codes: list[int] | None = None,
    ) -> None:
        self.paths = paths
        self.transform = transform
        self.augmentation_codes = augmentation_codes or [0] * len(paths)
        if len(self.augmentation_codes) != len(paths):
            raise ValueError("augmentation_codes length does not match paths")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as source:
            image = source.convert("RGB")
        code = self.augmentation_codes[index]
        if code == 1:
            image = ImageOps.mirror(image)
        elif code == 2:
            image = ImageOps.flip(image)
        elif code == 3:
            image = image.transpose(Image.Transpose.ROTATE_180)
        elif code == 4:
            image = image.transpose(Image.Transpose.ROTATE_90)
        elif code == 5:
            image = image.transpose(Image.Transpose.ROTATE_270)
        elif code != 0:
            raise ValueError(f"Unknown deterministic augmentation code: {code}")
        return self.transform(image)


def _forward_with_oom_split(
    bundle: EncoderBundle, images: torch.Tensor, device: torch.device
) -> torch.Tensor:
    try:
        prepared = images.to(device=device, dtype=torch.float32, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(
                device.type == "cuda"
                and bundle.name not in {"debug_rgb", "resnet50", "resnet50_clam"}
            ),
        ):
            return bundle.forward(prepared)
    except torch.OutOfMemoryError:
        if len(images) == 1:
            raise
        torch.cuda.empty_cache()
        midpoint = len(images) // 2
        first = _forward_with_oom_split(bundle, images[:midpoint], device)
        second = _forward_with_oom_split(bundle, images[midpoint:], device)
        return torch.cat((first, second), dim=0)


def extract_embeddings(
    bundle: EncoderBundle,
    paths: list[Path],
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    description: str,
    augmentation_codes: list[int] | None = None,
) -> np.ndarray:
    """Extract L2-normalized embeddings, splitting a batch only on CUDA OOM."""
    loader = DataLoader(
        TileDataset(paths, bundle.transform, augmentation_codes),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for images in tqdm(loader, desc=description):
            embeddings = _forward_with_oom_split(bundle, images, device).float()
            if embeddings.ndim != 2:
                raise ValueError(
                    f"{bundle.name} returned shape {tuple(embeddings.shape)}, expected [B, D]"
                )
            embeddings = torch.nn.functional.normalize(embeddings, dim=1)
            chunks.append(embeddings.cpu().numpy().astype(np.float32, copy=False))
    if not chunks:
        raise ValueError(f"No images supplied for {description}")
    result = np.concatenate(chunks, axis=0)
    if result.shape != (len(paths), bundle.feature_dim):
        raise ValueError(
            f"{bundle.name} produced {result.shape}; expected {(len(paths), bundle.feature_dim)}"
        )
    return result


def extract_embeddings_sharded(
    bundle: EncoderBundle,
    paths: list[Path],
    item_ids: np.ndarray,
    *,
    shard_root: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    description: str,
    augmentation_codes: list[int] | None = None,
    shard_size: int = 1_024,
) -> np.ndarray:
    """Extract embeddings in manifest-keyed shards that survive Colab resets."""
    if len(paths) != len(item_ids):
        raise ValueError("paths and item_ids differ")
    shard_size = max(shard_size, batch_size)
    signature = hashlib.sha256("\n".join(map(str, item_ids)).encode()).hexdigest()[:16]
    shard_dir = shard_root / f"{bundle.name}_{signature}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    chunks = []
    for start in range(0, len(paths), shard_size):
        stop = min(start + shard_size, len(paths))
        shard = shard_dir / f"{start:07d}_{stop:07d}.npy"
        expected_shape = (stop - start, bundle.feature_dim)
        values = None
        if shard.is_file():
            candidate = np.load(shard, allow_pickle=False)
            if candidate.shape == expected_shape and np.isfinite(candidate).all():
                values = candidate.astype(np.float32, copy=False)
        if values is None:
            codes = (
                None if augmentation_codes is None else augmentation_codes[start:stop]
            )
            values = extract_embeddings(
                bundle,
                paths[start:stop],
                device=device,
                batch_size=batch_size,
                num_workers=num_workers,
                description=f"{description} [{start}:{stop}]",
                augmentation_codes=codes,
            )
            temporary = shard.with_suffix(".npy.part")
            with temporary.open("wb") as handle:
                np.save(handle, values, allow_pickle=False)
            temporary.replace(shard)
        else:
            print(f"[{description}] reused shard {start}:{stop}", flush=True)
        chunks.append(values)
    if not chunks:
        raise ValueError(f"No images supplied for {description}")
    return np.concatenate(chunks, axis=0)


def release_encoder(bundle: EncoderBundle) -> None:
    bundle.model.to("cpu")
    del bundle.model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
