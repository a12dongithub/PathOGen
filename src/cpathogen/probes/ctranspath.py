"""Shared frozen CTransPath embedding utilities."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import timm_ctp
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ConvStem(nn.Module):
    """Convolutional patch stem required by the released weights."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Any | None = None,
        flatten: bool = True,
        **_: Any,
    ) -> None:
        super().__init__()
        image_size = (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)
        patch_dimensions = (
            (patch_size, patch_size) if isinstance(patch_size, int) else tuple(patch_size)
        )
        if patch_dimensions != (4, 4) or in_chans != 3:
            raise ValueError("CTransPath expects RGB input and patch_size=4")
        self.img_size = image_size
        self.patch_size = patch_dimensions
        self.grid_size = (
            image_size[0] // patch_dimensions[0],
            image_size[1] // patch_dimensions[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        stem: list[nn.Module] = []
        input_dim, output_dim = in_chans, embed_dim // 8
        for _ in range(2):
            stem.extend(
                (
                    nn.Conv2d(
                        input_dim,
                        output_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(output_dim),
                    nn.ReLU(inplace=True),
                )
            )
            input_dim, output_dim = output_dim, output_dim * 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.proj(inputs)
        if self.flatten:
            features = features.flatten(2).transpose(1, 2)
        return self.norm(features)


def build_encoder(checkpoint: Path, device: torch.device) -> nn.Module:
    model = timm_ctp.create_model(
        "swin_tiny_patch4_window7_224",
        embed_layer=ConvStem,
        pretrained=False,
    )
    model.head = nn.Identity()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.requires_grad_(False)
    return model.to(device)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def image_transform() -> transforms.Compose:
    return transforms.Compose(
        (
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        )
    )


class ImageDataset(Dataset):
    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths
        self.transform = image_transform()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


def extract_embeddings(
    model: nn.Module,
    paths: list[Path],
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    description: str,
) -> np.ndarray:
    loader = DataLoader(
        ImageDataset(paths),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for images in tqdm(loader, desc=description):
            embeddings = model(images.to(device, non_blocking=True))
            embeddings = torch.nn.functional.normalize(embeddings, dim=1)
            chunks.append(embeddings.cpu().numpy().astype(np.float32, copy=False))
    if not chunks:
        raise ValueError(f"No images supplied for {description}")
    return np.concatenate(chunks, axis=0)


def binary_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = (probability >= 0.5).astype(np.int64)
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "f1": float(f1_score(y_true, prediction)),
        "log_loss": float(log_loss(y_true, probability, labels=[0, 1])),
    }


def patient_from_stem(stem: str) -> str | None:
    parts = stem.split("_")[0].split("-")
    if len(parts) >= 3 and parts[0] == "TCGA":
        return "-".join(parts[:3])
    return None


def knob_sd(parameters: str, condition: str) -> float:
    if condition == "baseline":
        return 0.0
    import json

    try:
        value = json.loads(parameters).get("sd_steps")
        if value is not None:
            return float(value)
    except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
        pass
    raise ValueError(f"No sd_steps found for condition {condition!r}")
