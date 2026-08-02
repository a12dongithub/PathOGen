"""Validated tile datasets shared by both diffusion-training phases."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional
from torch.utils.data import Dataset

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def _load_rgb(path: Path, resolution: int) -> Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (resolution, resolution):
            image = image.resize((resolution, resolution), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1).div_(127.5).sub_(1.0)


def _load_spatial_map(path: Path, resolution: int) -> Tensor:
    with np.load(path, allow_pickle=False) as payload:
        if "map" not in payload:
            raise ValueError(f"Spatial map has no 'map' array: {path}")
        array = payload["map"]
    if array.ndim != 3:
        raise ValueError(
            f"Expected a three-dimensional spatial map, found {array.shape}: {path}"
        )
    if array.shape[-1] == 5:
        tensor = torch.from_numpy(array.astype(np.float32)).permute(2, 0, 1)
    elif array.shape[0] == 5:
        tensor = torch.from_numpy(array.astype(np.float32))
    else:
        raise ValueError(f"Expected five spatial channels, found {array.shape}: {path}")
    if float(tensor.max()) > 1.0:
        tensor = tensor / 255.0
    if tensor.shape[-2:] != (resolution, resolution):
        tensor = functional.interpolate(
            tensor.unsqueeze(0),
            size=(resolution, resolution),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    if not torch.isfinite(tensor).all():
        raise ValueError(f"Spatial map contains non-finite values: {path}")
    return tensor.clamp_(0.0, 1.0)


def _image_index(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Tile directory not found: {directory}")
    output: dict[str, Path] = {}
    for suffix in IMAGE_SUFFIXES:
        for path in sorted(directory.glob(f"*{suffix}")):
            if path.stem in output:
                raise ValueError(
                    f"More than one image has stem {path.stem!r} in {directory}"
                )
            output[path.stem] = path
    if not output:
        raise ValueError(f"No PNG or JPEG tiles found in {directory}")
    return output


class Phase1TileDataset(Dataset[dict[str, object]]):
    """H&E tiles and prompts described by an ImageFolder-style JSONL file."""

    def __init__(
        self,
        metadata_file: str | Path,
        *,
        resolution: int = 512,
        max_samples: int | None = None,
        random_flip: bool = True,
    ) -> None:
        self.metadata_file = Path(metadata_file).expanduser().resolve()
        if not self.metadata_file.is_file():
            raise FileNotFoundError(
                f"Training metadata not found: {self.metadata_file}"
            )
        self.resolution = resolution
        self.random_flip = random_flip
        records: list[tuple[Path, str]] = []
        for line_number, line in enumerate(
            self.metadata_file.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                relative_name = record["file_name"]
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(
                    f"Invalid metadata row {line_number} in {self.metadata_file}"
                ) from error
            image_path = Path(relative_name).expanduser()
            if not image_path.is_absolute():
                image_path = (self.metadata_file.parent / image_path).resolve()
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Metadata row {line_number} points to a missing tile: {image_path}"
                )
            records.append((image_path, str(record.get("text", "he"))))
        if max_samples is not None:
            records = records[:max_samples]
        if not records:
            raise ValueError(f"No training records found in {self.metadata_file}")
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        image_path, prompt = self.records[index]
        pixels = _load_rgb(image_path, self.resolution)
        if self.random_flip and bool(torch.rand(()) < 0.5):
            pixels = pixels.flip(-1)
        return {
            "pixel_values": pixels,
            "prompt": prompt,
            "stem": image_path.stem,
        }


class Phase2ConditionDataset(Dataset[dict[str, object]]):
    """Stem-aligned tiles, five-channel maps, and 16-value feature vectors."""

    def __init__(
        self,
        tiles_dir: str | Path,
        spatial_maps_dir: str | Path,
        morphology_table: str | Path,
        *,
        resolution: int = 512,
        prompt: str = "he",
        max_samples: int | None = None,
        random_flip: bool = True,
    ) -> None:
        self.tiles_dir = Path(tiles_dir).expanduser().resolve()
        self.spatial_maps_dir = Path(spatial_maps_dir).expanduser().resolve()
        self.morphology_table = Path(morphology_table).expanduser().resolve()
        self.resolution = resolution
        self.prompt = prompt
        self.random_flip = random_flip

        images = _image_index(self.tiles_dir)
        if not self.spatial_maps_dir.is_dir():
            raise FileNotFoundError(
                f"Spatial-map directory not found: {self.spatial_maps_dir}"
            )
        if not self.morphology_table.is_file():
            raise FileNotFoundError(
                f"Morphology table not found: {self.morphology_table}"
            )

        morphology = pd.read_parquet(self.morphology_table)
        if morphology.index.has_duplicates:
            raise ValueError("Morphology table contains duplicate stems")
        if morphology.shape[1] != 16:
            raise ValueError(
                f"Morphology table must contain exactly 16 features, found {morphology.shape[1]}"
            )
        missing_maps = sorted(
            stem
            for stem in images
            if not (self.spatial_maps_dir / f"{stem}.npz").is_file()
        )
        missing_morphology = sorted(
            stem for stem in images if stem not in morphology.index
        )
        if missing_maps or missing_morphology:
            raise ValueError(
                "Phase-2 inputs are not aligned by stem. "
                f"Missing maps: {missing_maps[:5]}; missing morphology: {missing_morphology[:5]}"
            )

        stems = sorted(images)
        if max_samples is not None:
            stems = stems[:max_samples]
        values = morphology.loc[stems].to_numpy(dtype=np.float32, copy=True)
        if not np.isfinite(values).all():
            raise ValueError("Selected morphology rows contain NaN or infinite values")
        self.stems = stems
        self.images = images
        self.morphology = torch.from_numpy(values)
        self.feature_names = tuple(str(column) for column in morphology.columns)

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, index: int) -> dict[str, object]:
        stem = self.stems[index]
        pixels = _load_rgb(self.images[stem], self.resolution)
        spatial = _load_spatial_map(
            self.spatial_maps_dir / f"{stem}.npz", self.resolution
        )
        if self.random_flip and bool(torch.rand(()) < 0.5):
            pixels = pixels.flip(-1)
            spatial = spatial.flip(-1)
        return {
            "pixel_values": pixels,
            "spatial_maps": spatial,
            "morphology": self.morphology[index],
            "prompt": self.prompt,
            "stem": stem,
        }


def make_collate_fn(
    tokenizer: object,
) -> Callable[[Sequence[dict[str, object]]], dict[str, object]]:
    """Build a collator that tokenizes prompts and stacks tensor fields."""

    def collate(examples: Sequence[dict[str, object]]) -> dict[str, object]:
        prompts = [str(example["prompt"]) for example in examples]
        encoded = tokenizer(
            prompts,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        batch: dict[str, object] = {
            "pixel_values": torch.stack(
                [example["pixel_values"] for example in examples]
            ).contiguous(),
            "input_ids": encoded.input_ids,
            "stems": [str(example["stem"]) for example in examples],
        }
        if "spatial_maps" in examples[0]:
            batch["spatial_maps"] = torch.stack(
                [example["spatial_maps"] for example in examples]
            ).contiguous()
            batch["morphology"] = torch.stack(
                [example["morphology"] for example in examples]
            ).contiguous()
        return batch

    return collate
