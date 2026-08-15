"""Centroid controls that reproduce the training-time spatial-map encoding."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch import Tensor

IMAGE_SIZE = 512
SPATIAL_SIGMA = 3.0


def inflammatory_centroids_from_geojson(path: str | Path) -> np.ndarray:
    """Extract the same rounded inflammatory polygon centers used in training."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    features = payload if isinstance(payload, list) else payload.get("features", [])
    centroids: list[tuple[int, int]] = []
    for feature in features:
        classification = (
            feature.get("properties", {}).get("classification", {}).get("name")
        )
        if classification != "Inflammatory":
            continue
        geometry = feature.get("geometry", {})
        coordinates = geometry.get("coordinates", [])
        polygons = []
        if geometry.get("type") == "Polygon" and coordinates:
            polygons.append(np.asarray(coordinates[0], dtype=np.int32))
        elif geometry.get("type") == "MultiPolygon":
            polygons.extend(
                np.asarray(part[0], dtype=np.int32) for part in coordinates if part
            )
        for polygon in polygons:
            if len(polygon) == 0:
                continue
            x = round(float(np.mean(polygon[:, 0])))
            y = round(float(np.mean(polygon[:, 1])))
            if 0 <= x < IMAGE_SIZE and 0 <= y < IMAGE_SIZE:
                centroids.append((x, y))
    return np.asarray(centroids, dtype=np.int16).reshape(-1, 2)


def load_inflammatory_centroids(data_root: str | Path, stem: str) -> np.ndarray:
    path = Path(data_root) / "cell_centroids" / f"{stem}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Inflammatory-centroid control not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if "inflammatory_xy" not in archive:
            raise ValueError(f"Missing 'inflammatory_xy' in centroid control: {path}")
        centroids = np.asarray(archive["inflammatory_xy"], dtype=np.int16)
    if centroids.ndim != 2 or centroids.shape[1] != 2:
        raise ValueError(f"Expected Nx2 inflammatory centroids in {path}")
    if len(centroids) == 0:
        raise ValueError(f"No inflammatory centroids in {path}")
    if (centroids < 0).any() or (centroids >= IMAGE_SIZE).any():
        raise ValueError(f"Out-of-bounds inflammatory centroid in {path}")
    return centroids


def load_centroid_reference_stats(
    data_root: str | Path,
) -> dict[str, float | int | str]:
    path = Path(data_root) / "cell_centroids" / "reference_stats.json"
    if not path.is_file():
        raise FileNotFoundError(f"Centroid reference statistics not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"transform", "sqrt_count_sd", "reference_count"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"Centroid reference statistics are missing: {missing}")
    if payload["transform"] != "sqrt":
        raise ValueError(f"Unsupported centroid-count transform in {path}")
    if float(payload["sqrt_count_sd"]) <= 0.0:
        raise ValueError(f"sqrt_count_sd must be positive in {path}")
    return payload


def render_centroid_channel(centroids: np.ndarray) -> Tensor:
    """Render impulses using the original blur, peak normalization, and uint8 step."""
    impulses = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    for x, y in centroids:
        impulses[int(y), int(x)] += 1.0
    blurred = gaussian_filter(impulses, sigma=SPATIAL_SIGMA)
    maximum = float(blurred.max(initial=0.0))
    if maximum > 0.0:
        blurred /= maximum
    quantized = (np.clip(blurred, 0.0, 1.0) * 255.0).astype(np.uint8)
    return torch.from_numpy(quantized.astype(np.float32) / 255.0)


def sd_target_count(original_count: int, sd_steps: float, sqrt_count_sd: float) -> int:
    """Shift sqrt(count) by a reference SD and map back to an integer count."""
    if original_count < 1:
        raise ValueError("At least one original inflammatory centroid is required")
    if sd_steps <= 0.0:
        raise ValueError("sd_steps must be positive")
    if sqrt_count_sd <= 0.0:
        raise ValueError("sqrt_count_sd must be positive")
    shifted = math.sqrt(original_count) + sd_steps * sqrt_count_sd
    return math.floor(shifted**2 + 0.5)


def add_jittered_centroids(
    original: np.ndarray,
    addition_count: int,
    *,
    rng: random.Random,
    jitter_sigma: float = 12.0,
    minimum_distance: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Add nuclei near the original distribution with deterministic nested draws."""
    if addition_count < 1:
        raise ValueError("addition_count must be positive")
    original = np.asarray(original, dtype=np.int16).reshape(-1, 2)
    accepted = [tuple(map(int, point)) for point in original]
    additions: list[tuple[int, int]] = []
    for _ in range(addition_count):
        candidate: tuple[int, int] | None = None
        for attempt in range(512):
            parent_x, parent_y = original[rng.randrange(len(original))]
            x = round(float(parent_x) + rng.gauss(0.0, jitter_sigma))
            y = round(float(parent_y) + rng.gauss(0.0, jitter_sigma))
            if not (0 <= x < IMAGE_SIZE and 0 <= y < IMAGE_SIZE):
                continue
            required_distance = minimum_distance if attempt < 256 else 2.0
            if all(
                (x - other_x) ** 2 + (y - other_y) ** 2 >= required_distance**2
                for other_x, other_y in accepted
            ):
                candidate = (x, y)
                break
        if candidate is None:
            raise RuntimeError(
                "Could not place a non-overlapping inflammatory centroid"
            )
        additions.append(candidate)
        accepted.append(candidate)
    added = np.asarray(additions, dtype=np.int16)
    combined = np.concatenate([original, added], axis=0)
    return combined, added


def centroid_sha256(centroids: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(centroids).tobytes()).hexdigest()
