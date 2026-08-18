"""Shared helpers for stronger tumor–immune spatial pilot interventions."""

from __future__ import annotations

import random

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, maximum_filter
from torch import Tensor

from cpathogen.counterfactuals.centroids import IMAGE_SIZE, render_centroid_channel

TUMOR_CHANNEL = 0
INFLAMMATORY_CHANNEL = 1
TUMOR_NUCLEUS_DIAMETER_PX = 40.0
TUMOR_NUCLEUS_RADIUS_PX = TUMOR_NUCLEUS_DIAMETER_PX / 2.0
PERITUMORAL_RING_WIDTH_PX = TUMOR_NUCLEUS_DIAMETER_PX


def nearest_centroid_distance_map(centroids: np.ndarray) -> Tensor:
    """Return each pixel's Euclidean distance to the nearest supplied centroid."""
    points = np.asarray(centroids, dtype=np.int16).reshape(-1, 2)
    if len(points) == 0:
        raise ValueError("At least one centroid is required")
    if (
        np.any(points[:, 0] < 0)
        or np.any(points[:, 0] >= IMAGE_SIZE)
        or np.any(points[:, 1] < 0)
        or np.any(points[:, 1] >= IMAGE_SIZE)
    ):
        raise ValueError("Centroids must lie inside the 512 x 512 tile")
    non_centroid = np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
    non_centroid[points[:, 1], points[:, 0]] = False
    distances = distance_transform_edt(non_centroid).astype(np.float32)
    return torch.from_numpy(distances)


def tumor_centroid_annulus_weights(
    tumor_centroids: np.ndarray,
    *,
    inner_radius_px: float = TUMOR_NUCLEUS_RADIUS_PX,
    width_px: float = PERITUMORAL_RING_WIDTH_PX,
) -> Tensor:
    """Return a one-diameter-wide ring outside the assumed tumor radius."""
    if inner_radius_px < 0.0:
        raise ValueError("inner_radius_px cannot be negative")
    if width_px <= 0.0:
        raise ValueError("width_px must be positive")
    distances = nearest_centroid_distance_map(tumor_centroids)
    outer_radius_px = inner_radius_px + width_px
    ring = (distances >= inner_radius_px) & (distances <= outer_radius_px)
    weights = ring.to(dtype=torch.float32)
    if float(weights.sum()) <= 0.0:
        raise ValueError("Could not construct a peritumoral centroid annulus")
    return weights


def normalized_weights(values: Tensor) -> Tensor:
    values = values.detach().to(dtype=torch.float32).clamp_min(0.0)
    maximum = float(values.max())
    if maximum <= 0.0:
        raise ValueError("Tumor spatial channel has no positive signal")
    return values / maximum


def intratumoral_weights(tumor: Tensor) -> Tensor:
    """Prefer the most tumor-rich pixels for infiltrating immune centroids."""
    normalized = normalized_weights(tumor)
    weights = normalized.square()
    if float(weights.sum()) <= 0.0:
        raise ValueError("Could not construct intratumoral sampling weights")
    return weights


def outer_tumor_ring_weights(
    tumor: Tensor, *, radius_px: int = 24, core_threshold: float = 0.25
) -> Tensor:
    """Return a discrete soft band outside the tumor-control support."""
    if radius_px < 1:
        raise ValueError("radius_px must be positive")
    if not 0.0 < core_threshold < 1.0:
        raise ValueError("core_threshold must be between zero and one")
    normalized = normalized_weights(tumor)
    normalized_array = normalized.cpu().numpy()
    core = (normalized_array >= core_threshold).astype(np.float32)
    kernel = 2 * radius_px + 1
    dilated = maximum_filter(core, size=kernel, mode="constant", cval=0.0)
    ring = np.clip(dilated - core, 0.0, None)
    # Prefer the outer side of the interface and suppress residual tumor signal.
    weights = ring * np.clip(1.0 - normalized_array, 0.05, None)
    if float(weights.sum()) <= 0.0:
        raise ValueError("Could not construct an outer peritumoral ring")
    return torch.from_numpy(np.ascontiguousarray(weights, dtype=np.float32))


def sample_nested_centroids(
    weights: Tensor,
    count: int,
    *,
    rng: random.Random,
    minimum_distance_px: float = 5.0,
    forbidden_centroids: np.ndarray | None = None,
) -> np.ndarray:
    """Draw deterministic, spatially separated centroids from a soft target map."""
    if count < 1:
        raise ValueError("count must be positive")
    values = weights.detach().cpu().numpy().astype(np.float64, copy=False)
    cumulative = np.cumsum(values.ravel())
    if not cumulative.size or cumulative[-1] <= 0.0:
        raise ValueError("Centroid sampling weights have no positive mass")
    height, width = values.shape
    forbidden = (
        np.empty((0, 2), dtype=np.int16)
        if forbidden_centroids is None
        else np.asarray(forbidden_centroids, dtype=np.int16).reshape(-1, 2)
    )
    accepted: list[tuple[int, int]] = [tuple(map(int, point)) for point in forbidden]
    accepted_set: set[tuple[int, int]] = set(accepted)
    additions: list[tuple[int, int]] = []
    minimum_squared = minimum_distance_px**2
    preferred_blocked = np.zeros((height, width), dtype=bool)
    relaxed_blocked = np.zeros((height, width), dtype=bool)
    fallback_order: np.ndarray | None = None
    fallback_only = False

    def mark_blocked(mask: np.ndarray, point: tuple[int, int], distance_squared: float) -> None:
        x, y = point
        radius = int(np.ceil(np.sqrt(distance_squared)))
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask[y0:y1, x0:x1] |= (xx - x) ** 2 + (yy - y) ** 2 < distance_squared

    def register(point: tuple[int, int]) -> None:
        mark_blocked(preferred_blocked, point, minimum_squared)
        mark_blocked(relaxed_blocked, point, 4.0)

    for point in accepted:
        register(point)

    def weighted_fallback_order() -> np.ndarray:
        clone = random.Random()
        clone.setstate(rng.getstate())
        generator = np.random.default_rng(clone.getrandbits(64))
        positive = np.flatnonzero(values.ravel() > 0.0)
        if len(positive) < count:
            raise RuntimeError(
                "Tumor-directed map has fewer positive pixels than requested centroids"
            )
        draws = np.maximum(generator.random(len(positive)), np.finfo(float).tiny)
        keys = -np.log(draws) / values.ravel()[positive]
        return positive[np.argsort(keys, kind="stable")]

    def fallback_candidate(require_diagonal_spacing: bool) -> tuple[int, int] | None:
        nonlocal fallback_order
        if fallback_order is None:
            fallback_order = weighted_fallback_order()
        for index in fallback_order:
            y, x = divmod(int(index), width)
            point = (x, y)
            if point in accepted_set:
                continue
            if require_diagonal_spacing and relaxed_blocked[y, x]:
                continue
            return point
        return None

    for _ in range(count):
        chosen: tuple[int, int] | None = None
        if not fallback_only:
            for attempt in range(2048):
                index = int(np.searchsorted(cumulative, rng.random() * cumulative[-1]))
                y, x = divmod(min(index, values.size - 1), width)
                blocked = preferred_blocked if attempt < 1024 else relaxed_blocked
                if not blocked[y, x]:
                    chosen = (x, y)
                    break
        if chosen is None:
            fallback_only = True
            chosen = fallback_candidate(require_diagonal_spacing=True)
        if chosen is None:
            chosen = fallback_candidate(require_diagonal_spacing=False)
        if chosen is None:
            raise RuntimeError("Could not place unique tumor-directed centroids")
        accepted.append(chosen)
        accepted_set.add(chosen)
        additions.append(chosen)
        register(chosen)
    return np.asarray(additions, dtype=np.int16).reshape(-1, 2)


def minimum_centroid_distance(centroids: np.ndarray) -> float:
    values = np.asarray(centroids, dtype=np.float64).reshape(-1, 2)
    if len(values) < 2:
        return float("inf")
    differences = values[:, None, :] - values[None, :, :]
    squared = np.sum(differences**2, axis=-1)
    np.fill_diagonal(squared, np.inf)
    return float(np.sqrt(squared.min()))


def add_centroid_signal(original: Tensor, centroids: np.ndarray, strength: float) -> Tensor:
    rendered = render_centroid_channel(centroids).to(device=original.device)
    return torch.clamp(original + strength * rendered, 0.0, 1.0)
