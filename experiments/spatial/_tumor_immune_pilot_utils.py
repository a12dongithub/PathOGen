"""Shared helpers for stronger tumor–immune spatial pilot interventions."""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor

from cpathogen.counterfactuals.centroids import render_centroid_channel

TUMOR_CHANNEL = 0
INFLAMMATORY_CHANNEL = 1


def normalized_weights(values: Tensor) -> Tensor:
    values = values.detach().to(dtype=torch.float32).clamp_min(0.0)
    maximum = float(values.max())
    if maximum <= 0.0:
        raise ValueError("Tumor spatial channel has no positive signal")
    return values / maximum


def outer_tumor_ring_weights(tumor: Tensor, radius_px: int = 24) -> Tensor:
    """Return a soft band immediately outside tumor-rich regions."""
    normalized = normalized_weights(tumor)
    kernel = 2 * radius_px + 1
    dilated = functional.max_pool2d(
        normalized[None, None], kernel_size=kernel, stride=1, padding=radius_px
    )[0, 0]
    weights = (dilated - normalized).clamp_min(0.0) * dilated
    if float(weights.sum()) <= 0.0:
        return normalized
    return weights


def intratumoral_weights(tumor: Tensor) -> Tensor:
    """Prefer the most tumor-rich pixels for infiltrating immune centroids."""
    normalized = normalized_weights(tumor)
    weights = normalized.square()
    if float(weights.sum()) <= 0.0:
        raise ValueError("Could not construct intratumoral sampling weights")
    return weights


def sample_nested_centroids(
    weights: Tensor,
    count: int,
    *,
    rng: random.Random,
    minimum_distance_px: float = 5.0,
) -> np.ndarray:
    """Draw deterministic, spatially separated centroids from a soft target map."""
    if count < 1:
        raise ValueError("count must be positive")
    values = weights.detach().cpu().numpy().astype(np.float64, copy=False)
    cumulative = np.cumsum(values.ravel())
    if not cumulative.size or cumulative[-1] <= 0.0:
        raise ValueError("Centroid sampling weights have no positive mass")
    height, width = values.shape
    accepted: list[tuple[int, int]] = []
    minimum_squared = minimum_distance_px**2
    for _ in range(count):
        chosen: tuple[int, int] | None = None
        for attempt in range(2048):
            index = int(np.searchsorted(cumulative, rng.random() * cumulative[-1]))
            y, x = divmod(min(index, values.size - 1), width)
            required = minimum_squared if attempt < 1024 else 4.0
            if all((x - px) ** 2 + (y - py) ** 2 >= required for px, py in accepted):
                chosen = (x, y)
                break
        if chosen is None:
            raise RuntimeError("Could not place separated tumor-directed centroids")
        accepted.append(chosen)
    return np.asarray(accepted, dtype=np.int16)


def add_centroid_signal(original: Tensor, centroids: np.ndarray, strength: float) -> Tensor:
    rendered = render_centroid_channel(centroids).to(device=original.device)
    return torch.clamp(original + strength * rendered, 0.0, 1.0)


def boundary_transfer_weights(tumor: Tensor, radius_px: int = 18) -> Tensor:
    """Return a normalized soft tumor-boundary mask on both sides of the interface."""
    normalized = normalized_weights(tumor)
    kernel = 2 * radius_px + 1
    local_max = functional.max_pool2d(
        normalized[None, None], kernel_size=kernel, stride=1, padding=radius_px
    )[0, 0]
    local_min = -functional.max_pool2d(
        -normalized[None, None], kernel_size=kernel, stride=1, padding=radius_px
    )[0, 0]
    contrast = (local_max - local_min).clamp_min(0.0)
    maximum = float(contrast.max())
    return contrast / maximum if maximum > 0.0 else normalized
