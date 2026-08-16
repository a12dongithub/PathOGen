"""Shared helpers for stronger tumor–immune spatial pilot interventions."""

from __future__ import annotations

import random

import numpy as np
import torch
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
    accepted_set: set[tuple[int, int]] = set()
    minimum_squared = minimum_distance_px**2
    fallback_order: np.ndarray | None = None
    fallback_only = False

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
            if require_diagonal_spacing and any(
                neighbor in accepted_set
                for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))
            ):
                continue
            return point
        return None

    for _ in range(count):
        chosen: tuple[int, int] | None = None
        if not fallback_only:
            for attempt in range(2048):
                index = int(np.searchsorted(cumulative, rng.random() * cumulative[-1]))
                y, x = divmod(min(index, values.size - 1), width)
                required = minimum_squared if attempt < 1024 else 4.0
                if all(
                    (x - px) ** 2 + (y - py) ** 2 >= required
                    for px, py in accepted
                ):
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
    return np.asarray(accepted, dtype=np.int16)


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
