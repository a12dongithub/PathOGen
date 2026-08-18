"""Relocate a fixed inflammatory-cell count along a mixing-to-separation axis."""

from __future__ import annotations

import random
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from torch import Tensor

from cpathogen.counterfactuals import (
    ConditionIntervention,
    InterventionContext,
    cell_centroids_by_class_from_geojson,
    render_centroid_channel,
)
from cpathogen.counterfactuals.centroids import IMAGE_SIZE, centroid_sha256
from experiments.spatial._tumor_immune_pilot_utils import (
    TUMOR_NUCLEUS_DIAMETER_PX,
    TUMOR_NUCLEUS_RADIUS_PX,
    nearest_centroid_distance_map,
)

TUMOR_CHANNEL = 0
INFLAMMATORY_CHANNEL = 1
RNG_NAMESPACE = "tumor-immune-centroid-separation-v3-tumor-diameter-40px"
GRID_SPACING_PX = 6
GRID_JITTER_PX = 1
MINIMUM_TUMOR_DISTANCE_PX = TUMOR_NUCLEUS_RADIUS_PX
DISTANCE_QUANTILE_WINDOW = 0.20

# Zero is maximally mixed under the no-overlapping-centroid constraint; one is
# maximally separated within the available tile area.
SEPARATION_LEVELS = (
    ("maximal_mixing", 0.00),
    ("low_separation", 0.25),
    ("intermediate", 0.50),
    ("high_separation", 0.75),
    ("maximal_segregation", 1.00),
)


def _geojson_path(context: InterventionContext) -> Path:
    path = context.store.data_root / "geojsons" / f"{context.original_stem}.geojson"
    if not path.is_file():
        raise FileNotFoundError(
            "Exact tumor-immune mixing requires source nuclei at "
            f"data_root/geojsons/<stem>.geojson; missing: {path}"
        )
    return path


def _rng_seed(context: InterventionContext) -> int:
    return context.rng(RNG_NAMESPACE).getrandbits(64)


def _candidate_lattice(rng: random.Random) -> np.ndarray:
    """Build a deterministic, near-Poisson candidate lattice for one tile."""
    offset_x = rng.randrange(GRID_SPACING_PX)
    offset_y = rng.randrange(GRID_SPACING_PX)
    points: list[tuple[int, int]] = []
    for y0 in range(offset_y, IMAGE_SIZE, GRID_SPACING_PX):
        for x0 in range(offset_x, IMAGE_SIZE, GRID_SPACING_PX):
            x = x0 + rng.randint(-GRID_JITTER_PX, GRID_JITTER_PX)
            y = y0 + rng.randint(-GRID_JITTER_PX, GRID_JITTER_PX)
            if 0 <= x < IMAGE_SIZE and 0 <= y < IMAGE_SIZE:
                points.append((x, y))
    return np.asarray(points, dtype=np.int16).reshape(-1, 2)


def _select_distance_quantile_band(
    candidates: np.ndarray,
    distances: np.ndarray,
    *,
    count: int,
    separation_fraction: float,
    rng: random.Random,
) -> np.ndarray:
    """Select an exact count from a monotone moving distance-quantile window."""
    if count < 1:
        raise ValueError("At least one inflammatory centroid is required")
    if not 0.0 <= separation_fraction <= 1.0:
        raise ValueError("separation_fraction must be in [0, 1]")
    if len(candidates) < count:
        raise RuntimeError(
            f"Only {len(candidates)} valid positions are available for {count} cells"
        )

    # Random values resolve equal-distance ties. Distance rank remains the dose,
    # and every level receives the same randomized lattice and tie ordering.
    tie_breakers = np.asarray([rng.random() for _ in range(len(candidates))])
    order = np.lexsort((tie_breakers, distances))
    ordered = candidates[order]
    available = len(ordered)
    window_count = max(count, round(DISTANCE_QUANTILE_WINDOW * available))
    maximum_start = available - window_count
    start = round(separation_fraction * maximum_start)
    window = ordered[start : start + window_count]

    # Spread selections over the complete band instead of taking a dense block.
    indices = np.floor(
        (np.arange(count, dtype=np.float64) + 0.5) * len(window) / count
    ).astype(int)
    return np.ascontiguousarray(window[indices], dtype=np.int16)


@lru_cache(maxsize=8192)
def _relocated_centroids_cached(
    geojson_path: str,
    separation_fraction: float,
    rng_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = Path(geojson_path)
    by_class = cell_centroids_by_class_from_geojson(path)
    tumor = by_class.get("Neoplastic", np.empty((0, 2), dtype=np.int16))
    original_inflammatory = by_class.get(
        "Inflammatory", np.empty((0, 2), dtype=np.int16)
    )
    if len(tumor) == 0 or len(original_inflammatory) == 0:
        raise ValueError(
            f"Both Neoplastic and Inflammatory centroids are required: {path.name}"
        )

    rng = random.Random(rng_seed)
    candidates = _candidate_lattice(rng)
    distance_map = nearest_centroid_distance_map(tumor).numpy()
    candidate_distances = distance_map[candidates[:, 1], candidates[:, 0]]
    valid = candidate_distances >= MINIMUM_TUMOR_DISTANCE_PX
    candidates = candidates[valid]
    candidate_distances = candidate_distances[valid]
    relocated = _select_distance_quantile_band(
        candidates,
        candidate_distances,
        count=len(original_inflammatory),
        separation_fraction=separation_fraction,
        rng=rng,
    )
    relocated_distances = distance_map[relocated[:, 1], relocated[:, 0]]
    original_distances = distance_map[
        original_inflammatory[:, 1], original_inflammatory[:, 0]
    ]
    distance_summary = np.asarray(
        [
            original_distances.mean(),
            np.median(original_distances),
            relocated_distances.mean(),
            np.median(relocated_distances),
        ],
        dtype=np.float64,
    )
    return tumor, original_inflammatory, relocated, distance_summary


def _relocated_centroids(
    context: InterventionContext, separation_fraction: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _relocated_centroids_cached(
        str(_geojson_path(context).resolve()),
        float(separation_fraction),
        _rng_seed(context),
    )


class TumorImmuneCentroidSeparation(ConditionIntervention):
    """Move all immune centroids while retaining exact tumor and immune counts."""

    def __init__(self, label: str, separation_fraction: float) -> None:
        self.label = str(label)
        self.separation_fraction = float(separation_fraction)
        if not 0.0 <= self.separation_fraction <= 1.0:
            raise ValueError("separation_fraction must be in [0, 1]")
        self.name = f"tumor_immune_{self.label}"

    def parameters(self) -> dict[str, Any]:
        return {
            "pattern": self.label,
            "separation_fraction": self.separation_fraction,
            "tumor_nucleus_diameter_px": TUMOR_NUCLEUS_DIAMETER_PX,
            "tumor_nucleus_radius_px": TUMOR_NUCLEUS_RADIUS_PX,
            "distance_definition": (
                "inflammatory-centroid to nearest neoplastic-centroid"
            ),
            "selection": "moving 20% nearest-distance quantile band",
            "minimum_tumor_centroid_distance_px": MINIMUM_TUMOR_DISTANCE_PX,
            "candidate_grid_spacing_px": GRID_SPACING_PX,
            "candidate_grid_jitter_px": GRID_JITTER_PX,
            "preserved_quantities": [
                "neoplastic centroid count and coordinates",
                "inflammatory centroid count",
                "other cell-type spatial channels",
                "morphology",
                "generation seed",
            ],
        }

    def modify_spatial(self, spatial: Tensor, context: InterventionContext) -> Tensor:
        _, _, relocated, _ = _relocated_centroids(
            context, self.separation_fraction
        )
        spatial[INFLAMMATORY_CHANNEL] = render_centroid_channel(relocated).to(
            device=spatial.device
        )
        return spatial

    def details(self, context: InterventionContext) -> dict[str, Any]:
        tumor, original, relocated, distances = _relocated_centroids(
            context, self.separation_fraction
        )
        return {
            "pattern": self.label,
            "separation_fraction": self.separation_fraction,
            "neoplastic_centroid_count_before": len(tumor),
            "neoplastic_centroid_count_after": len(tumor),
            "inflammatory_centroid_count_before": len(original),
            "inflammatory_centroid_count_after": len(relocated),
            "mean_nearest_tumor_distance_before_px": float(distances[0]),
            "median_nearest_tumor_distance_before_px": float(distances[1]),
            "mean_nearest_tumor_distance_after_px": float(distances[2]),
            "median_nearest_tumor_distance_after_px": float(distances[3]),
            "original_inflammatory_centroids_sha256": centroid_sha256(original),
            "relocated_inflammatory_centroids_sha256": centroid_sha256(relocated),
            "tumor_centroids_sha256": centroid_sha256(tumor),
            "changed_spatial_channel": "inflammatory",
            "preserved_spatial_channels": [
                "neoplastic",
                "connective",
                "dead",
                "epithelial",
            ],
            "note": (
                "Exact centroid count is preserved; rendered channel mass is not a "
                "count because training-time peak normalization is retained."
            ),
        }


def build_interventions() -> list[ConditionIntervention]:
    return [
        TumorImmuneCentroidSeparation(label, fraction)
        for label, fraction in SEPARATION_LEVELS
    ]
