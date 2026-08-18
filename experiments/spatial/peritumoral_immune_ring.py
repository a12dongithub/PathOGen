"""Add exact-count inflammatory centroids in an outer tumor ring."""

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
    cell_centroids_from_geojson,
    render_centroid_channel,
)
from cpathogen.counterfactuals.centroids import centroid_sha256
from experiments.spatial._tumor_immune_pilot_utils import (
    INFLAMMATORY_CHANNEL,
    TUMOR_CHANNEL,
    minimum_centroid_distance,
    outer_tumor_ring_weights,
    sample_nested_centroids,
)

RNG_NAMESPACE = "peritumoral-immune-ring-v2"
RING_RADIUS_PX = 24
TUMOR_CORE_THRESHOLD = 0.25


def _geojson_path(context: InterventionContext) -> Path:
    path = context.store.data_root / "geojsons" / f"{context.original_stem}.geojson"
    if not path.is_file():
        raise FileNotFoundError(
            "Peritumoral ring generation requires source nuclei at "
            f"data_root/geojsons/<stem>.geojson; missing: {path}"
        )
    return path


def _original_inflammatory(context: InterventionContext) -> np.ndarray:
    centroids = cell_centroids_from_geojson(_geojson_path(context), "Inflammatory")
    if len(centroids) == 0:
        raise ValueError(
            f"{context.original_stem} has no original inflammatory centroids"
        )
    return centroids


@lru_cache(maxsize=4096)
def _all_added_centroids_cached(
    map_path_string: str,
    geojson_path_string: str,
    rng_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    original = cell_centroids_from_geojson(
        Path(geojson_path_string), "Inflammatory"
    )
    if len(original) == 0:
        raise ValueError("At least one original inflammatory centroid is required")
    with np.load(map_path_string, allow_pickle=False) as archive:
        spatial_map = np.asarray(archive["map"], dtype=np.float32)
    if spatial_map.max(initial=0.0) > 1.0:
        spatial_map /= 255.0
    tumor = Tensor(spatial_map[:, :, TUMOR_CHANNEL])
    weights = outer_tumor_ring_weights(
        tumor,
        radius_px=RING_RADIUS_PX,
        core_threshold=TUMOR_CORE_THRESHOLD,
    )
    added = sample_nested_centroids(
        weights,
        320,
        rng=random.Random(rng_seed),
        forbidden_centroids=original,
    )
    return original, added, weights.numpy()


def _all_added_centroids(
    context: InterventionContext,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    map_path = context.store.spatial_maps_dir / f"{context.original_stem}.npz"
    rng_seed = context.rng(RNG_NAMESPACE).getrandbits(64)
    return _all_added_centroids_cached(
        str(map_path.resolve()), str(_geojson_path(context).resolve()), rng_seed
    )


class PeritumoralImmuneRing(ConditionIntervention):
    """Add nested inflammatory nuclei immediately outside tumor support."""

    def __init__(self, centroid_count: int) -> None:
        self.centroid_count = int(centroid_count)
        if self.centroid_count < 1:
            raise ValueError("centroid_count must be positive")
        self.name = f"peritumoral_ring_plus_{centroid_count}"

    def parameters(self) -> dict[str, Any]:
        return {
            "added_inflammatory_centroids": self.centroid_count,
            "ring_radius_px": RING_RADIUS_PX,
            "tumor_core_threshold": TUMOR_CORE_THRESHOLD,
            "preferred_minimum_centroid_distance_px": 5.0,
            "spatial_render_sigma_px": 3.0,
            "spatial_render_normalization": "joint peak normalization after addition",
            "preserved_controls": [
                "neoplastic centroid count and spatial channel",
                "connective spatial channel",
                "dead spatial channel",
                "epithelial spatial channel",
                "morphology",
                "generation seed",
            ],
        }

    def _added_centroids(
        self, spatial: Tensor, context: InterventionContext
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        del spatial
        original, all_added, weights = _all_added_centroids(context)
        return original, all_added[: self.centroid_count], weights

    def modify_spatial(self, spatial: Tensor, context: InterventionContext) -> Tensor:
        original, added, _ = self._added_centroids(spatial, context)
        combined = np.concatenate([original, added], axis=0)
        spatial[INFLAMMATORY_CHANNEL] = render_centroid_channel(combined).to(
            device=spatial.device
        )
        return spatial

    def details(self, context: InterventionContext) -> dict[str, Any]:
        original_spatial = context.store.load_spatial(context.original_stem)
        original, added, weights = self._added_centroids(original_spatial, context)
        combined = np.concatenate([original, added], axis=0)
        in_ring = [bool(weights[int(y), int(x)] > 0) for x, y in added]
        return {
            "original_inflammatory_centroid_count": len(original),
            "added_inflammatory_centroid_count": len(added),
            "resulting_inflammatory_centroid_count": len(combined),
            "added_centroid_fraction_in_declared_ring": float(np.mean(in_ring)),
            "achieved_minimum_added_centroid_distance_px": minimum_centroid_distance(
                added
            ),
            "added_centroids_sha256": centroid_sha256(added),
            "resulting_centroids_sha256": centroid_sha256(combined),
            "changed_spatial_channel": "inflammatory",
            "preserved_spatial_channels": [
                "neoplastic",
                "connective",
                "dead",
                "epithelial",
            ],
        }


def build_interventions() -> list[ConditionIntervention]:
    return [PeritumoralImmuneRing(count) for count in (80, 160, 320)]
