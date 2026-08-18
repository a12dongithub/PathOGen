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
    cell_centroids_by_class_from_geojson,
    render_centroid_channel,
)
from cpathogen.counterfactuals.centroids import centroid_sha256
from experiments.spatial._tumor_immune_pilot_utils import (
    INFLAMMATORY_CHANNEL,
    PERITUMORAL_RING_WIDTH_PX,
    TUMOR_NUCLEUS_DIAMETER_PX,
    TUMOR_NUCLEUS_RADIUS_PX,
    minimum_centroid_distance,
    nearest_centroid_distance_map,
    sample_nested_centroids,
    tumor_centroid_annulus_weights,
)

RNG_NAMESPACE = "peritumoral-immune-ring-v3-tumor-diameter-40px"
RING_INNER_RADIUS_PX = TUMOR_NUCLEUS_RADIUS_PX
RING_WIDTH_PX = PERITUMORAL_RING_WIDTH_PX
RING_OUTER_RADIUS_PX = RING_INNER_RADIUS_PX + RING_WIDTH_PX


def _geojson_path(context: InterventionContext) -> Path:
    path = context.store.data_root / "geojsons" / f"{context.original_stem}.geojson"
    if not path.is_file():
        raise FileNotFoundError(
            "Peritumoral ring generation requires source nuclei at "
            f"data_root/geojsons/<stem>.geojson; missing: {path}"
        )
    return path


@lru_cache(maxsize=4096)
def _all_added_centroids_cached(
    geojson_path_string: str,
    rng_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    by_class = cell_centroids_by_class_from_geojson(Path(geojson_path_string))
    tumor = by_class.get("Neoplastic", np.empty((0, 2), dtype=np.int16))
    original = by_class.get(
        "Inflammatory", np.empty((0, 2), dtype=np.int16)
    )
    if len(tumor) == 0 or len(original) == 0:
        raise ValueError(
            "At least one Neoplastic and one Inflammatory centroid are required"
        )
    weights = tumor_centroid_annulus_weights(
        tumor,
        inner_radius_px=RING_INNER_RADIUS_PX,
        width_px=RING_WIDTH_PX,
    )
    added = sample_nested_centroids(
        weights,
        320,
        rng=random.Random(rng_seed),
        forbidden_centroids=original,
    )
    return tumor, original, added, weights.numpy()


def _all_added_centroids(
    context: InterventionContext,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng_seed = context.rng(RNG_NAMESPACE).getrandbits(64)
    return _all_added_centroids_cached(
        str(_geojson_path(context).resolve()), rng_seed
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
            "tumor_nucleus_diameter_px": TUMOR_NUCLEUS_DIAMETER_PX,
            "ring_inner_centroid_distance_px": RING_INNER_RADIUS_PX,
            "ring_outer_centroid_distance_px": RING_OUTER_RADIUS_PX,
            "ring_width_px": RING_WIDTH_PX,
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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        del spatial
        tumor, original, all_added, weights = _all_added_centroids(context)
        return tumor, original, all_added[: self.centroid_count], weights

    def modify_spatial(self, spatial: Tensor, context: InterventionContext) -> Tensor:
        _, original, added, _ = self._added_centroids(spatial, context)
        combined = np.concatenate([original, added], axis=0)
        spatial[INFLAMMATORY_CHANNEL] = render_centroid_channel(combined).to(
            device=spatial.device
        )
        return spatial

    def details(self, context: InterventionContext) -> dict[str, Any]:
        original_spatial = context.store.load_spatial(context.original_stem)
        tumor, original, added, weights = self._added_centroids(
            original_spatial, context
        )
        combined = np.concatenate([original, added], axis=0)
        in_ring = [bool(weights[int(y), int(x)] > 0) for x, y in added]
        distance_map = nearest_centroid_distance_map(tumor).numpy()
        tumor_distances = distance_map[added[:, 1], added[:, 0]]
        return {
            "original_inflammatory_centroid_count": len(original),
            "added_inflammatory_centroid_count": len(added),
            "resulting_inflammatory_centroid_count": len(combined),
            "added_centroid_fraction_in_declared_ring": float(np.mean(in_ring)),
            "minimum_added_to_tumor_centroid_distance_px": float(
                tumor_distances.min()
            ),
            "median_added_to_tumor_centroid_distance_px": float(
                np.median(tumor_distances)
            ),
            "maximum_added_to_tumor_centroid_distance_px": float(
                tumor_distances.max()
            ),
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
