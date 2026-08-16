"""Add discrete inflammatory hotspots inside tumor-rich regions."""

from __future__ import annotations

from typing import Any

from torch import Tensor

from cpathogen.counterfactuals import ConditionIntervention, InterventionContext
from cpathogen.counterfactuals.centroids import render_centroid_channel
from experiments.spatial._tumor_immune_pilot_utils import (
    INFLAMMATORY_CHANNEL,
    TUMOR_CHANNEL,
    add_centroid_signal,
    intratumoral_weights,
    minimum_centroid_distance,
    sample_nested_centroids,
)

RNG_NAMESPACE = "intratumoral-immune-hotspots-v1"


class IntratumoralImmuneHotspots(ConditionIntervention):
    def __init__(self, centroid_count: int) -> None:
        self.centroid_count = int(centroid_count)
        self.name = f"intratumoral_hotspots_{centroid_count}_centroids"

    def parameters(self) -> dict[str, Any]:
        return {
            "added_centroids": self.centroid_count,
            "target": "tumor-rich pixels",
            "target_weighting": "squared peak-normalized neoplastic signal",
            "high_tumor_mask_threshold": "50% of neoplastic channel peak",
            "preferred_minimum_centroid_distance_px": 5.0,
            "dense_region_fallback": "sqrt(2) px when possible; unique pixels otherwise",
            "spatial_render_sigma_px": 3.0,
            "signal_strength": 0.75,
        }

    def modify_spatial(self, spatial: Tensor, context: InterventionContext) -> Tensor:
        centroids = sample_nested_centroids(
            intratumoral_weights(spatial[TUMOR_CHANNEL]),
            self.centroid_count,
            rng=context.rng(RNG_NAMESPACE),
        )
        spatial[INFLAMMATORY_CHANNEL] = add_centroid_signal(
            spatial[INFLAMMATORY_CHANNEL], centroids, 0.75
        )
        return spatial

    def details(self, context: InterventionContext) -> dict[str, Any]:
        original = context.store.load_spatial(context.original_stem)
        centroids = sample_nested_centroids(
            intratumoral_weights(original[TUMOR_CHANNEL]),
            self.centroid_count,
            rng=context.rng(RNG_NAMESPACE),
        )
        converted = add_centroid_signal(
            original[INFLAMMATORY_CHANNEL], centroids, 0.75
        )
        rendered = render_centroid_channel(centroids).to(original.device)
        unclipped = original[INFLAMMATORY_CHANNEL] + 0.75 * rendered
        tumor = original[TUMOR_CHANNEL]
        high_tumor = tumor >= 0.5 * tumor.max()
        inside_count = sum(
            bool(high_tumor[int(y), int(x)]) for x, y in centroids
        )
        return {
            "added_centroid_count": self.centroid_count,
            "inflammatory_mass_before": float(original[INFLAMMATORY_CHANNEL].sum()),
            "inflammatory_mass_after": float(converted.sum()),
            "clipped_pixel_fraction": float((unclipped > 1.0).float().mean()),
            "high_tumor_mask_threshold_fraction": 0.5,
            "added_centroid_fraction_in_high_tumor_mask": (
                inside_count / self.centroid_count
            ),
            "achieved_minimum_centroid_distance_px": minimum_centroid_distance(
                centroids
            ),
            "changed_spatial_channel": "inflammatory",
            "preserved_spatial_channels": [
                "neoplastic",
                "connective",
                "dead",
                "epithelial",
            ],
        }


def build_interventions() -> list[ConditionIntervention]:
    return [IntratumoralImmuneHotspots(count) for count in (80, 160, 320)]
