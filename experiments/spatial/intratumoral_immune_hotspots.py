"""Add discrete inflammatory hotspots inside tumor-rich regions."""

from __future__ import annotations

from typing import Any

from torch import Tensor

from cpathogen.counterfactuals import ConditionIntervention, InterventionContext
from experiments.spatial._tumor_immune_pilot_utils import (
    INFLAMMATORY_CHANNEL,
    TUMOR_CHANNEL,
    add_centroid_signal,
    intratumoral_weights,
    sample_nested_centroids,
)

RNG_NAMESPACE = "intratumoral-immune-hotspots-v1"


class IntratumoralImmuneHotspots(ConditionIntervention):
    def __init__(self, centroid_count: int) -> None:
        self.centroid_count = int(centroid_count)
        self.name = f"intratumoral_hotspots_{centroid_count}_centroids"

    def parameters(self) -> dict[str, Any]:
        return {"added_centroids": self.centroid_count, "target": "tumor-rich pixels", "signal_strength": 0.75}

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


def build_interventions() -> list[ConditionIntervention]:
    return [IntratumoralImmuneHotspots(count) for count in (80, 160, 320)]
