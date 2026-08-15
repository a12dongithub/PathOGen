"""Increase inflammatory count in square-root-count SD units."""

from __future__ import annotations

from typing import Any

from cpathogen.counterfactuals import (
    AppliedIntervention,
    ConditionBundle,
    ConditionIntervention,
    InterventionContext,
)
from cpathogen.counterfactuals.centroids import (
    add_jittered_centroids,
    centroid_sha256,
    load_centroid_reference_stats,
    load_inflammatory_centroids,
    render_centroid_channel,
    sd_target_count,
)

INFLAMMATORY_CHANNEL = 1
RNG_NAMESPACE = "inflammatory-centroid-density-v1"


class InflammatoryCentroidDensityIncrease(ConditionIntervention):
    """Add nuclei near existing inflammatory regions and rebuild the map channel."""

    def __init__(self, sd_steps: float) -> None:
        self.sd_steps = float(sd_steps)
        if self.sd_steps <= 0.0:
            raise ValueError("sd_steps must be positive")
        label = str(self.sd_steps).replace(".", "p")
        self.name = f"inflammatory_centroids_plus_{label}sd"

    def parameters(self) -> dict[str, Any]:
        return {
            "sd_steps": self.sd_steps,
            "count_transform": "sqrt",
            "channel": "inflammatory",
            "jitter_sigma_px": 12.0,
            "minimum_distance_px": 4.0,
            "spatial_render_sigma_px": 3.0,
        }

    def apply(
        self, original: ConditionBundle, context: InterventionContext
    ) -> AppliedIntervention:
        if original.stem != context.original_stem:
            raise ValueError("Intervention context does not match the condition stem")
        centroids = load_inflammatory_centroids(
            context.store.data_root, context.original_stem
        )
        reference = load_centroid_reference_stats(context.store.data_root)
        sqrt_count_sd = float(reference["sqrt_count_sd"])
        target_count = sd_target_count(len(centroids), self.sd_steps, sqrt_count_sd)
        addition_count = target_count - len(centroids)
        combined, added = add_jittered_centroids(
            centroids,
            addition_count,
            rng=context.rng(RNG_NAMESPACE),
        )
        spatial = original.spatial.detach().clone()
        spatial[INFLAMMATORY_CHANNEL] = render_centroid_channel(combined)
        converted = ConditionBundle(
            stem=original.stem,
            spatial=spatial,
            morphology=original.morphology.detach().clone(),
            metadata={
                **original.metadata,
                "intervention": self.slug,
                "intervention_parameters": self.parameters(),
            },
        )
        converted.validate()
        return AppliedIntervention(
            converted,
            {
                "channel": "inflammatory",
                "channel_index": INFLAMMATORY_CHANNEL,
                "requested_sd_steps": self.sd_steps,
                "count_transform": "sqrt",
                "reference_sqrt_count_sd": sqrt_count_sd,
                "reference_count": int(reference["reference_count"]),
                "original_centroid_count": len(centroids),
                "added_centroid_count": len(added),
                "resulting_centroid_count": len(combined),
                "achieved_sd_steps": (
                    (len(combined) ** 0.5 - len(centroids) ** 0.5) / sqrt_count_sd
                ),
                "added_centroids_sha256": centroid_sha256(added),
                "placement_rng_namespace": RNG_NAMESPACE,
            },
        )


def build_interventions() -> list[ConditionIntervention]:
    return [
        InflammatoryCentroidDensityIncrease(0.5),
        InflammatoryCentroidDensityIncrease(1.0),
        InflammatoryCentroidDensityIncrease(1.5),
    ]
