"""Increase mean nuclear size in standardized morphology-feature units."""

from __future__ import annotations

from typing import Any

from torch import Tensor

from cpathogen.counterfactuals import ConditionIntervention, InterventionContext
from cpathogen.counterfactuals.conditions import MORPHOLOGY_FEATURE_NAMES

AREA_MEAN_INDEX = MORPHOLOGY_FEATURE_NAMES.index("area_mean")
PERIMETER_MEAN_INDEX = MORPHOLOGY_FEATURE_NAMES.index("perimeter_mean")


class NuclearEnlargement(ConditionIntervention):
    """Shift mean area and perimeter together while preserving heterogeneity."""

    def __init__(self, sd_steps: float) -> None:
        self.sd_steps = float(sd_steps)
        if self.sd_steps == 0.0:
            raise ValueError("sd_steps must be non-zero; baseline represents zero")
        label = str(abs(self.sd_steps)).replace(".", "p")
        direction = "plus" if self.sd_steps > 0 else "minus"
        self.name = f"nuclear_enlargement_{direction}_{label}sd"

    def parameters(self) -> dict[str, Any]:
        return {
            "sd_steps": self.sd_steps,
            "features": ["area_mean", "perimeter_mean"],
            "feature_space": "training-standardized z scores",
            "preserved_features": ["area_var", "perimeter_var"],
        }

    def modify_morphology(
        self, morphology: Tensor, context: InterventionContext
    ) -> Tensor:
        del context
        morphology[AREA_MEAN_INDEX] += self.sd_steps
        morphology[PERIMETER_MEAN_INDEX] += self.sd_steps
        return morphology

    def details(self, context: InterventionContext) -> dict[str, Any]:
        original = context.store.load_morphology(context.original_stem)
        return {
            "requested_sd_steps_per_feature": self.sd_steps,
            "area_mean_before": float(original[AREA_MEAN_INDEX]),
            "area_mean_after": float(original[AREA_MEAN_INDEX] + self.sd_steps),
            "perimeter_mean_before": float(original[PERIMETER_MEAN_INDEX]),
            "perimeter_mean_after": float(
                original[PERIMETER_MEAN_INDEX] + self.sd_steps
            ),
            "variance_features_changed": False,
        }


def build_interventions() -> list[ConditionIntervention]:
    return [
        NuclearEnlargement(-2.0),
        NuclearEnlargement(-1.0),
        NuclearEnlargement(0.5),
        NuclearEnlargement(1.0),
        NuclearEnlargement(1.5),
        NuclearEnlargement(2.0),
    ]
