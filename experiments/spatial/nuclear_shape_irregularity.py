"""Increase nuclear shape irregularity while preserving nuclear area."""

from __future__ import annotations

from typing import Any

from torch import Tensor

from cpathogen.counterfactuals import ConditionIntervention, InterventionContext
from cpathogen.counterfactuals.conditions import MORPHOLOGY_FEATURE_NAMES

ECCENTRICITY_MEAN_INDEX = MORPHOLOGY_FEATURE_NAMES.index("eccentricity_mean")
SOLIDITY_MEAN_INDEX = MORPHOLOGY_FEATURE_NAMES.index("solidity_mean")
PERIMETER_MEAN_INDEX = MORPHOLOGY_FEATURE_NAMES.index("perimeter_mean")


class NuclearShapeIrregularity(ConditionIntervention):
    """Move along an interpretable irregular-shape direction in z-score space."""

    def __init__(self, sd_steps: float) -> None:
        self.sd_steps = float(sd_steps)
        if self.sd_steps == 0.0:
            raise ValueError("sd_steps must be non-zero; baseline represents zero")
        label = str(abs(self.sd_steps)).replace(".", "p")
        direction = "plus" if self.sd_steps > 0 else "minus"
        self.name = f"nuclear_shape_irregularity_{direction}_{label}sd"

    def parameters(self) -> dict[str, Any]:
        return {
            "sd_steps": self.sd_steps,
            "feature_space": "training-standardized z scores",
            "increased_features": ["eccentricity_mean", "perimeter_mean"],
            "decreased_features": ["solidity_mean"],
            "preserved_features": ["area_mean", "area_var"],
        }

    def modify_morphology(
        self, morphology: Tensor, context: InterventionContext
    ) -> Tensor:
        del context
        morphology[ECCENTRICITY_MEAN_INDEX] += self.sd_steps
        morphology[PERIMETER_MEAN_INDEX] += self.sd_steps
        morphology[SOLIDITY_MEAN_INDEX] -= self.sd_steps
        return morphology

    def details(self, context: InterventionContext) -> dict[str, Any]:
        original = context.store.load_morphology(context.original_stem)
        return {
            "requested_sd_steps_per_feature": self.sd_steps,
            "eccentricity_mean_before": float(original[ECCENTRICITY_MEAN_INDEX]),
            "eccentricity_mean_after": float(
                original[ECCENTRICITY_MEAN_INDEX] + self.sd_steps
            ),
            "perimeter_mean_before": float(original[PERIMETER_MEAN_INDEX]),
            "perimeter_mean_after": float(
                original[PERIMETER_MEAN_INDEX] + self.sd_steps
            ),
            "solidity_mean_before": float(original[SOLIDITY_MEAN_INDEX]),
            "solidity_mean_after": float(original[SOLIDITY_MEAN_INDEX] - self.sd_steps),
            "area_features_changed": False,
        }


def build_interventions() -> list[ConditionIntervention]:
    return [
        NuclearShapeIrregularity(-2.0),
        NuclearShapeIrregularity(-1.0),
        NuclearShapeIrregularity(0.5),
        NuclearShapeIrregularity(1.0),
        NuclearShapeIrregularity(1.5),
        NuclearShapeIrregularity(2.0),
    ]
