"""Legacy experiments 22/23: set every morphology feature across a z-score grid."""

from __future__ import annotations

from torch import Tensor

from cpathogen.counterfactuals import (
    MORPHOLOGY_FEATURE_NAMES,
    ConditionIntervention,
    InterventionContext,
)


class SetMorphologyFeature(ConditionIntervention):
    def __init__(self, feature_index: int, value: float) -> None:
        self.feature_index = feature_index
        self.value = value
        feature = MORPHOLOGY_FEATURE_NAMES[feature_index]
        value_slug = f"{value:+.1f}".replace("+", "p").replace("-", "m").replace(".", "p")
        self.name = f"set_{feature}_{value_slug}sd"

    def parameters(self) -> dict[str, object]:
        return {
            "feature_index": self.feature_index,
            "feature_name": MORPHOLOGY_FEATURE_NAMES[self.feature_index],
            "standardized_value": self.value,
            "operation": "set",
        }

    def modify_morphology(self, morphology: Tensor, context: InterventionContext) -> Tensor:
        morphology[self.feature_index] = self.value
        return morphology


SWEEP_VALUES = (-2.0, -1.0, 0.0, 1.0, 2.0)


def build_interventions() -> list[ConditionIntervention]:
    return [
        SetMorphologyFeature(index, value)
        for index in range(len(MORPHOLOGY_FEATURE_NAMES))
        for value in SWEEP_VALUES
    ]
