"""Increase H&E tile brightness through the standardized RGB mean controls."""

from __future__ import annotations

from typing import Any

from torch import Tensor

from cpathogen.counterfactuals import ConditionIntervention, InterventionContext
from cpathogen.counterfactuals.conditions import MORPHOLOGY_FEATURE_NAMES

RGB_MEAN_NAMES = ("r_mean", "g_mean", "b_mean")
RGB_MEAN_INDICES = tuple(MORPHOLOGY_FEATURE_NAMES.index(name) for name in RGB_MEAN_NAMES)


class StainBrightnessIncrease(ConditionIntervention):
    """Shift all RGB means equally in z-score space, preserving color balance."""

    def __init__(self, sd_steps: float) -> None:
        self.sd_steps = float(sd_steps)
        if self.sd_steps <= 0.0:
            raise ValueError("sd_steps must be positive")
        label = str(self.sd_steps).replace(".", "p")
        self.name = f"stain_brightness_plus_{label}sd"

    def parameters(self) -> dict[str, Any]:
        return {
            "sd_steps": self.sd_steps,
            "feature_space": "training-standardized z scores",
            "increased_features": list(RGB_MEAN_NAMES),
            "preserved_features": ["r_var", "g_var", "b_var"],
            "direction": "brighter / higher RGB intensity",
        }

    def modify_morphology(
        self, morphology: Tensor, context: InterventionContext
    ) -> Tensor:
        del context
        for index in RGB_MEAN_INDICES:
            morphology[index] += self.sd_steps
        return morphology

    def details(self, context: InterventionContext) -> dict[str, Any]:
        original = context.store.load_morphology(context.original_stem)
        return {
            "requested_sd_steps_per_channel_mean": self.sd_steps,
            "rgb_means_before": {
                name: float(original[index])
                for name, index in zip(RGB_MEAN_NAMES, RGB_MEAN_INDICES, strict=True)
            },
            "rgb_means_after": {
                name: float(original[index] + self.sd_steps)
                for name, index in zip(RGB_MEAN_NAMES, RGB_MEAN_INDICES, strict=True)
            },
            "rgb_variances_changed": False,
        }


def build_interventions() -> list[ConditionIntervention]:
    return [
        StainBrightnessIncrease(0.5),
        StainBrightnessIncrease(1.0),
        StainBrightnessIncrease(1.5),
    ]
