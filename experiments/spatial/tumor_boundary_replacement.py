"""Progressively replace tumor-boundary signal with inflammatory signal."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from cpathogen.counterfactuals import ConditionIntervention, InterventionContext
from experiments.spatial._tumor_immune_pilot_utils import (
    INFLAMMATORY_CHANNEL,
    TUMOR_CHANNEL,
    boundary_transfer_weights,
)


class TumorBoundaryReplacement(ConditionIntervention):
    def __init__(self, fraction: float) -> None:
        self.fraction = float(fraction)
        self.name = f"tumor_boundary_replacement_{str(fraction).replace('.', 'p')}"

    def parameters(self) -> dict[str, Any]:
        return {"replacement_fraction": self.fraction, "boundary_radius_px": 18}

    def modify_spatial(self, spatial: Tensor, context: InterventionContext) -> Tensor:
        del context
        tumor = spatial[TUMOR_CHANNEL].clone()
        transfer = self.fraction * boundary_transfer_weights(tumor) * tumor
        spatial[TUMOR_CHANNEL] = (tumor - transfer).clamp_min(0.0)
        spatial[INFLAMMATORY_CHANNEL] = torch.clamp(
            spatial[INFLAMMATORY_CHANNEL] + transfer, 0.0, 1.0
        )
        return spatial


def build_interventions() -> list[ConditionIntervention]:
    return [TumorBoundaryReplacement(fraction) for fraction in (0.25, 0.50, 0.75)]
