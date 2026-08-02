"""Legacy experiment 04: replace the entire map with one cell-type plane."""

from __future__ import annotations

import torch
from torch import Tensor

from cpathogen.counterfactuals import CELL_TYPE_NAMES, ConditionIntervention, InterventionContext


class FullPlaneCategory(ConditionIntervention):
    def __init__(self, target_channel: int) -> None:
        self.target_channel = target_channel
        self.name = f"full_plane_{CELL_TYPE_NAMES[target_channel]}"

    def parameters(self) -> dict[str, object]:
        return {
            "target_channel": self.target_channel,
            "target_cell_type": CELL_TYPE_NAMES[self.target_channel],
            "preserve_cell_envelope": False,
        }

    def modify_spatial(self, spatial: Tensor, context: InterventionContext) -> Tensor:
        output = torch.zeros_like(spatial)
        output[self.target_channel].fill_(1.0)
        return output


def build_interventions() -> list[ConditionIntervention]:
    return [FullPlaneCategory(channel) for channel in (0, 1, 2)]
