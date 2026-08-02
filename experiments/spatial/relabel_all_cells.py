"""Legacy experiments 05/07: relabel the existing cell envelope."""

from __future__ import annotations

import torch
from torch import Tensor

from cpathogen.counterfactuals import CELL_TYPE_NAMES, ConditionIntervention, InterventionContext


class RelabelAllCells(ConditionIntervention):
    def __init__(self, target_channel: int) -> None:
        self.target_channel = target_channel
        self.name = f"all_cells_{CELL_TYPE_NAMES[target_channel]}"

    def parameters(self) -> dict[str, object]:
        return {
            "target_channel": self.target_channel,
            "target_cell_type": CELL_TYPE_NAMES[self.target_channel],
            "preserve_cell_envelope": True,
        }

    def modify_spatial(self, spatial: Tensor, context: InterventionContext) -> Tensor:
        cell_envelope = torch.amax(spatial, dim=0)
        output = torch.zeros_like(spatial)
        output[self.target_channel] = cell_envelope
        return output


def build_interventions() -> list[ConditionIntervention]:
    return [RelabelAllCells(channel) for channel in (0, 1, 2)]
