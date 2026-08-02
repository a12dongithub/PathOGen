"""Legacy experiment 08: rotate the spatial control while holding morphology fixed."""

from __future__ import annotations

import torch
from torch import Tensor

from cpathogen.counterfactuals import ConditionIntervention, InterventionContext


class RotateSpatialMap(ConditionIntervention):
    def __init__(self, quarter_turns: int) -> None:
        self.quarter_turns = quarter_turns
        self.name = f"rotate_spatial_{quarter_turns * 90}deg"

    def parameters(self) -> dict[str, int]:
        return {"quarter_turns_counterclockwise": self.quarter_turns}

    def modify_spatial(self, spatial: Tensor, context: InterventionContext) -> Tensor:
        return torch.rot90(spatial, k=self.quarter_turns, dims=(-2, -1)).contiguous()


def build_interventions() -> list[ConditionIntervention]:
    return [RotateSpatialMap(turns) for turns in (1, 2, 3)]
