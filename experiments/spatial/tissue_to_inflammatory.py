"""Legacy experiments 20/21: progressively relabel cells as inflammatory."""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import label
from torch import Tensor

from cpathogen.counterfactuals import CELL_TYPE_NAMES, ConditionIntervention, InterventionContext


class TissueToInflammatory(ConditionIntervention):
    def __init__(self, fraction: float) -> None:
        self.fraction = fraction
        self.name = f"tissue_to_inflammatory_{int(round(fraction * 100)):03d}pct"

    def parameters(self) -> dict[str, object]:
        return {
            "fraction": self.fraction,
            "source_channels": [0, 4],
            "source_cell_types": [CELL_TYPE_NAMES[0], CELL_TYPE_NAMES[4]],
            "target_channel": 1,
            "target_cell_type": CELL_TYPE_NAMES[1],
            "component_threshold": 0.0,
        }

    def modify_spatial(self, spatial: Tensor, context: InterventionContext) -> Tensor:
        output = spatial.detach().cpu().numpy().copy()
        components: list[tuple[int, np.ndarray]] = []
        for source_channel in (0, 4):
            labeled, count = label(output[source_channel] > 0.0)
            components.extend(
                (source_channel, labeled == component_id)
                for component_id in range(1, count + 1)
            )
        count_to_convert = int(len(components) * self.fraction)
        if count_to_convert:
            selected = context.rng(self.name).sample(components, count_to_convert)
            for source_channel, mask in selected:
                output[1, mask] = output[source_channel, mask]
                output[source_channel, mask] = 0.0
        return torch.from_numpy(output).to(dtype=spatial.dtype)


def build_interventions() -> list[ConditionIntervention]:
    return [TissueToInflammatory(fraction) for fraction in (0.0, 0.25, 0.5, 0.75, 1.0)]
