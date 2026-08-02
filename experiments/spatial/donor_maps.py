"""Legacy experiment 10: replace the spatial map with a deterministic random donor."""

from __future__ import annotations

from torch import Tensor

from cpathogen.counterfactuals import ConditionIntervention, InterventionContext


class DonorSpatialMap(ConditionIntervention):
    def __init__(self, draw_index: int) -> None:
        self.draw_index = draw_index
        self.name = f"donor_spatial_{draw_index + 1:02d}"

    @property
    def namespace(self) -> str:
        return f"donor_spatial:{self.draw_index}"

    def parameters(self) -> dict[str, int]:
        return {"draw_index": self.draw_index}

    def modify_spatial(self, spatial: Tensor, context: InterventionContext) -> Tensor:
        return context.store.load_spatial(context.donor_stem(self.namespace))

    def details(self, context: InterventionContext) -> dict[str, str]:
        return {"spatial_donor_stem": context.donor_stem(self.namespace)}


def build_interventions() -> list[ConditionIntervention]:
    return [DonorSpatialMap(index) for index in range(20)]
