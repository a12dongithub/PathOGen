"""Legacy experiment 16: a 4-by-5 donor morphology/spatial control grid."""

from __future__ import annotations

from torch import Tensor

from cpathogen.counterfactuals import ConditionIntervention, InterventionContext


class DonorControlPair(ConditionIntervention):
    def __init__(self, morphology_index: int, spatial_index: int) -> None:
        self.morphology_index = morphology_index
        self.spatial_index = spatial_index
        self.name = f"donor_pair_m{morphology_index + 1:02d}_s{spatial_index + 1:02d}"

    @property
    def morphology_namespace(self) -> str:
        return f"joint_grid_morphology:{self.morphology_index}"

    @property
    def spatial_namespace(self) -> str:
        return f"joint_grid_spatial:{self.spatial_index}"

    def parameters(self) -> dict[str, int]:
        return {
            "morphology_draw_index": self.morphology_index,
            "spatial_draw_index": self.spatial_index,
        }

    def modify_spatial(self, spatial: Tensor, context: InterventionContext) -> Tensor:
        return context.store.load_spatial(
            context.donor_stem(self.spatial_namespace)
        )

    def modify_morphology(self, morphology: Tensor, context: InterventionContext) -> Tensor:
        return context.store.load_morphology(
            context.donor_stem(self.morphology_namespace)
        )

    def details(self, context: InterventionContext) -> dict[str, str]:
        return {
            "morphology_donor_stem": context.donor_stem(self.morphology_namespace),
            "spatial_donor_stem": context.donor_stem(self.spatial_namespace),
        }


def build_interventions() -> list[ConditionIntervention]:
    return [
        DonorControlPair(morphology_index, spatial_index)
        for morphology_index in range(4)
        for spatial_index in range(5)
    ]
