"""Legacy experiments 11/17/19: replace morphology with random donor vectors."""

from __future__ import annotations

from torch import Tensor

from cpathogen.counterfactuals import ConditionIntervention, InterventionContext


class DonorMorphologyVector(ConditionIntervention):
    def __init__(self, draw_index: int) -> None:
        self.draw_index = draw_index
        self.name = f"donor_morphology_{draw_index + 1:02d}"

    @property
    def namespace(self) -> str:
        return f"donor_morphology:{self.draw_index}"

    def parameters(self) -> dict[str, int]:
        return {"draw_index": self.draw_index}

    def modify_morphology(self, morphology: Tensor, context: InterventionContext) -> Tensor:
        return context.store.load_morphology(context.donor_stem(self.namespace))

    def details(self, context: InterventionContext) -> dict[str, str]:
        return {"morphology_donor_stem": context.donor_stem(self.namespace)}


def build_interventions() -> list[ConditionIntervention]:
    return [DonorMorphologyVector(index) for index in range(20)]
