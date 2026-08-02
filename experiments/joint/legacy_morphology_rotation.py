"""Legacy experiment 08's complete donor-swap and rotation variant set."""

from __future__ import annotations

from cpathogen.counterfactuals import ConditionIntervention
from experiments.morphology.donor_vectors import DonorMorphologyVector
from experiments.spatial.donor_maps import DonorSpatialMap
from experiments.spatial.rotate_maps import RotateSpatialMap


def build_interventions() -> list[ConditionIntervention]:
    return [
        DonorMorphologyVector(0),
        DonorSpatialMap(0),
        RotateSpatialMap(1),
        RotateSpatialMap(2),
        RotateSpatialMap(3),
    ]
