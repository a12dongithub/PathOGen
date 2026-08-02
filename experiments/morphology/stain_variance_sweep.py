"""Legacy experiment 27: set RGB-variance controls across a z-score grid."""

from __future__ import annotations

from cpathogen.counterfactuals import ConditionIntervention
from experiments.morphology.full_feature_sweep import SetMorphologyFeature, SWEEP_VALUES


def build_interventions() -> list[ConditionIntervention]:
    return [
        SetMorphologyFeature(index, value)
        for index in (11, 13, 15)
        for value in SWEEP_VALUES
    ]
