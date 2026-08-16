from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cpathogen.counterfactuals import (
    ConditionStore,
    InterventionContext,
)
from cpathogen.counterfactuals.conditions import MORPHOLOGY_FEATURE_NAMES
from experiments.spatial.nuclear_enlargement import build_interventions


def make_store(tmp_path: Path) -> ConditionStore:
    maps = tmp_path / "spatial_maps"
    maps.mkdir()
    np.savez_compressed(maps / "tile.npz", map=np.zeros((8, 8, 5), dtype=np.float32))
    pd.DataFrame(
        [np.arange(16, dtype=np.float32)],
        index=["tile"],
        columns=MORPHOLOGY_FEATURE_NAMES,
    ).to_parquet(tmp_path / "morphology_stats.parquet")
    return ConditionStore(tmp_path)


def test_nuclear_enlargement_changes_only_mean_size_features(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    original = store.load("tile")
    area = MORPHOLOGY_FEATURE_NAMES.index("area_mean")
    perimeter = MORPHOLOGY_FEATURE_NAMES.index("perimeter_mean")
    expected_levels = [-2.0, -1.0, 0.5, 1.0, 1.5, 2.0]
    for intervention, level in zip(build_interventions(), expected_levels, strict=True):
        applied = intervention.apply(
            original,
            InterventionContext(store, "tile", intervention_seed=42, generation_seed=7),
        )
        delta = applied.condition.morphology - original.morphology
        assert torch.equal(applied.condition.spatial, original.spatial)
        assert float(delta[area]) == level
        assert float(delta[perimeter]) == level
        changed = torch.nonzero(delta, as_tuple=False).flatten().tolist()
        assert changed == [area, perimeter]
        assert applied.details["variance_features_changed"] is False


def test_nuclear_enlargement_levels_are_nested() -> None:
    assert [item.parameters()["sd_steps"] for item in build_interventions()] == [
        -2.0,
        -1.0,
        0.5,
        1.0,
        1.5,
        2.0,
    ]
