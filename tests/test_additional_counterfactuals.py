from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cpathogen.counterfactuals import ConditionStore, InterventionContext
from cpathogen.counterfactuals.conditions import MORPHOLOGY_FEATURE_NAMES
from experiments.spatial.nuclear_shape_irregularity import (
    build_interventions as shape_interventions,
)
from experiments.spatial.stain_brightness import (
    build_interventions as brightness_interventions,
)
from experiments.spatial.tumor_immune_mixing import (
    build_interventions as mixing_interventions,
    tumor_weighted_overlap,
)


def make_store(tmp_path: Path) -> ConditionStore:
    maps = tmp_path / "spatial_maps"
    maps.mkdir()
    y, x = np.mgrid[0:32, 0:32]
    tumor = (x / 31.0).astype(np.float32)
    inflammatory = np.exp(-((x - 7) ** 2 + (y - 16) ** 2) / 50.0).astype(np.float32)
    other = np.zeros_like(tumor)
    spatial = np.stack([tumor, inflammatory, other, other, other], axis=-1)
    np.savez_compressed(maps / "tile.npz", map=spatial)
    pd.DataFrame(
        [np.linspace(-1.0, 1.0, 16, dtype=np.float32)],
        index=["tile"],
        columns=MORPHOLOGY_FEATURE_NAMES,
    ).to_parquet(tmp_path / "morphology_stats.parquet")
    return ConditionStore(tmp_path)


def context(store: ConditionStore) -> InterventionContext:
    return InterventionContext(store, "tile", intervention_seed=42, generation_seed=7)


def changed_indices(before: torch.Tensor, after: torch.Tensor) -> list[int]:
    return torch.nonzero(after - before, as_tuple=False).flatten().tolist()


def test_shape_irregularity_changes_only_declared_means(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    original = store.load("tile")
    expected = sorted(
        MORPHOLOGY_FEATURE_NAMES.index(name)
        for name in ("eccentricity_mean", "solidity_mean", "perimeter_mean")
    )
    for intervention, level in zip(
        shape_interventions(), (0.5, 1.0, 1.5), strict=True
    ):
        applied = intervention.apply(original, context(store))
        delta = applied.condition.morphology - original.morphology
        assert changed_indices(original.morphology, applied.condition.morphology) == expected
        assert float(delta[MORPHOLOGY_FEATURE_NAMES.index("eccentricity_mean")]) == level
        assert float(delta[MORPHOLOGY_FEATURE_NAMES.index("perimeter_mean")]) == level
        assert float(delta[MORPHOLOGY_FEATURE_NAMES.index("solidity_mean")]) == -level
        assert torch.equal(applied.condition.spatial, original.spatial)


def test_stain_brightness_changes_only_rgb_means(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    original = store.load("tile")
    expected = [MORPHOLOGY_FEATURE_NAMES.index(name) for name in ("r_mean", "g_mean", "b_mean")]
    for intervention, level in zip(
        brightness_interventions(), (0.5, 1.0, 1.5), strict=True
    ):
        applied = intervention.apply(original, context(store))
        delta = applied.condition.morphology - original.morphology
        assert changed_indices(original.morphology, applied.condition.morphology) == expected
        torch.testing.assert_close(delta[expected], torch.full((3,), level))
        assert torch.equal(applied.condition.spatial, original.spatial)


def test_tumor_immune_mixing_preserves_mass_and_increases_overlap(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path)
    original = store.load("tile")
    original_mass = original.spatial[1].sum()
    overlaps = [tumor_weighted_overlap(original.spatial[0], original.spatial[1])]
    converted_channels = []
    for intervention in mixing_interventions():
        applied = intervention.apply(original, context(store))
        converted = applied.condition.spatial
        torch.testing.assert_close(converted[1].sum(), original_mass, rtol=1e-5, atol=1e-4)
        assert torch.equal(converted[0], original.spatial[0])
        assert torch.equal(converted[2:], original.spatial[2:])
        assert torch.equal(applied.condition.morphology, original.morphology)
        assert float(converted.min()) >= 0.0 and float(converted.max()) <= 1.0
        overlaps.append(tumor_weighted_overlap(converted[0], converted[1]))
        converted_channels.append(converted[1])
    assert overlaps == sorted(overlaps)
    first_delta = converted_channels[0] - original.spatial[1]
    second_delta = converted_channels[1] - original.spatial[1]
    third_delta = converted_channels[2] - original.spatial[1]
    torch.testing.assert_close(second_delta, 2.0 * first_delta)
    torch.testing.assert_close(third_delta, 3.0 * first_delta)
