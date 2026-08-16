from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cpathogen.counterfactuals import ConditionStore, InterventionContext
from cpathogen.counterfactuals.conditions import MORPHOLOGY_FEATURE_NAMES
from experiments.spatial.intratumoral_immune_hotspots import build_interventions as hotspot_interventions
from experiments.spatial.peritumoral_immune_ring import build_interventions as ring_interventions
from experiments.spatial.tumor_boundary_replacement import build_interventions as replacement_interventions


def make_store(tmp_path: Path) -> ConditionStore:
    maps = tmp_path / "spatial_maps"
    maps.mkdir()
    y, x = np.mgrid[0:512, 0:512]
    tumor = np.exp(-((x - 256) ** 2 + (y - 256) ** 2) / 12000.0).astype(np.float32)
    inflammatory = np.zeros_like(tumor)
    spatial = np.stack([tumor, inflammatory, inflammatory, inflammatory, inflammatory], axis=-1)
    np.savez_compressed(maps / "tile.npz", map=spatial)
    pd.DataFrame(
        [np.zeros(16, dtype=np.float32)], index=["tile"], columns=MORPHOLOGY_FEATURE_NAMES
    ).to_parquet(tmp_path / "morphology_stats.parquet")
    return ConditionStore(tmp_path)


def context(store: ConditionStore) -> InterventionContext:
    return InterventionContext(store, "tile", intervention_seed=42, generation_seed=7)


def test_centroid_pilots_are_nested_and_only_add_inflammatory_signal(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    original = store.load("tile")
    for interventions in (ring_interventions(), hotspot_interventions()):
        masses = []
        for intervention in interventions:
            converted = intervention.apply(original, context(store)).condition
            assert torch.equal(converted.spatial[0], original.spatial[0])
            assert torch.equal(converted.spatial[2:], original.spatial[2:])
            assert torch.equal(converted.morphology, original.morphology)
            masses.append(float(converted.spatial[1].sum()))
        assert masses == sorted(masses)


def test_boundary_replacement_moves_signal_between_declared_channels(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    original = store.load("tile")
    tumor_masses = []
    immune_masses = []
    for intervention in replacement_interventions():
        converted = intervention.apply(original, context(store)).condition
        tumor_masses.append(float(converted.spatial[0].sum()))
        immune_masses.append(float(converted.spatial[1].sum()))
        assert torch.equal(converted.spatial[2:], original.spatial[2:])
        assert torch.equal(converted.morphology, original.morphology)
    assert tumor_masses == sorted(tumor_masses, reverse=True)
    assert immune_masses == sorted(immune_masses)
