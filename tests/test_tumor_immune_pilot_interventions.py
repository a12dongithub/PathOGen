from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch

from cpathogen.counterfactuals import ConditionStore, InterventionContext
from cpathogen.counterfactuals.conditions import MORPHOLOGY_FEATURE_NAMES
from experiments.spatial.intratumoral_immune_hotspots import build_interventions as hotspot_interventions
from experiments.spatial._tumor_immune_pilot_utils import (
    intratumoral_weights,
    sample_nested_centroids,
)


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


def test_hotspots_are_nested_and_only_add_inflammatory_signal(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    original = store.load("tile")
    immune_masses = []
    for intervention in hotspot_interventions():
        applied = intervention.apply(original, context(store))
        converted = applied.condition
        assert torch.equal(converted.spatial[0], original.spatial[0])
        assert torch.equal(converted.spatial[2:], original.spatial[2:])
        assert torch.equal(converted.morphology, original.morphology)
        assert applied.details["added_centroid_count"] == intervention.centroid_count
        assert 0.0 <= applied.details["clipped_pixel_fraction"] <= 1.0
        assert (
            0.0
            <= applied.details["added_centroid_fraction_in_high_tumor_mask"]
            <= 1.0
        )
        immune_masses.append(float(converted.spatial[1].sum()))
    assert immune_masses == sorted(immune_masses)


def test_hotspot_centroids_remain_nested_in_dense_regions(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    original = store.load("tile")
    weights = intratumoral_weights(original.spatial[0])
    centroids = [
        sample_nested_centroids(weights, count, rng=random.Random(1729))
        for count in (80, 160, 320)
    ]
    assert np.array_equal(centroids[0], centroids[1][:80])
    assert np.array_equal(centroids[1], centroids[2][:160])
    assert len(np.unique(centroids[2], axis=0)) == 320
