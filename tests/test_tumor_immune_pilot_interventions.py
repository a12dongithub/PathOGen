import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cpathogen.counterfactuals import ConditionStore, InterventionContext
from cpathogen.counterfactuals.centroids import render_centroid_channel
from cpathogen.counterfactuals.conditions import MORPHOLOGY_FEATURE_NAMES
from experiments.spatial._tumor_immune_pilot_utils import (
    intratumoral_weights,
    sample_nested_centroids,
)
from experiments.spatial.intratumoral_immune_hotspots import (
    build_interventions as hotspot_interventions,
)
from experiments.spatial.peritumoral_immune_ring import (
    build_interventions as ring_interventions,
)


def make_store(tmp_path: Path) -> ConditionStore:
    maps = tmp_path / "spatial_maps"
    maps.mkdir()
    y, x = np.mgrid[0:512, 0:512]
    tumor = np.exp(-((x - 256) ** 2 + (y - 256) ** 2) / 12000.0).astype(np.float32)
    inflammatory_centroids = np.asarray(
        [(80, 80), (120, 400), (390, 100), (430, 430), (60, 260)],
        dtype=np.int16,
    )
    inflammatory = render_centroid_channel(inflammatory_centroids).numpy()
    spatial = np.stack([tumor, inflammatory, inflammatory, inflammatory, inflammatory], axis=-1)
    np.savez_compressed(maps / "tile.npz", map=spatial)
    geojsons = tmp_path / "geojsons"
    geojsons.mkdir()
    features = []
    for x_value, y_value in inflammatory_centroids:
        x_value, y_value = int(x_value), int(y_value)
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [x_value - 1, y_value - 1],
                        [x_value + 1, y_value - 1],
                        [x_value + 1, y_value + 1],
                        [x_value - 1, y_value + 1],
                    ]],
                },
                "properties": {"classification": {"name": "Inflammatory"}},
            }
        )
    (geojsons / "tile.geojson").write_text(json.dumps(features), encoding="utf-8")
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


def test_peritumoral_ring_adds_exact_nested_counts(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    original = store.load("tile")
    resulting_counts = []
    for intervention, expected_addition in zip(
        ring_interventions(), (80, 160, 320), strict=True
    ):
        applied = intervention.apply(original, context(store))
        assert torch.equal(applied.condition.spatial[0], original.spatial[0])
        assert torch.equal(applied.condition.spatial[2:], original.spatial[2:])
        assert torch.equal(applied.condition.morphology, original.morphology)
        assert applied.details["original_inflammatory_centroid_count"] == 5
        assert applied.details["added_inflammatory_centroid_count"] == expected_addition
        assert applied.details["resulting_inflammatory_centroid_count"] == 5 + expected_addition
        assert applied.details["added_centroid_fraction_in_declared_ring"] == 1.0
        resulting_counts.append(
            applied.details["resulting_inflammatory_centroid_count"]
        )
    assert resulting_counts == sorted(resulting_counts)
