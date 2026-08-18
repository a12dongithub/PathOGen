import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cpathogen.counterfactuals import ConditionStore, InterventionContext
from cpathogen.counterfactuals.centroids import render_centroid_channel
from cpathogen.counterfactuals.conditions import MORPHOLOGY_FEATURE_NAMES
from experiments.spatial.nuclear_shape_irregularity import (
    build_interventions as shape_interventions,
)
from experiments.spatial.stain_brightness import (
    build_interventions as brightness_interventions,
)
from experiments.spatial.tumor_immune_mixing import (
    build_interventions as mixing_interventions,
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
        shape_interventions(), (-2.0, -1.0, 0.5, 1.0, 1.5, 2.0), strict=True
    ):
        applied = intervention.apply(original, context(store))
        delta = applied.condition.morphology - original.morphology
        assert (
            changed_indices(original.morphology, applied.condition.morphology)
            == expected
        )
        assert (
            float(delta[MORPHOLOGY_FEATURE_NAMES.index("eccentricity_mean")]) == level
        )
        assert float(delta[MORPHOLOGY_FEATURE_NAMES.index("perimeter_mean")]) == level
        assert float(delta[MORPHOLOGY_FEATURE_NAMES.index("solidity_mean")]) == -level
        assert torch.equal(applied.condition.spatial, original.spatial)


def test_stain_brightness_changes_only_rgb_means(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    original = store.load("tile")
    expected = [
        MORPHOLOGY_FEATURE_NAMES.index(name) for name in ("r_mean", "g_mean", "b_mean")
    ]
    for intervention, level in zip(
        brightness_interventions(), (-2.0, -1.0, 0.5, 1.0, 1.5, 2.0), strict=True
    ):
        applied = intervention.apply(original, context(store))
        delta = applied.condition.morphology - original.morphology
        assert (
            changed_indices(original.morphology, applied.condition.morphology)
            == expected
        )
        torch.testing.assert_close(delta[expected], torch.full((3,), level))
        assert torch.equal(applied.condition.spatial, original.spatial)


def _centroid_feature(x: int, y: int, classification: str) -> dict:
    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [[x - 1, y - 1], [x + 1, y - 1], [x + 1, y + 1], [x - 1, y + 1]]
            ],
        },
        "properties": {"classification": {"name": classification}},
    }


def make_centroid_store(tmp_path: Path) -> ConditionStore:
    maps = tmp_path / "spatial_maps"
    geojsons = tmp_path / "geojsons"
    maps.mkdir()
    geojsons.mkdir()
    tumor = np.asarray(
        [(x, y) for y in range(180, 333, 38) for x in range(180, 333, 38)],
        dtype=np.int16,
    )
    inflammatory = np.asarray(
        [(40 + 30 * index, 60 + 17 * (index % 5)) for index in range(12)],
        dtype=np.int16,
    )
    spatial = np.zeros((512, 512, 5), dtype=np.float32)
    spatial[:, :, 0] = render_centroid_channel(tumor).numpy()
    spatial[:, :, 1] = render_centroid_channel(inflammatory).numpy()
    np.savez_compressed(maps / "tile.npz", map=spatial)
    features = [
        *[_centroid_feature(int(x), int(y), "Neoplastic") for x, y in tumor],
        *[
            _centroid_feature(int(x), int(y), "Inflammatory")
            for x, y in inflammatory
        ],
    ]
    (geojsons / "tile.geojson").write_text(json.dumps(features), encoding="utf-8")
    pd.DataFrame(
        [np.zeros(16, dtype=np.float32)],
        index=["tile"],
        columns=MORPHOLOGY_FEATURE_NAMES,
    ).to_parquet(tmp_path / "morphology_stats.parquet")
    return ConditionStore(tmp_path)


def test_tumor_immune_mixing_preserves_exact_counts_and_orders_distance(
    tmp_path: Path,
) -> None:
    store = make_centroid_store(tmp_path)
    original = store.load("tile")
    distances = []
    hashes = []
    for intervention in mixing_interventions():
        applied = intervention.apply(original, context(store))
        converted = applied.condition.spatial
        assert torch.equal(converted[0], original.spatial[0])
        assert torch.equal(converted[2:], original.spatial[2:])
        assert torch.equal(applied.condition.morphology, original.morphology)
        assert float(converted.min()) >= 0.0 and float(converted.max()) <= 1.0
        assert applied.details["neoplastic_centroid_count_before"] == 25
        assert applied.details["neoplastic_centroid_count_after"] == 25
        assert applied.details["inflammatory_centroid_count_before"] == 12
        assert applied.details["inflammatory_centroid_count_after"] == 12
        distances.append(applied.details["median_nearest_tumor_distance_after_px"])
        hashes.append(applied.details["relocated_inflammatory_centroids_sha256"])
    assert distances == sorted(distances)
    assert len(set(hashes)) == len(hashes)
