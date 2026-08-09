from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from experiments.fidelity.cellvit import CellViTRunner
from experiments.fidelity.constants import MORPH_FEATURES
from experiments.fidelity.data import CellObservation, load_cells
from experiments.fidelity.generation import resolve_memory_mode
from experiments.fidelity.guidance import (
    CandidateDecision,
    GenerationContext,
    apply_retry_feedback,
)
from experiments.fidelity.measurements import morphology_measurements
from experiments.fidelity.spatial import cell_counts, coordinate_metrics, density_grid
from experiments.fidelity.statistics import benjamini_hochberg, spearman_with_bootstrap
from experiments.fidelity.workflow import deterministic_seed


def cell(cell_type: str, x: float, y: float, radius: int = 4) -> CellObservation:
    contour = np.asarray(
        [
            [x - radius, y - radius],
            [x + radius, y - radius],
            [x + radius, y + radius],
            [x - radius, y + radius],
        ],
        dtype=np.float32,
    )
    return CellObservation(cell_type, (x, y), contour)


class FidelityCoreTests(unittest.TestCase):
    def test_generator_memory_mode_uses_large_gpus_for_throughput(self):
        self.assertEqual(resolve_memory_mode("auto", 40.0, "fp16"), "throughput")
        self.assertEqual(resolve_memory_mode("auto", 22.0, "fp16"), "throughput")
        self.assertEqual(resolve_memory_mode("auto", 16.0, "fp16"), "balanced")
        self.assertEqual(resolve_memory_mode("auto", 40.0, "fp32"), "low-vram")
        self.assertEqual(resolve_memory_mode("balanced", 40.0, "fp16"), "balanced")

    def test_morphology_pair_analysis(self):
        script = Path(__file__).parents[1] / "02_morphology_fidelity.py"
        spec = importlib.util.spec_from_file_location(
            "morphology_fidelity_script", script
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rows = []
        for index in range(5):
            rows.extend(
                [
                    {
                        "stem": f"case_{index}",
                        "condition_id": "baseline",
                        "input_area_mean": float(index),
                        "measured_area_mean": float(index * 2 + 10),
                    },
                    {
                        "stem": f"case_{index}",
                        "condition_id": "increase__area_mean",
                        "input_area_mean": float(index + 0.5),
                        "measured_area_mean": float(index * 2 + 11 + index / 10),
                    },
                ]
            )
        result = module.analyze(pd.DataFrame(rows), ["area_mean"], bootstrap=0, seed=7)
        self.assertEqual(int(result.loc[0, "pairs"]), 5)
        self.assertGreater(float(result.loc[0, "pooled_rho"]), 0.9)
        self.assertEqual(float(result.loc[0, "direction_accuracy"]), 1.0)

    def test_morphology_feature_order_and_values(self):
        image = Image.new("RGB", (512, 512), (100, 120, 140))
        result = morphology_measurements(
            image,
            [cell("Neoplastic", 100, 100), cell("Inflammatory", 200, 200, 6)],
        )
        self.assertEqual(list(result)[:-1], MORPH_FEATURES)
        self.assertEqual(result["detected_nuclei"], 2)
        self.assertGreater(result["area_mean"], 0)
        self.assertAlmostEqual(result["r_mean"], 100, places=4)
        self.assertAlmostEqual(result["g_mean"], 120, places=4)
        self.assertAlmostEqual(result["b_mean"], 140, places=4)

    def test_spatial_counts_grids_and_coordinate_metrics(self):
        source = [
            cell("Neoplastic", 50, 50),
            cell("Neoplastic", 300, 300),
            cell("Inflammatory", 100, 400),
        ]
        predicted = [
            cell("Neoplastic", 52, 49),
            cell("Neoplastic", 302, 301),
            cell("Inflammatory", 103, 398),
        ]
        self.assertEqual(cell_counts(source)["Total"], 3)
        self.assertEqual(cell_counts(source)["Neoplastic"], 2)
        self.assertEqual(density_grid(source, "Total", 8).sum(), 3)
        metrics, matches, source_grid, predicted_grid = coordinate_metrics(
            source, predicted, "Total", 8, 16, 0, 42
        )
        self.assertEqual(metrics["matched_pairs"], 3)
        self.assertLess(metrics["median_match_distance"], 5)
        self.assertEqual(source_grid.shape, (8, 8))
        self.assertEqual(predicted_grid.shape, (8, 8))
        self.assertEqual(len(matches.distances), 3)

    def test_spearman_and_fdr(self):
        x = np.arange(20)
        result = spearman_with_bootstrap(x, x * 2, bootstrap=50, seed=7)
        self.assertAlmostEqual(result.rho, 1.0)
        adjusted = benjamini_hochberg([0.01, 0.04, 0.03])
        self.assertTrue(all(0 <= value <= 1 for value in adjusted))
        self.assertLessEqual(adjusted[0], adjusted[1])

    def test_amp_predictions_cast_to_float32(self):
        predictions = {
            "nuclei_binary_map": torch.randn(1, 2, 8, 8, dtype=torch.float16),
            "nuclei_type_map": torch.randn(1, 6, 8, 8, dtype=torch.float16),
            "hv_map": torch.randn(1, 2, 8, 8, dtype=torch.float16),
        }
        prepared = CellViTRunner._prepare(predictions)
        self.assertEqual(prepared["nuclei_binary_map"].dtype, torch.float32)
        self.assertEqual(tuple(prepared["nuclei_type_map"].shape), (1, 8, 8, 6))

    def test_guidance_retry_feedback_changes_only_requested_state(self):
        context = GenerationContext(
            stem="case",
            condition_id="baseline",
            spatial_map=np.zeros((512, 512, 5), dtype=np.uint8),
            morphology=np.zeros(16, dtype=np.float32),
            seed=10,
        )
        delta = np.zeros(16, dtype=np.float32)
        delta[0] = 0.25
        decision = CandidateDecision(
            accept=False,
            score=0.1,
            reason="weak",
            next_morphology_delta=tuple(delta),
            next_spatial_scale=1.1,
        )
        updated = apply_retry_feedback(context, decision, retry_seed=11)
        self.assertEqual(updated.seed, 11)
        self.assertAlmostEqual(float(updated.morphology[0]), 0.25)
        self.assertEqual(np.count_nonzero(updated.morphology), 1)
        self.assertEqual(updated.metadata["guidance_spatial_scale"], 1.1)

    def test_geojson_alias_and_seed_determinism(self):
        payload = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[1, 1], [5, 1], [5, 5], [1, 5], [1, 1]]],
                    },
                    "properties": {
                        "classification": {"name": "Non-neoplastic epithelium"},
                        "centroid": [3, 3],
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cells.geojson"
            import json

            path.write_text(json.dumps(payload), encoding="utf-8")
            cells = load_cells(path)
        self.assertEqual(cells[0].cell_type, "Epithelial")
        self.assertEqual(
            deterministic_seed(42, "a", "b"), deterministic_seed(42, "a", "b")
        )


if __name__ == "__main__":
    unittest.main()
