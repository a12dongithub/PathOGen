from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

from experiments.colab.layout import REPO_ROOT
from experiments.fidelity.constants import MORPH_FEATURES
from experiments.fidelity.data import CellObservation, load_cells
from experiments.fidelity.evaluators import (
    ensure_hovernet_predictions,
    load_hovernet_json,
    save_observations_geojson,
)
from experiments.fidelity.table_metrics import (
    summarize_controlled_morphology,
    summarize_spatial,
)


def load_table_script():
    script = REPO_ROOT / "experiments" / "06_generate_fidelity_tables.py"
    spec = importlib.util.spec_from_file_location(
        "paper_fidelity_tables_script", script
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def observation(
    cell_type: str, x: float, y: float, radius: float = 3
) -> CellObservation:
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


class FakeCatalog:
    def __init__(self, frame: pd.DataFrame):
        self.morphology = frame


class PaperTableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = load_table_script()

    def test_unclassified_round_trip_is_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stardist.geojson"
            save_observations_geojson(
                [observation("Unclassified", 10, 20)], path, "StarDist"
            )
            cells = load_cells(path)
        self.assertEqual(len(cells), 1)
        self.assertEqual(cells[0].cell_type, "Unclassified")

    def test_hovernet_pannuke_mapping(self):
        payload = {
            "mag": None,
            "nuc": {
                "1": {
                    "type": 1,
                    "type_prob": 0.9,
                    "centroid": [20, 30],
                    "contour": [[18, 28], [22, 28], [22, 32], [18, 32]],
                },
                "2": {
                    "type": 2,
                    "centroid": [40, 50],
                    "contour": [[38, 48], [42, 48], [42, 52], [38, 52]],
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hover.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            cells = load_hovernet_json(path)
        self.assertEqual(
            [cell.cell_type for cell in cells], ["Neoplastic", "Inflammatory"]
        )

    def test_hovernet_salvages_raw_scratch_before_reinference(self):
        payload = {
            "nuc": {
                "1": {
                    "type": 1,
                    "centroid": [20, 30],
                    "contour": [[18, 28], [22, 28], [22, 32], [18, 32]],
                }
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "scratch" / "hovernet_raw" / "json" / "case.json"
            raw.parent.mkdir(parents=True)
            raw.write_text(json.dumps(payload), encoding="utf-8")
            outputs = ensure_hovernet_predictions(
                {"case": root / "case.png"},
                root / "predictions",
                root / "scratch",
                project_root=None,
                model_path=None,
                predictions_dir=None,
                batch_size=32,
            )
            self.assertTrue(outputs["case"].is_file())
            self.assertEqual(load_cells(outputs["case"])[0].cell_type, "Neoplastic")

    def test_spatial_summary_perfect_typed_matching(self):
        pairs = []
        for index in range(4):
            source = [
                observation("Neoplastic", 20 + offset * 20, 20)
                for offset in range(index + 1)
            ]
            predicted = [
                observation("Neoplastic", cell.centroid[0] + 2, cell.centroid[1] + 1)
                for cell in source
            ]
            pairs.append((f"TCGA-AA-000{index}_x0_y0", source, predicted))
        row, cases, _, detail = summarize_spatial(
            pairs, "CellViT++", typed=True, bootstrap=0, seed=42
        )
        self.assertAlmostEqual(row["Total Count ρ"], 1.0)
        self.assertAlmostEqual(row["Per Type Count ρ"], 1.0)
        self.assertAlmostEqual(row["Centroid F1 @ 25 px"], 1.0)
        self.assertEqual(len(cases), 4)
        self.assertTrue(detail["typed_centroid_matching"])

    def test_controlled_summary_uses_within_source_rho(self):
        rows = []
        for feature in (
            "area_mean",
            "eccentricity_mean",
            "solidity_mean",
            "grad_mean",
            "r_mean",
            "g_mean",
            "b_mean",
        ):
            for stem_index in range(3):
                for level in (-1, -0.5, 0, 0.5, 1):
                    rows.append(
                        {
                            "feature": feature,
                            "stem": f"TCGA-AA-00{stem_index}_x0_y0",
                            "patient": f"TCGA-AA-00{stem_index}",
                            "input_value": level,
                            "measured_value": 2 * level + stem_index,
                        }
                    )
        row, _, per_source = summarize_controlled_morphology(
            pd.DataFrame(rows), "Within Image Controlled CellViT++", 0, 42
        )
        self.assertTrue(
            all(np.isclose(row[column], 1.0) for column in row if column != "Method")
        )
        self.assertEqual(len(per_source), 21)

    def test_controlled_plan_has_five_doses_and_unique_plan_ids(self):
        stems = [f"TCGA-AA-{index:04d}_x0_y0" for index in range(40)]
        values = np.linspace(-0.5, 0.5, len(stems))
        frame = pd.DataFrame(
            {
                feature: values + feature_index * 0.01
                for feature_index, feature in enumerate(MORPH_FEATURES)
            },
            index=stems,
        )
        selected = pd.DataFrame(
            {"stem": stems, "selected_seed": np.arange(len(stems), dtype=int)}
        )
        plan = self.script.build_controlled_plan(
            selected,
            FakeCatalog(frame),
            count=5,
            levels=[-0.5, -0.25, 0.0, 0.25, 0.5],
            seed=42,
        )
        self.assertEqual(len(plan), 5 * 5 * 7)
        self.assertEqual(plan["plan_id"].nunique(), len(plan))
        self.assertTrue((plan.groupby(["feature", "stem"]).size() == 5).all())
        self.assertTrue((plan["input_delta_std"] == plan["dose_sd"]).all())
        self.assertTrue(
            (plan["input_value"] == plan["baseline_value"] + plan["dose_sd"]).all()
        )
        source_sets = plan.groupby("feature")["stem"].apply(set)
        self.assertTrue(
            all(feature_stems == source_sets.iloc[0] for feature_stems in source_sets)
        )

    def test_random_tile_pairing_changes_patient(self):
        stems = [f"TCGA-AA-{index:04d}_x0_y0" for index in range(20)]
        mapping = self.script.random_tile_derangement(stems, 42)
        self.assertEqual(set(mapping), set(stems))
        self.assertEqual(set(mapping.values()), set(stems))
        self.assertTrue(
            all(
                self.script.patient_id(source) != self.script.patient_id(target)
                for source, target in mapping.items()
            )
        )

    def test_cli_dry_run_writes_both_experiment_plans(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "512_final_dataset"
            rerank = root / "rerank"
            selected_dir = rerank / "metric_sets" / "selected"
            output = root / "tables"
            for child in ("images", "spatial_maps", "geojsons"):
                (data / child).mkdir(parents=True, exist_ok=True)
            selected_dir.mkdir(parents=True)

            stems = [f"TCGA-AA-{index:04d}_x0_y0" for index in range(40)]
            values = np.linspace(-0.5, 0.5, len(stems))
            frame = pd.DataFrame(
                {
                    feature: values + feature_index * 0.01
                    for feature_index, feature in enumerate(MORPH_FEATURES)
                },
                index=stems,
            )
            frame.to_parquet(data / "morphology_stats.parquet")
            selections = []
            for index, stem in enumerate(stems):
                Image.new("RGB", (8, 8), (100, 120, 140)).save(
                    data / "images" / f"{stem}.png"
                )
                Image.new("RGB", (8, 8), (90, 110, 130)).save(
                    selected_dir / f"{stem}.png"
                )
                (data / "spatial_maps" / f"{stem}.npz").touch()
                (data / "geojsons" / f"{stem}.geojson").write_text(
                    '{"type":"FeatureCollection","features":[]}', encoding="utf-8"
                )
                selections.append(
                    {
                        "stem": stem,
                        "seed": index,
                        "selected_image": f"/stale/path/{stem}.png",
                        "green_applied": float(frame.loc[stem, "g_mean"]),
                    }
                )
            pd.DataFrame(selections).to_csv(
                rerank / "selected_candidates.csv", index=False
            )

            argv = [
                "06_generate_fidelity_tables.py",
                "--config",
                str(root / "missing_config.json"),
                "--data-dir",
                str(data),
                "--rerank-dir",
                str(rerank),
                "--output-dir",
                str(output),
                "--scratch-dir",
                str(root / "scratch"),
                "--num-images",
                "20",
                "--controlled-images",
                "3",
                "--controlled-levels",
                "-0.5",
                "-0.25",
                "0",
                "0.25",
                "0.5",
                "--dry-run",
            ]
            with patch.object(sys, "argv", argv):
                self.script.main()
            self.assertEqual(len(pd.read_csv(output / "spatial_case_plan.csv")), 20)
            controlled = pd.read_csv(output / "controlled_condition_plan.csv")
            self.assertEqual(len(controlled), 3 * 5 * 7)
            self.assertTrue((output / "experiment_manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
