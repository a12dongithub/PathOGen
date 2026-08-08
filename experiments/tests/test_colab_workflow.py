from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

from experiments.colab.layout import (
    REPO_ROOT,
    RuntimePaths,
    find_cellvit_model,
    find_cellvit_source,
    find_checkpoint,
    find_dataset,
)
from experiments.colab.setup_colab import (
    locate_or_prepare_zip,
    resolve_cellvit_model,
    safe_extract_zip,
)
from experiments.fidelity.constants import MORPH_FEATURES
from experiments.fidelity.data import CellObservation
from experiments.fidelity.workflow import load_rgb_with_retry


def load_rerank_module():
    script = REPO_ROOT / "experiments" / "05_cellvit_rerank_fid_kid.py"
    spec = importlib.util.spec_from_file_location("cellvit_rerank_script", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def observation(cell_type: str, x: float, y: float) -> CellObservation:
    contour = np.asarray(
        [[x - 2, y - 2], [x + 2, y - 2], [x + 2, y + 2], [x - 2, y + 2]],
        dtype=np.float32,
    )
    return CellObservation(cell_type=cell_type, centroid=(x, y), contour=contour)


class ColabWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rerank = load_rerank_module()

    def create_layout(self, root: Path) -> RuntimePaths:
        data = root / "nested" / "512_final_dataset"
        for child in ("images", "spatial_maps", "geojsons"):
            (data / child).mkdir(parents=True, exist_ok=True)
        (data / "morphology_stats.parquet").touch()

        checkpoint = root / "models" / "checkpoint-30000"
        (checkpoint / "unet").mkdir(parents=True)
        for relative in (
            "unet/config.json",
            "unet/diffusion_pytorch_model.safetensors",
            "film_mlps.pt",
            "spatial_encoder.pt",
        ):
            (checkpoint / relative).touch()

        cellvit = root / "external" / "repository"
        segmentation = cellvit / "cellvit" / "models" / "cell_segmentation"
        segmentation.mkdir(parents=True)
        (segmentation / "cellvit.py").touch()
        (segmentation / "postprocessing.py").touch()
        cellvit_model = root / "cellvit_models" / "CellViT-256-x40-AMP.pth"
        cellvit_model.parent.mkdir(parents=True)
        cellvit_model.touch()
        output = root / "outputs"
        return RuntimePaths(
            repo_root=REPO_ROOT,
            asset_root=root,
            data_dir=data,
            checkpoint_dir=checkpoint,
            cellvit_root=cellvit,
            cellvit_commit=None,
            cellvit_model=cellvit_model,
            output_root=output,
        )

    def test_asset_finders_and_runtime_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.create_layout(root)
            self.assertEqual(find_dataset(root), paths.data_dir)
            self.assertEqual(find_checkpoint(root), paths.checkpoint_dir)
            self.assertEqual(find_cellvit_source(root), paths.cellvit_root)
            self.assertEqual(find_cellvit_model(root), paths.cellvit_model)
            config = root / "runtime_paths.json"
            paths.write(config)
            loaded = RuntimePaths.read(config)
            self.assertEqual(loaded, paths)

    def test_safe_zip_rejects_parent_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("../escape.txt", "bad")
            with self.assertRaises(RuntimeError):
                safe_extract_zip(archive, root / "extract")
            self.assertFalse((root / "escape.txt").exists())

    def test_cellvit_model_accepts_zip_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "cellvit-model.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr(
                    "nested/CellViT-256-x40-AMP.pth", b"checkpoint"
                )
            args = Namespace(
                cellvit_model=archive,
                cellvit_model_name="CellViT-256-x40-AMP.pth",
            )
            resolved = resolve_cellvit_model(args, root / "assets")
            self.assertEqual(resolved.name, "CellViT-256-x40-AMP.pth")
            self.assertTrue(resolved.is_file())

    def test_mounted_archive_is_extracted_and_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "mounted" / "dataset.zip"
            archive.parent.mkdir()
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("dataset/ready.txt", "ready")

            def find_ready(search_root: Path) -> Path:
                ready = search_root / "dataset" / "ready.txt"
                if not ready.is_file():
                    raise FileNotFoundError(ready)
                return ready.parent

            resolved = locate_or_prepare_zip(
                root / "extract",
                root / "unused-download.zip",
                "https://unused.invalid/file.zip",
                find_ready,
                keep_archive=False,
                source_archive=archive,
            )
            self.assertEqual(resolved, root / "extract" / "dataset")
            self.assertTrue(archive.is_file())

    def test_rerank_has_focused_green_seed_configs(self):
        configs = self.rerank.DEFAULT_CONFIGS
        self.assertEqual(len(configs), 2)
        self.assertEqual(
            self.rerank.DEFAULT_SEEDS_PER_CONFIG * len(configs), 16
        )
        self.assertEqual({config.green_sd for config in configs}, {-1.0, 0.0})
        self.assertEqual({config.controlnet_strength for config in configs}, {2.0})
        self.assertEqual({config.denoising_steps for config in configs}, {30})

    def test_rerank_batches_preserve_order_and_remainder(self):
        self.assertEqual(
            self.rerank.batches(list(range(10)), 4),
            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]],
        )

    def test_rerank_batch_32_spans_two_source_inputs(self):
        self.assertEqual(
            self.rerank.rerank_input_window_size(
                self.rerank.DEFAULT_CONFIGS,
                seeds_per_config=8,
                generation_batch_size=32,
                cellvit_batch_size=32,
            ),
            2,
        )

    def test_rerank_window_accounts_for_different_generation_settings(self):
        configs = (
            self.rerank.CandidateConfig("a", 0.0, 1.0, 20),
            self.rerank.CandidateConfig("b", -1.0, 2.0, 30),
        )
        self.assertEqual(
            self.rerank.rerank_input_window_size(
                configs,
                seeds_per_config=8,
                generation_batch_size=32,
                cellvit_batch_size=32,
            ),
            4,
        )

    def test_drive_image_read_retries_transient_oserror(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "candidate.png"
            Image.new("RGB", (8, 8), (12, 34, 56)).save(image_path)
            real_open = Image.open
            calls = 0

            def flaky_open(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise OSError(5, "Input/output error")
                return real_open(*args, **kwargs)

            with (
                patch("experiments.fidelity.workflow.Image.open", side_effect=flaky_open),
                patch("experiments.fidelity.workflow.time.sleep"),
            ):
                loaded = load_rgb_with_retry(image_path, attempts=2)
            self.assertEqual(loaded.getpixel((0, 0)), (12, 34, 56))
            self.assertEqual(calls, 2)

    def test_rerank_progress_resumes_complete_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stems = ["case_a", "case_b", "case_c"]
            configs = len(self.rerank.DEFAULT_CONFIGS)
            seeds = 2
            score_rows = [
                {"stem": stem, "candidate_order": candidate}
                for stem in stems[:2]
                for candidate in range(configs * seeds)
            ]
            selection_rows = [
                {"stem": stem, "selected_image": f"{stem}.png"}
                for stem in stems[:2]
            ]
            pd.DataFrame(score_rows).to_csv(root / "candidate_scores.csv", index=False)
            pd.DataFrame(selection_rows).to_csv(
                root / "selected_candidates.csv", index=False
            )
            scores, selections, completed = self.rerank.load_rerank_progress(
                root, stems, seeds_per_config=seeds, overwrite=False
            )
            self.assertEqual(completed, 2)
            self.assertEqual(len(scores), 2 * configs * seeds)
            self.assertEqual(len(selections), 2)

    def test_spatial_point_score_is_plus_one_zero_minus_one(self):
        source = [
            observation("Neoplastic", 10, 10),
            observation("Inflammatory", 100, 100),
            observation("Connective", 300, 300),
        ]
        predicted = [
            observation("Neoplastic", 20, 10),
            observation("Neoplastic", 105, 100),
            observation("Epithelial", 500, 500),
        ]
        score = self.rerank.score_spatial_cells(source, predicted, radius=50)
        self.assertEqual(score["same_type_matches"], 1)
        self.assertEqual(score["different_type_matches"], 1)
        self.assertEqual(score["missing_matches"], 1)
        self.assertEqual(score["extra_predictions"], 1)
        self.assertEqual(score["spatial_points"], 0)
        self.assertEqual(score["spatial_score"], 0.0)

    def test_spatial_score_uses_each_prediction_once(self):
        source = [
            observation("Neoplastic", 10, 10),
            observation("Neoplastic", 20, 10),
        ]
        predicted = [observation("Neoplastic", 15, 10)]
        score = self.rerank.score_spatial_cells(source, predicted, radius=50)
        self.assertEqual(score["same_type_matches"], 1)
        self.assertEqual(score["missing_matches"], 1)
        self.assertEqual(score["spatial_points"], 0)

    def test_green_change_is_clamped_and_changes_only_green(self):
        baseline = np.zeros(len(MORPH_FEATURES), dtype=np.float32)
        changed, details = self.rerank.green_condition(
            baseline, green_sd=2.0, lower=-1.0, upper=1.0
        )
        changed_indices = np.flatnonzero(changed != baseline).tolist()
        self.assertEqual(changed_indices, [MORPH_FEATURES.index("g_mean")])
        self.assertEqual(float(changed[MORPH_FEATURES.index("g_mean")]), 1.0)
        self.assertTrue(details["green_clipped"])

    def test_suite_print_only_builds_all_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.create_layout(root)
            config = root / "runtime_paths.json"
            paths.write(config)
            script = REPO_ROOT / "experiments" / "colab" / "run_fidelity_suite.py"
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--config",
                    str(config),
                    "--dry-run",
                    "--print-only",
                    "--num-images",
                    "3",
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            manifest = json.loads((paths.output_root / "suite_commands.json").read_text())
            self.assertEqual(len(manifest["commands"]), 3)
            self.assertIn("02_morphology_fidelity.py", result.stdout)
            self.assertIn("04_spatial_coordinate_fidelity.py", result.stdout)


if __name__ == "__main__":
    unittest.main()
