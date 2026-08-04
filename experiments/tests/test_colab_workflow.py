from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from experiments.colab.layout import (
    REPO_ROOT,
    RuntimePaths,
    find_cellvit_model,
    find_cellvit_source,
    find_checkpoint,
    find_dataset,
)
from experiments.colab.setup_colab import safe_extract_zip


class ColabWorkflowTests(unittest.TestCase):
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
