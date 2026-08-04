#!/usr/bin/env python
"""Verify Colab GPU, dependencies, asset paths, dataset alignment, and unit tests."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.colab.layout import (
    DEFAULT_CONFIG,
    RuntimePaths,
    valid_cellvit_source,
    valid_checkpoint,
    valid_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = args.config.expanduser().resolve()
    if not config.is_file():
        raise FileNotFoundError(f"Runtime config missing: {config}")
    paths = RuntimePaths.read(config)

    import cupy as cp
    import diffusers
    import numpy as np
    import pandas as pd
    import torch
    import transformers

    if not valid_dataset(paths.data_dir):
        raise RuntimeError(f"Dataset layout is invalid: {paths.data_dir}")
    if not valid_checkpoint(paths.checkpoint_dir):
        raise RuntimeError(f"PathOGen checkpoint layout is invalid: {paths.checkpoint_dir}")
    if not valid_cellvit_source(paths.cellvit_root):
        raise RuntimeError(f"CellViT++ source layout is invalid: {paths.cellvit_root}")
    if paths.cellvit_model is None or not paths.cellvit_model.is_file():
        raise RuntimeError(
            "CellViT++ model is missing. Rerun setup_colab.py with --cellvit-model or "
            "--cellvit-model-url."
        )
    if paths.cellvit_model.stat().st_size < 1_000_000:
        raise RuntimeError(f"CellViT++ checkpoint looks incomplete: {paths.cellvit_model}")
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is unavailable. In Colab select Runtime > Change runtime type > GPU.")

    from experiments.fidelity.data import DatasetCatalog

    catalog = DatasetCatalog(paths.data_dir)
    sample = catalog.sample(catalog.stems[0])
    with np.load(sample.spatial_path) as archive:
        spatial_key = "map" if "map" in archive.files else archive.files[0]
        spatial_shape = list(archive[spatial_key].shape)

    sys.path.insert(0, str(paths.cellvit_root))
    from cellvit.models.cell_segmentation.postprocessing import (
        DetectionCellPostProcessor,
    )

    _ = DetectionCellPostProcessor
    cupy_devices = int(cp.cuda.runtime.getDeviceCount()) if torch.cuda.is_available() else 0
    report = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
        "pandas": pd.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cupy_devices": cupy_devices,
        "aligned_cases": len(catalog.stems),
        "sample_stem": sample.stem,
        "sample_spatial_shape": spatial_shape,
        "checkpoint_dir": str(paths.checkpoint_dir),
        "cellvit_root": str(paths.cellvit_root),
        "cellvit_commit": paths.cellvit_commit,
        "cellvit_model": str(paths.cellvit_model),
        "output_root": str(paths.output_root),
    }
    print(json.dumps(report, indent=2))

    if not args.skip_tests:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "unittest",
                "experiments.tests.test_colab_workflow",
                "experiments.tests.test_fidelity_core",
                "-v",
            ],
            cwd=paths.repo_root,
            check=True,
        )
    print("[verify] Colab environment and all required assets passed validation")


if __name__ == "__main__":
    main()
