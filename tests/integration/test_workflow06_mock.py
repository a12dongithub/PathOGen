from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


REPO = Path(__file__).resolve().parents[2]
RUN_PATH = REPO / "workflows/06_evaluate_phase2_fid_kid/run.py"


def _load_run_module():
    spec = importlib.util.spec_from_file_location("workflow06_run", RUN_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_workflow06_writes_matched_pairs_manifest_and_legacy_grid(tmp_path, monkeypatch) -> None:
    data = tmp_path / "data"
    images = data / "images"
    maps = data / "spatial_maps"
    images.mkdir(parents=True)
    maps.mkdir()
    stems = ["tile_a", "tile_b"]
    for index, stem in enumerate(stems):
        Image.new("RGB", (512, 512), (index * 80, 10, 20)).save(images / f"{stem}.png")
        value = np.zeros((512, 512, 5), dtype=np.uint8)
        value[100:200, 100:200, index] = 255
        np.savez_compressed(maps / f"{stem}.npz", map=value)
    columns = [
        "area_mean", "area_var", "eccentricity_mean", "eccentricity_var",
        "solidity_mean", "solidity_var", "perimeter_mean", "perimeter_var",
        "grad_mean", "grad_var", "r_mean", "r_var", "g_mean", "g_var", "b_mean", "b_var",
    ]
    pd.DataFrame(np.zeros((2, 16)), index=stems, columns=columns).to_parquet(
        data / "morphology_stats.parquet"
    )

    module = _load_run_module()
    monkeypatch.setattr(module, "load_phase2_generation_models", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        module,
        "generate_matched_conditions",
        lambda _models, conditions, **kwargs: [
            Image.new("RGB", (512, 512), (0, 100 + i, 0)) for i, _ in enumerate(conditions)
        ],
    )
    output = tmp_path / "run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py", "--data-root", str(data), "--images-dir", str(images),
            "--checkpoint", str(tmp_path / "checkpoint_30000"), "--output-dir", str(output),
            "--num-tiles", "2", "--sample-seed", "42", "--skip-metrics", "--num-grids", "1",
        ],
    )
    module.main()

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["steps"] == 30
    assert manifest["spatial_strength"] == 2.0
    assert len(manifest["records"]) == 2
    assert len(list((output / "generated").glob("*.png"))) == 2
    assert len(list((output / "real").glob("*.png"))) == 2
    assert len(list((output / "grids").glob("*.png"))) == 1
