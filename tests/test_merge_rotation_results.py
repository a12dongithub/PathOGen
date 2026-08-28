from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

WORKFLOW = (
    Path(__file__).parents[1]
    / "workflows"
    / "11_tile_local_xai_rotation_virchow2"
)
SCRIPT = WORKFLOW / "merge_rotation_results.py"


def test_merge_rotation_results_emits_subset_table(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.syspath_prepend(str(WORKFLOW))
    spec = importlib.util.spec_from_file_location("merge_rotation_results", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    base = tmp_path / "base"
    output = tmp_path / "output"
    base.mkdir()
    experiments = (
        ("stain_brightness", "nuisance"),
        ("nuclear_enlargement", "biological"),
        ("nuclear_shape_irregularity", "biological"),
        ("peritumoral_immune_ring_diameter40px", "biological"),
        ("tumor_immune_separation_diameter40px", "biological"),
    )
    rows = []
    for endpoint, bag_size in (("PAM50", None), ("Overall survival", 16)):
        for experiment, family in experiments:
            rows.append(
                {
                    "model_id": "resnet50",
                    "endpoint": endpoint,
                    "bag_size": bag_size,
                    "experiment": experiment,
                    "family": family,
                    "mean_tvd": 0.2 if family == "biological" else 0.1,
                    "flip_rate": 0.05,
                }
            )
    pd.DataFrame(rows).to_csv(
        base / "experiment_summary_with_pathlupi.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "model_id": "resnet50",
                "endpoint": "PAM50",
                "bag_size": None,
                "patient_macro_auc_ovr": 0.75,
                "c_index": None,
            },
            {
                "model_id": "resnet50",
                "endpoint": "Overall survival",
                "bag_size": 16,
                "patient_macro_auc_ovr": None,
                "c_index": 0.6,
            },
        ]
    ).to_csv(base / "performance_with_pathlupi.csv", index=False)
    rotation = tmp_path / "rotation.csv"
    pd.DataFrame(
        [
            {
                "model_id": "resnet50",
                "endpoint": endpoint,
                "bag_size": bag_size,
                "experiment": "image_rotation",
                "family": "nuisance",
                "mean_tvd": 0.1,
                "flip_rate": 0.04,
            }
            for endpoint, bag_size in (("PAM50", None), ("Overall survival", 16))
        ]
    ).to_csv(rotation, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--base-results-root",
            str(base),
            "--rotation-summary",
            str(rotation),
            "--output-root",
            str(output),
            "--models",
            "resnet50",
        ],
    )
    module.main()

    table = pd.read_csv(output / "table4_rotation_without_virchow.csv")
    assert len(table) == 2
    assert set(table["Image rotation"]) == {"0.1000 / 0.0400"}
    assert set(table["BNR"].round(4)) == {2.0}
