from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

SCRIPT = (
    Path(__file__).parents[1]
    / "workflows"
    / "11_tile_local_xai_rotation_virchow2"
    / "prepare_colab_inputs.py"
)


def load_script():
    spec = importlib.util.spec_from_file_location("prepare_colab_inputs", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_colab_inputs_discovers_and_stages(tmp_path: Path, monkeypatch) -> None:
    module = load_script()
    mydrive = tmp_path / "MyDrive"
    cvpr = mydrive / "PTRI" / "CVPR"
    cvpr.mkdir(parents=True)

    with zipfile.ZipFile(cvpr / "512_final_dataset.zip", "w") as archive:
        archive.writestr("512_final_dataset/morphology_stats.parquet", b"fixture")
        archive.writestr("512_final_dataset/images/tile.png", b"fixture")

    for index in range(1, 8):
        archive_path = cvpr / f"CPathOGen_Counterfactuals-part-{index}.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            root = "CPathOGen_Counterfactuals"
            if index == 1:
                archive.writestr(f"{root}/organized_bucket_images.csv", "fixture")
                archive.writestr(f"{root}/nuclear_enlargement/item.txt", "fixture")
                archive.writestr(f"{root}/stain_brightness/item.txt", "fixture")
            else:
                archive.writestr(f"{root}/part_{index}.txt", "fixture")

    endpoint = cvpr / "PathOGenResults" / "endpoint_models"
    (endpoint / "embedding_cache").mkdir(parents=True)
    for name in module.REQUIRED_ENDPOINT_FILES:
        (endpoint / name).write_text("fixture", encoding="utf-8")
    for name in module.REQUIRED_CACHES:
        (endpoint / "embedding_cache" / name).write_bytes(b"fixture")
    (endpoint / "models" / "resnet50").mkdir(parents=True)
    for name in module.REQUIRED_FOLD_FILES:
        (endpoint / "models" / "resnet50" / name).write_text(
            "fixture", encoding="utf-8"
        )

    work = tmp_path / "work"
    output = cvpr / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--mydrive-root",
            str(mydrive),
            "--cvpr-root",
            str(cvpr),
            "--work-root",
            str(work),
            "--output-root",
            str(output),
        ],
    )
    module.main()

    resolved = json.loads((output / "resolved_paths.json").read_text())
    assert Path(resolved["real_images_dir"]).is_dir()
    assert Path(resolved["counterfactual_root"]).name == "CPathOGen_Counterfactuals"
    assert Path(resolved["endpoint_root"]) == output / "endpoint_models"
    assert module.valid_endpoint_root(output / "endpoint_models")


def test_prepare_colab_inputs_uses_extracted_counterfactuals_without_dataset(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_script()
    mydrive = tmp_path / "MyDrive"
    cvpr = mydrive / "PTRI" / "CVPR"
    counterfactuals = cvpr / "CPathOGen_Counterfactuals"
    (counterfactuals / "nuclear_enlargement").mkdir(parents=True)
    (counterfactuals / "stain_brightness").mkdir()
    (counterfactuals / "organized_bucket_images.csv").write_text(
        "fixture", encoding="utf-8"
    )

    endpoint = cvpr / "PathOGenResults" / "endpoint_models"
    (endpoint / "embedding_cache").mkdir(parents=True)
    for name in module.REQUIRED_ENDPOINT_FILES:
        (endpoint / name).write_text("fixture", encoding="utf-8")
    for name in module.REQUIRED_CACHES:
        (endpoint / "embedding_cache" / name).write_bytes(b"fixture")
    (endpoint / "models" / "resnet50").mkdir(parents=True)
    for name in module.REQUIRED_FOLD_FILES:
        (endpoint / "models" / "resnet50" / name).write_text(
            "fixture", encoding="utf-8"
        )

    output = cvpr / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--mydrive-root",
            str(mydrive),
            "--cvpr-root",
            str(cvpr),
            "--work-root",
            str(tmp_path / "work"),
            "--output-root",
            str(output),
            "--counterfactual-source",
            str(counterfactuals),
            "--skip-dataset",
        ],
    )
    module.main()

    resolved = json.loads((output / "resolved_paths.json").read_text())
    assert resolved["dataset_root"] is None
    assert resolved["real_images_dir"] is None
    assert Path(resolved["counterfactual_root"]) == counterfactuals
    assert module.valid_endpoint_root(output / "endpoint_models")
