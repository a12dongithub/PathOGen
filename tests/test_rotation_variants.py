import importlib.util
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

from cpathogen.endpoints.variants import normalize_variant_manifests

PREPARE_SCRIPT = (
    Path(__file__).parents[1]
    / "workflows"
    / "10_train_evaluate_endpoint_models"
    / "prepare_rotation_nuisance.py"
)


def test_rotation_manifest_preserves_shared_image_with_distinct_transforms(
    tmp_path: Path,
) -> None:
    image = tmp_path / "tile.png"
    Image.new("RGB", (4, 4), "white").save(image)
    manifest = tmp_path / "images.csv"
    pd.DataFrame(
        [
            {
                "stem": "TCGA-AA-0001_x0_y0_TL",
                "experiment": "image_rotation",
                "condition": condition,
                "local_path": image,
                "augmentation_code": code,
                "seed": 42,
            }
            for condition, code in (
                ("rotation_0", 0),
                ("rotation_90", 4),
                ("rotation_180", 3),
                ("rotation_270", 5),
            )
        ]
    ).to_csv(manifest, index=False)

    result = normalize_variant_manifests([manifest])
    assert len(result) == 4
    assert set(result["condition"]) == {
        "rotation_0",
        "rotation_90",
        "rotation_180",
        "rotation_270",
    }
    assert set(result["augmentation_code"]) == {0, 3, 4, 5}


def test_prepare_rotation_can_cache_source_images_locally(
    tmp_path: Path, monkeypatch
) -> None:
    spec = importlib.util.spec_from_file_location("prepare_rotation", PREPARE_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    stem = "TCGA-AA-0001_x0_y0_TL"
    counterfactuals = tmp_path / "counterfactuals"
    source_dir = counterfactuals / "nuclear_enlargement" / stem
    source_dir.mkdir(parents=True)
    Image.new("RGB", (4, 4), "white").save(source_dir / "baseline.png")
    pd.DataFrame(
        [
            {
                "stem": stem,
                "experiment": "nuclear_enlargement",
                "condition": "baseline",
            }
        ]
    ).to_csv(counterfactuals / "organized_bucket_images.csv", index=False)
    tile_manifest = tmp_path / "tile_manifest.csv"
    pd.DataFrame([{"tile_id": stem}]).to_csv(tile_manifest, index=False)
    output = tmp_path / "rotation"
    cache = tmp_path / "cache"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(PREPARE_SCRIPT),
            "--counterfactual-root",
            str(counterfactuals),
            "--output-dir",
            str(output),
            "--tile-manifest",
            str(tile_manifest),
            "--local-image-cache-dir",
            str(cache),
            "--num-images",
            "1",
        ],
    )
    module.main()

    manifest = pd.read_csv(output / "images.csv")
    assert len(manifest) == 4
    assert manifest["local_path"].nunique() == 1
    assert Path(manifest.iloc[0]["local_path"]) == cache / f"{stem}.png"
    assert (cache / f"{stem}.png").is_file()
