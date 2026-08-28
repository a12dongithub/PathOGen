from pathlib import Path

import pandas as pd
from PIL import Image

from cpathogen.endpoints.variants import normalize_variant_manifests


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
