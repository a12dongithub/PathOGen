from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cpathogen.annotation.geojson import (
    NucleusPrediction,
    predictions_to_geojson,
    validate_geojson,
)


class AnnotationGeoJSONTest(unittest.TestCase):
    def test_round_trip_contract(self) -> None:
        prediction = NucleusPrediction(
            contour=((2, 2), (7, 2), (7, 8), (2, 8)),
            type_id=2,
            type_probability=0.875,
            centroid=(4.5, 5.0),
            bbox=((2, 2), (7, 8)),
        )
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "tile.png"
            payload = predictions_to_geojson(
                [prediction],
                image_path=image_path,
                image_width=16,
                image_height=16,
                model={"name": "test"},
            )
        summary = validate_geojson(payload, image_width=16, image_height=16)
        self.assertEqual(summary["nucleus_count"], 1)
        self.assertEqual(summary["class_counts"], {"Inflammatory": 1})
        feature = payload["features"][0]
        self.assertEqual(
            feature["properties"]["cellvit_plus_plus"]["type_probability"],
            0.875,
        )
        self.assertEqual(
            feature["geometry"]["coordinates"][0][0],
            feature["geometry"]["coordinates"][0][-1],
        )

    def test_rejects_out_of_bounds_polygon(self) -> None:
        payload = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "id": "bad",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0, 0], [20, 0], [0, 2], [0, 0]]],
                    },
                    "properties": {
                        "classification": {"name": "Neoplastic"}
                    },
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "out of bounds"):
            validate_geojson(payload, image_width=16, image_height=16)

        summary = validate_geojson(
            payload,
            image_width=16,
            image_height=16,
            strict_bounds=False,
        )
        self.assertEqual(summary["out_of_bounds_point_count"], 1)

    def test_allows_explicit_empty_annotation(self) -> None:
        payload = {"type": "FeatureCollection", "features": []}
        summary = validate_geojson(payload, allow_empty=True)
        self.assertEqual(summary["nucleus_count"], 0)


if __name__ == "__main__":
    unittest.main()
