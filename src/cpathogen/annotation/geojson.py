"""Canonical per-nucleus GeoJSON conversion and validation."""

from __future__ import annotations

import json
import math
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

PANNUKE_CLASS_NAMES = {
    1: "Neoplastic",
    2: "Inflammatory",
    3: "Connective",
    4: "Dead",
    5: "Epithelial",
}

PANNUKE_CLASS_COLORS = {
    1: [255, 0, 0],
    2: [34, 221, 77],
    3: [35, 92, 236],
    4: [254, 255, 0],
    5: [255, 159, 68],
}


@dataclass(frozen=True)
class NucleusPrediction:
    """One CellViT++ nucleus instance in image pixel coordinates."""

    contour: tuple[tuple[float, float], ...]
    type_id: int
    type_probability: float
    centroid: tuple[float, float]
    bbox: tuple[tuple[float, float], tuple[float, float]]


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _closed_ring(contour: Iterable[Iterable[float]]) -> list[list[float]]:
    ring = [[float(point[0]), float(point[1])] for point in contour]
    if len(ring) < 3:
        raise ValueError("A nucleus contour must contain at least three points")
    if ring[0] != ring[-1]:
        ring.append(ring[0].copy())
    if len({(point[0], point[1]) for point in ring[:-1]}) < 3:
        raise ValueError("A nucleus contour must contain three distinct points")
    return ring


def predictions_to_geojson(
    predictions: Iterable[NucleusPrediction],
    *,
    image_path: str | Path,
    image_width: int,
    image_height: int,
    model: dict[str, Any],
    source_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a standard FeatureCollection with one polygon per nucleus."""
    image_path = Path(image_path).expanduser().resolve()
    features: list[dict[str, Any]] = []
    for index, prediction in enumerate(predictions):
        if prediction.type_id not in PANNUKE_CLASS_NAMES:
            raise ValueError(f"Unsupported PanNuke nucleus type: {prediction.type_id}")
        ring = _closed_ring(prediction.contour)
        identifier = uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"{image_path}|{index}|{prediction.type_id}|{ring}",
        )
        features.append(
            {
                "type": "Feature",
                "id": str(identifier),
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": {
                    "objectType": "annotation",
                    "classification": {
                        "name": PANNUKE_CLASS_NAMES[prediction.type_id],
                        "color": PANNUKE_CLASS_COLORS[prediction.type_id],
                    },
                    "cellvit_plus_plus": {
                        "type_id": prediction.type_id,
                        "type_probability": float(prediction.type_probability),
                        "centroid": [
                            float(prediction.centroid[0]),
                            float(prediction.centroid[1]),
                        ],
                        "bbox": [
                            [float(value) for value in prediction.bbox[0]],
                            [float(value) for value in prediction.bbox[1]],
                        ],
                    },
                },
            }
        )
    return {
        "type": "FeatureCollection",
        "features": features,
        "cpathogen_annotation": {
            "schema_version": 1,
            "source_image": str(image_path),
            "image_width": image_width,
            "image_height": image_height,
            "coordinate_system": "image_pixels_xy_origin_top_left",
            "model": model,
            "source_metadata": source_metadata or {},
        },
    }


def _features(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and payload.get("type") == "FeatureCollection":
        features = payload.get("features")
        if isinstance(features, list):
            return features
    if isinstance(payload, dict) and isinstance(payload.get("features"), list):
        return payload["features"]
    raise ValueError("GeoJSON must be a feature list or FeatureCollection")


def _polygon_rings(geometry: dict[str, Any]) -> list[list[list[float]]]:
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if geometry_type == "Polygon" and isinstance(coordinates, list) and coordinates:
        return [coordinates[0]]
    if geometry_type == "MultiPolygon" and isinstance(coordinates, list):
        return [part[0] for part in coordinates if isinstance(part, list) and part]
    raise ValueError(f"Expected Polygon or MultiPolygon, found {geometry_type!r}")


def validate_geojson(
    payload: Any,
    *,
    image_width: int | None = None,
    image_height: int | None = None,
    allow_empty: bool = False,
    strict_bounds: bool = True,
) -> dict[str, Any]:
    """Validate geometry/classes and return nucleus and class counts.

    ``strict_bounds=False`` is intended only for legacy annotations whose
    contours can cross a tile edge. Newly generated annotations are clipped
    and should always use the strict default.
    """
    features = _features(payload)
    if not features and not allow_empty:
        raise ValueError("Annotation contains no nucleus features")

    identifiers: set[str] = set()
    class_counts: Counter[str] = Counter()
    nucleus_count = 0
    out_of_bounds_point_count = 0
    for feature_index, feature in enumerate(features):
        if not isinstance(feature, dict) or feature.get("type") != "Feature":
            raise ValueError(f"Feature {feature_index} is not a GeoJSON Feature")
        identifier = str(feature.get("id", ""))
        if identifier:
            if identifier in identifiers:
                raise ValueError(f"Duplicate GeoJSON feature id: {identifier}")
            identifiers.add(identifier)
        properties = feature.get("properties") or {}
        classification = properties.get("classification") or {}
        class_name = classification.get("name")
        if class_name not in PANNUKE_CLASS_NAMES.values():
            raise ValueError(
                f"Feature {feature_index} has unsupported class {class_name!r}"
            )
        cellvit = properties.get("cellvit_plus_plus") or {}
        probability = cellvit.get("type_probability")
        if probability is not None and (
            not _finite_number(probability) or not 0.0 <= float(probability) <= 1.0
        ):
            raise ValueError(
                f"Feature {feature_index} has invalid type probability {probability!r}"
            )
        rings = _polygon_rings(feature.get("geometry") or {})
        if not rings:
            raise ValueError(f"Feature {feature_index} has no polygon rings")
        for ring in rings:
            if len(ring) < 4:
                raise ValueError(f"Feature {feature_index} polygon is not closed")
            if list(ring[0]) != list(ring[-1]):
                raise ValueError(f"Feature {feature_index} polygon is not closed")
            for point in ring:
                if not isinstance(point, list) or len(point) < 2:
                    raise ValueError(f"Feature {feature_index} has an invalid point")
                x, y = point[:2]
                if not _finite_number(x) or not _finite_number(y):
                    raise ValueError(f"Feature {feature_index} has non-finite coordinates")
                point_out_of_bounds = (
                    image_width is not None and not 0 <= float(x) < image_width
                ) or (
                    image_height is not None and not 0 <= float(y) < image_height
                )
                if point_out_of_bounds:
                    if strict_bounds:
                        raise ValueError(
                            f"Feature {feature_index} coordinate is out of bounds"
                        )
                    out_of_bounds_point_count += 1
            nucleus_count += 1
            class_counts[class_name] += 1
    return {
        "feature_count": len(features),
        "nucleus_count": nucleus_count,
        "class_counts": dict(sorted(class_counts.items())),
        "out_of_bounds_point_count": out_of_bounds_point_count,
    }


def load_and_validate_geojson(
    path: str | Path,
    *,
    image_width: int | None = None,
    image_height: int | None = None,
    allow_empty: bool = False,
    strict_bounds: bool = True,
) -> tuple[dict[str, Any] | list[Any], dict[str, Any]]:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    summary = validate_geojson(
        payload,
        image_width=image_width,
        image_height=image_height,
        allow_empty=allow_empty,
        strict_bounds=strict_bounds,
    )
    return payload, summary


def write_geojson(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
