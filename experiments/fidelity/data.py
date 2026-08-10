"""Aligned PathOGen dataset and GeoJSON access."""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .constants import CELL_TYPE_ALIASES, MORPH_FEATURES


@dataclass(frozen=True)
class CellObservation:
    cell_type: str
    centroid: tuple[float, float]
    contour: np.ndarray
    type_probability: float | None = None


@dataclass(frozen=True)
class ConditionSample:
    stem: str
    image_path: Path
    spatial_path: Path
    geojson_path: Path
    morphology: np.ndarray


def morphology_file(data_dir: Path) -> Path:
    candidates = (
        data_dir / "morphology_stats.parquet",
        data_dir / "morphology_features" / "morphology_stats.parquet",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Morphology parquet not found under {data_dir}")


def normalize_cell_name(name: str) -> str | None:
    return CELL_TYPE_ALIASES.get(str(name).strip().lower())


def _feature_iter(payload: object) -> list[dict]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        features = payload.get("features", [])
        if isinstance(features, list):
            return [item for item in features if isinstance(item, dict)]
    return []


def load_cells(path: Path) -> list[CellObservation]:
    last_error: OSError | None = None
    for attempt in range(1, 7):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            break
        except OSError as error:
            last_error = error
            if attempt == 6:
                raise OSError(
                    f"Could not read GeoJSON after 6 attempts: {path}"
                ) from last_error
            delay = min(0.5 * (2 ** (attempt - 1)), 8.0)
            print(
                f"[io] GeoJSON read failed for {path} ({error}); retry "
                f"{attempt + 1}/6 in {delay:.1f}s",
                flush=True,
            )
            time.sleep(delay)
    cells: list[CellObservation] = []
    for feature in _feature_iter(payload):
        properties = feature.get("properties", {})
        classification = properties.get("classification", {})
        cell_type = normalize_cell_name(classification.get("name", ""))
        if cell_type is None:
            continue
        geometry = feature.get("geometry", {})
        geometry_type = geometry.get("type")
        coordinates = geometry.get("coordinates", [])
        polygons = []
        if geometry_type == "Polygon" and coordinates:
            polygons = [coordinates[0]]
        elif geometry_type == "MultiPolygon":
            polygons = [part[0] for part in coordinates if part]
        for polygon in polygons:
            contour = np.asarray(polygon, dtype=np.float32)
            if contour.ndim != 2 or contour.shape[0] < 3 or contour.shape[1] != 2:
                continue
            centroid_value = properties.get("centroid")
            if centroid_value is not None and len(centroid_value) >= 2:
                centroid = (float(centroid_value[0]), float(centroid_value[1]))
            else:
                centroid = (float(contour[:, 0].mean()), float(contour[:, 1].mean()))
            probability = properties.get("type_probability")
            cells.append(
                CellObservation(
                    cell_type=cell_type,
                    centroid=centroid,
                    contour=contour,
                    type_probability=float(probability)
                    if probability is not None
                    else None,
                )
            )
    return cells


class DatasetCatalog:
    """Indexes aligned image, spatial-map, morphology, and source-cell records."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir.resolve()
        self.image_dir = self.data_dir / "images"
        self.spatial_dir = self.data_dir / "spatial_maps"
        self.geojson_dir = self.data_dir / "geojsons"
        for directory in (self.image_dir, self.spatial_dir, self.geojson_dir):
            if not directory.is_dir():
                raise FileNotFoundError(
                    f"Required dataset directory missing: {directory}"
                )

        self.morphology_path = morphology_file(self.data_dir)
        frame = pd.read_parquet(self.morphology_path)
        missing = [name for name in MORPH_FEATURES if name not in frame.columns]
        if missing:
            raise ValueError(f"Morphology parquet is missing columns: {missing}")
        self.morphology = frame[MORPH_FEATURES].copy()
        self.morphology.index = self.morphology.index.astype(str)

        image_by_stem: dict[str, Path] = {}
        for suffix in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
            for path in self.image_dir.glob(suffix):
                image_by_stem.setdefault(path.stem, path)
        self.image_by_stem = image_by_stem
        self.stems = sorted(
            stem
            for stem in self.morphology.index
            if stem in image_by_stem
            and (self.spatial_dir / f"{stem}.npz").is_file()
            and (self.geojson_dir / f"{stem}.geojson").is_file()
        )
        if not self.stems:
            raise RuntimeError(f"No fully aligned cases found under {self.data_dir}")

    def select(
        self, count: int, seed: int, requested: list[str] | None = None
    ) -> list[str]:
        if count < 1:
            raise ValueError("num_images must be positive")
        if requested:
            unknown = sorted(set(requested) - set(self.stems))
            if unknown:
                raise KeyError(f"Unknown or incomplete requested stems: {unknown}")
            if count > len(requested):
                raise ValueError(
                    "num_images exceeds the number of explicitly requested stems"
                )
            return requested[:count]
        rng = random.Random(seed)
        return rng.sample(self.stems, min(count, len(self.stems)))

    def sample(self, stem: str) -> ConditionSample:
        vector = self.morphology.loc[stem].to_numpy(dtype=np.float32)
        if vector.shape != (16,) or not np.isfinite(vector).all():
            raise ValueError(f"Invalid morphology vector for {stem}: {vector.shape}")
        return ConditionSample(
            stem=stem,
            image_path=self.image_by_stem[stem],
            spatial_path=self.spatial_dir / f"{stem}.npz",
            geojson_path=self.geojson_dir / f"{stem}.geojson",
            morphology=vector,
        )

    @staticmethod
    def load_spatial(path: Path) -> np.ndarray:
        with np.load(path) as archive:
            key = "map" if "map" in archive.files else archive.files[0]
            spatial = np.asarray(archive[key])
        if spatial.shape == (5, 512, 512):
            spatial = spatial.transpose(1, 2, 0)
        if spatial.shape != (512, 512, 5):
            raise ValueError(f"Unexpected spatial-map shape: {spatial.shape} ({path})")
        if spatial.dtype != np.uint8:
            if float(spatial.max(initial=0.0)) <= 1.0:
                spatial = spatial * 255.0
            spatial = np.clip(spatial, 0, 255).astype(np.uint8)
        return spatial

    @staticmethod
    def load_image(path: Path) -> Image.Image:
        return Image.open(path).convert("RGB").resize((512, 512))

    def feature_ranges(self, lower: float, upper: float) -> pd.DataFrame:
        if not 0 <= lower < upper <= 1:
            raise ValueError("Range quantiles must satisfy 0 <= lower < upper <= 1")
        quantiles = self.morphology.quantile([lower, 0.5, upper])
        return pd.DataFrame(
            {
                "feature": MORPH_FEATURES,
                "lower_quantile": lower,
                "lower_value": [quantiles.loc[lower, name] for name in MORPH_FEATURES],
                "median_value": [quantiles.loc[0.5, name] for name in MORPH_FEATURES],
                "upper_quantile": upper,
                "upper_value": [quantiles.loc[upper, name] for name in MORPH_FEATURES],
                "observed_min": [
                    self.morphology[name].min() for name in MORPH_FEATURES
                ],
                "observed_max": [
                    self.morphology[name].max() for name in MORPH_FEATURES
                ],
            }
        )

    def increase_feature(
        self,
        vector: np.ndarray,
        feature: str,
        quantile_shift: float,
        lower: float,
        upper: float,
    ) -> tuple[np.ndarray, dict[str, float]]:
        if feature not in MORPH_FEATURES:
            raise KeyError(f"Unknown morphology feature: {feature}")
        if not 0 < quantile_shift < 1:
            raise ValueError("quantile_shift must be between zero and one")
        index = MORPH_FEATURES.index(feature)
        series = self.morphology[feature].dropna().sort_values()
        baseline = float(vector[index])
        percentile = float((series <= baseline).mean())
        target_percentile = min(max(percentile + quantile_shift, lower), upper)
        target = float(series.quantile(target_percentile))
        if target <= baseline:
            larger = series[series > baseline]
            upper_value = float(series.quantile(upper))
            larger = larger[larger <= upper_value]
            if larger.empty:
                raise ValueError(
                    f"Cannot increase {feature} for baseline {baseline:.6g} within q{upper:.3f}"
                )
            target = float(larger.iloc[0])

        changed = np.asarray(vector, dtype=np.float32).copy()
        changed[index] = target
        changed_indices = np.flatnonzero(
            ~np.isclose(changed, vector, rtol=0, atol=1e-7)
        )
        if changed_indices.tolist() != [index]:
            raise AssertionError(
                f"Intervention must change only {feature}; changed indices={changed_indices.tolist()}"
            )
        lower_value = float(series.quantile(lower))
        upper_value = float(series.quantile(upper))
        if not lower_value <= target <= upper_value:
            raise AssertionError(
                f"Target {target} is outside [{lower_value}, {upper_value}] for {feature}"
            )
        return changed, {
            "baseline_value": baseline,
            "target_value": target,
            "input_delta": target - baseline,
            "baseline_percentile": percentile,
            "target_percentile": target_percentile,
            "allowed_lower": lower_value,
            "allowed_upper": upper_value,
        }
