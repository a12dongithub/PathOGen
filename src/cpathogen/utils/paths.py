"""Canonical repository-relative paths for the compact data/model layout."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("CPATHOGEN_DATA_ROOT", PROJECT_ROOT / "data"))
MODEL_ROOT = Path(os.environ.get("CPATHOGEN_MODEL_ROOT", PROJECT_ROOT / "models"))

IMAGES_DIR = DATA_ROOT / "images"
GEOJSON_DIR = DATA_ROOT / "geojsons"
SPATIAL_MAPS = DATA_ROOT / "spatial_maps"
MORPHOLOGY_STATS = DATA_ROOT / "morphology_stats.parquet"
MORPHOLOGY_DIR = DATA_ROOT / "morphology"
CONDITIONS_METADATA = DATA_ROOT / "metadata.jsonl"
EVALUATIONS_ROOT = DATA_ROOT / "evaluations"

# Compatibility aliases for preprocessing code.  They intentionally resolve to
# the compact layout rather than the former interim/processed hierarchy.
TCGA_TILES = IMAGES_DIR
TCGA_GEOJSON = GEOJSON_DIR
CONDITIONS_ROOT = DATA_ROOT
