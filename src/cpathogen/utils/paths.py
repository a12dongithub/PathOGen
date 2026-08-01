"""Canonical repository-relative paths.

Runtime jobs may override the data and artifact roots with
``CPATHOGEN_DATA_ROOT`` and ``CPATHOGEN_ARTIFACT_ROOT``.
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("CPATHOGEN_DATA_ROOT", PROJECT_ROOT / "data"))
ARTIFACT_ROOT = Path(
    os.environ.get("CPATHOGEN_ARTIFACT_ROOT", PROJECT_ROOT / "artifacts")
)

TCGA_TILES = DATA_ROOT / "interim" / "tiles" / "tcga_brca"
TCGA_GEOJSON = (
    DATA_ROOT / "interim" / "annotations" / "tcga_brca" / "geojson"
)
GENERATOR_ROOT = DATA_ROOT / "processed" / "generator"
SPATIAL_MAPS = GENERATOR_ROOT / "spatial_maps"
MORPHOLOGY_FEATURES = GENERATOR_ROOT / "morphology_features"
GENERATOR_MANIFESTS = GENERATOR_ROOT / "manifests"
