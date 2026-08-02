"""Interfaces for nucleus annotation systems and annotation formats."""

from typing import TYPE_CHECKING, Any

from .geojson import (
    NucleusPrediction,
    PANNUKE_CLASS_NAMES,
    load_and_validate_geojson,
    predictions_to_geojson,
    validate_geojson,
)

if TYPE_CHECKING:
    from .cellvit_adapter import CellViTTileAnnotator


def __getattr__(name: str) -> Any:
    """Keep the GeoJSON contract usable without heavyweight model imports."""
    if name == "CellViTTileAnnotator":
        from .cellvit_adapter import CellViTTileAnnotator

        return CellViTTileAnnotator
    raise AttributeError(name)

__all__ = [
    "CellViTTileAnnotator",
    "NucleusPrediction",
    "PANNUKE_CLASS_NAMES",
    "load_and_validate_geojson",
    "predictions_to_geojson",
    "validate_geojson",
]
