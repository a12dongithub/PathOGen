"""Condition tensors and lazy access to an aligned PathOGen dataset."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor

CELL_TYPE_NAMES = (
    "neoplastic",
    "inflammatory",
    "connective",
    "dead",
    "epithelial",
)

MORPHOLOGY_FEATURE_NAMES = (
    "area_mean",
    "area_var",
    "eccentricity_mean",
    "eccentricity_var",
    "solidity_mean",
    "solidity_var",
    "perimeter_mean",
    "perimeter_var",
    "grad_mean",
    "grad_var",
    "r_mean",
    "r_var",
    "g_mean",
    "g_var",
    "b_mean",
    "b_var",
)


@dataclass(frozen=True)
class ConditionBundle:
    """The two Phase-2 controls for one tile.

    ``spatial`` is normalized to ``[0, 1]`` and has shape ``(5, H, W)``.
    ``morphology`` is the standardized 16-value vector used during training.
    """

    stem: str
    spatial: Tensor
    morphology: Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    def clone(self) -> "ConditionBundle":
        return ConditionBundle(
            stem=self.stem,
            spatial=self.spatial.detach().clone(),
            morphology=self.morphology.detach().clone(),
            metadata=dict(self.metadata),
        )

    def validate(self) -> None:
        if self.spatial.ndim != 3 or self.spatial.shape[0] != len(CELL_TYPE_NAMES):
            raise ValueError(
                "Spatial condition must have shape (5, H, W); found "
                f"{tuple(self.spatial.shape)}"
            )
        if tuple(self.morphology.shape) != (len(MORPHOLOGY_FEATURE_NAMES),):
            raise ValueError(
                "Morphology condition must have shape (16,); found "
                f"{tuple(self.morphology.shape)}"
            )
        if not torch.isfinite(self.spatial).all():
            raise ValueError("Spatial condition contains NaN or infinity")
        if not torch.isfinite(self.morphology).all():
            raise ValueError("Morphology condition contains NaN or infinity")
        minimum = float(self.spatial.min())
        maximum = float(self.spatial.max())
        if minimum < 0.0 or maximum > 1.0:
            raise ValueError(
                f"Spatial condition must be in [0, 1]; found [{minimum}, {maximum}]"
            )


class ConditionStore:
    """Load aligned spatial maps and morphology rows without copying a dataset."""

    def __init__(
        self,
        data_root: str | Path,
        *,
        spatial_maps_dir: str | Path | None = None,
        morphology_table: str | Path | None = None,
        images_dir: str | Path | None = None,
    ) -> None:
        self.data_root = Path(data_root).expanduser().resolve()
        self.spatial_maps_dir = self._resolve(
            spatial_maps_dir, self.data_root / "spatial_maps"
        )
        if morphology_table is None:
            canonical_morphology = (
                self.data_root / "morphology" / "standardized.parquet"
            )
            flat_morphology = self.data_root / "morphology_stats.parquet"
            default_morphology = (
                flat_morphology
                if flat_morphology.is_file()
                else canonical_morphology
            )
        else:
            default_morphology = self.data_root / "morphology" / "standardized.parquet"
        self.morphology_table = self._resolve(
            morphology_table, default_morphology
        )
        self.images_dir = self._resolve(images_dir, self.data_root / "images")

        if not self.spatial_maps_dir.is_dir():
            raise FileNotFoundError(f"Spatial-map directory not found: {self.spatial_maps_dir}")
        if not self.morphology_table.is_file():
            raise FileNotFoundError(f"Morphology table not found: {self.morphology_table}")

        morphology = pd.read_parquet(self.morphology_table)
        missing_columns = [
            name for name in MORPHOLOGY_FEATURE_NAMES if name not in morphology.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Morphology table is missing required columns: {missing_columns}"
            )
        morphology.index = morphology.index.map(str)
        if not morphology.index.is_unique:
            raise ValueError("Morphology table index must contain unique tile stems")
        self._morphology = morphology.loc[:, MORPHOLOGY_FEATURE_NAMES]
        spatial_stems = {path.stem for path in self.spatial_maps_dir.glob("*.npz")}
        self.stems = tuple(sorted(spatial_stems.intersection(self._morphology.index)))
        if not self.stems:
            raise ValueError("No aligned spatial maps and morphology rows were found")
        self._stem_set = set(self.stems)

    def _resolve(self, supplied: str | Path | None, default: Path) -> Path:
        path = Path(supplied).expanduser() if supplied is not None else default
        if not path.is_absolute():
            path = self.data_root / path
        return path.resolve()

    def __len__(self) -> int:
        return len(self.stems)

    def __contains__(self, stem: str) -> bool:
        return stem in self._stem_set

    def load_spatial(self, stem: str) -> Tensor:
        self._require_stem(stem)
        path = self.spatial_maps_dir / f"{stem}.npz"
        with np.load(path, allow_pickle=False) as archive:
            if "map" not in archive:
                raise ValueError(f"Expected key 'map' in {path}")
            values = np.asarray(archive["map"], dtype=np.float32)
        if values.ndim != 3 or values.shape[-1] != len(CELL_TYPE_NAMES):
            raise ValueError(f"Expected HWC five-channel spatial map in {path}")
        if values.max(initial=0.0) > 1.0:
            values = values / 255.0
        return torch.from_numpy(np.ascontiguousarray(values.transpose(2, 0, 1)))

    def load_morphology(self, stem: str) -> Tensor:
        self._require_stem(stem)
        values = self._morphology.loc[stem].to_numpy(dtype=np.float32, copy=True)
        return torch.from_numpy(values)

    def load(self, stem: str) -> ConditionBundle:
        bundle = ConditionBundle(
            stem=stem,
            spatial=self.load_spatial(stem),
            morphology=self.load_morphology(stem),
            metadata={
                "spatial_map": str(self.spatial_maps_dir / f"{stem}.npz"),
                "morphology_table": str(self.morphology_table),
            },
        )
        bundle.validate()
        return bundle

    def image_path(self, stem: str) -> Path | None:
        if not self.images_dir.is_dir():
            return None
        for suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
            path = self.images_dir / f"{stem}{suffix}"
            if path.is_file():
                return path
        return None

    def _require_stem(self, stem: str) -> None:
        if stem not in self._stem_set:
            raise KeyError(f"Tile stem is not aligned in the condition store: {stem}")
