"""Resolve and persist the large-file layout used by Colab experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSET_ROOT = REPO_ROOT / "assets"
DEFAULT_CONFIG = DEFAULT_ASSET_ROOT / "runtime_paths.json"
PREFERRED_CELLVIT_MODEL = "CellViT-256-x40-AMP.pth"


def morphology_file(data_dir: Path) -> Path | None:
    candidates = (
        data_dir / "morphology_stats.parquet",
        data_dir / "morphology_features" / "morphology_stats.parquet",
    )
    return next((path for path in candidates if path.is_file()), None)


def valid_dataset(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "images").is_dir()
        and (path / "spatial_maps").is_dir()
        and (path / "geojsons").is_dir()
        and morphology_file(path) is not None
    )


def find_dataset(root: Path) -> Path:
    root = root.expanduser().resolve()
    candidates = [root, root / "512_final_dataset"]
    if root.is_dir():
        candidates.extend(root.rglob("512_final_dataset"))
    for candidate in candidates:
        if valid_dataset(candidate):
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not locate a complete 512_final_dataset under {root}. Expected "
        "images/, spatial_maps/, geojsons/ and morphology_stats.parquet."
    )


def valid_checkpoint(path: Path) -> bool:
    return all(
        required.is_file()
        for required in (
            path / "unet" / "config.json",
            path / "unet" / "diffusion_pytorch_model.safetensors",
            path / "film_mlps.pt",
            path / "spatial_encoder.pt",
        )
    )


def find_checkpoint(root: Path) -> Path:
    root = root.expanduser().resolve()
    candidates = [root]
    if root.is_dir():
        candidates.extend(path for path in root.rglob("checkpoint-*") if path.is_dir())
    for candidate in candidates:
        if valid_checkpoint(candidate):
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not locate a PathOGen checkpoint under {root}. Expected unet weights, "
        "film_mlps.pt and spatial_encoder.pt."
    )


def valid_cellvit_source(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "cellvit" / "models" / "cell_segmentation" / "cellvit.py").is_file()
        and (path / "cellvit" / "models" / "cell_segmentation" / "postprocessing.py").is_file()
    )


def find_cellvit_source(root: Path) -> Path:
    root = root.expanduser().resolve()
    candidates = [root, root / "repository"]
    if root.is_dir():
        candidates.extend(
            package.parent for package in root.rglob("cellvit") if package.is_dir()
        )
    for candidate in candidates:
        if valid_cellvit_source(candidate):
            return candidate.resolve()
    raise FileNotFoundError(f"Could not locate CellViT++ source under {root}")


def find_cellvit_model(root: Path, preferred: str = PREFERRED_CELLVIT_MODEL) -> Path:
    root = root.expanduser().resolve()
    if root.is_file() and root.suffix.lower() == ".pth":
        return root
    preferred_path = root / preferred
    if preferred_path.is_file():
        return preferred_path.resolve()
    candidates = sorted(root.rglob("*.pth")) if root.is_dir() else []
    if not candidates:
        raise FileNotFoundError(f"Could not locate a CellViT++ .pth checkpoint under {root}")
    return candidates[0].resolve()


@dataclass(frozen=True)
class RuntimePaths:
    repo_root: Path
    asset_root: Path
    data_dir: Path
    checkpoint_dir: Path
    cellvit_root: Path
    cellvit_commit: str | None
    cellvit_model: Path | None
    output_root: Path

    def to_payload(self) -> dict[str, str | None]:
        return {
            key: str(value) if value is not None else None
            for key, value in asdict(self).items()
        }

    def write(self, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_payload(), indent=2), encoding="utf-8")

    @classmethod
    def read(cls, source: Path) -> RuntimePaths:
        payload: dict[str, Any] = json.loads(source.read_text(encoding="utf-8"))
        required = {
            "repo_root",
            "asset_root",
            "data_dir",
            "checkpoint_dir",
            "cellvit_root",
            "cellvit_model",
            "output_root",
        }
        missing = sorted(required - payload.keys())
        if missing:
            raise ValueError(f"Runtime path config is missing keys: {missing}")
        return cls(
            repo_root=Path(payload["repo_root"]),
            asset_root=Path(payload["asset_root"]),
            data_dir=Path(payload["data_dir"]),
            checkpoint_dir=Path(payload["checkpoint_dir"]),
            cellvit_root=Path(payload["cellvit_root"]),
            cellvit_commit=payload.get("cellvit_commit"),
            cellvit_model=(
                Path(payload["cellvit_model"]) if payload["cellvit_model"] else None
            ),
            output_root=Path(payload["output_root"]),
        )
