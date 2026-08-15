#!/usr/bin/env python3
"""Build reproducible input archives for the inflammatory-mass experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selected-candidates",
        type=Path,
        default=REPOSITORY_ROOT / "data/selected_candidates.csv",
    )
    parser.add_argument(
        "--spatial-maps-dir",
        type=Path,
        default=REPOSITORY_ROOT / "data/spatial_maps",
    )
    parser.add_argument(
        "--morphology-table",
        type=Path,
        default=REPOSITORY_ROOT / "data/morphology_stats.parquet",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-count", type=int, default=1000)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_zip(source: Path, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file() and path.name != ".DS_Store":
                archive.write(path, path.relative_to(source))
    temporary.replace(destination)
    destination.with_suffix(destination.suffix + ".sha256").write_text(
        f"{_sha256(destination)}  {destination.name}\n", encoding="utf-8"
    )


def _load_map(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if "map" not in archive:
            raise ValueError(f"Missing 'map' key: {path}")
        values = np.asarray(archive["map"])
    if values.ndim != 3 or values.shape[-1] != len(CELL_TYPE_NAMES):
        raise ValueError(f"Expected HWC five-channel map: {path}")
    return values


def _selected_table(path: Path, spatial_maps_dir: Path, count: int) -> pd.DataFrame:
    table = pd.read_csv(path)
    required = {"stem", "seed", "green_sd", "spatial_score"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Selected-candidate CSV is missing columns: {missing}")
    neutral = table.loc[table["green_sd"].fillna(float("nan")) == 0.0].copy()
    neutral["stem"] = neutral["stem"].astype(str)
    inflammatory_masses: list[float] = []
    for stem in neutral["stem"]:
        spatial_map = _load_map(spatial_maps_dir / f"{stem}.npz")
        inflammatory = spatial_map[:, :, CELL_TYPE_NAMES.index("inflammatory")]
        if inflammatory.max(initial=0.0) > 1.0:
            inflammatory = inflammatory / 255.0
        inflammatory_masses.append(float(inflammatory.sum(dtype=np.float64)))
    neutral["inflammatory_mass_baseline"] = inflammatory_masses
    neutral = neutral.loc[neutral["inflammatory_mass_baseline"] > 0.0]
    neutral = neutral.sort_values(
        ["spatial_score", "stem"], ascending=[False, True], kind="stable"
    ).head(count)
    if len(neutral) != count:
        raise ValueError(f"Needed {count} neutral candidates, found {len(neutral)}")
    if not neutral["stem"].is_unique:
        raise ValueError("Selected candidate stems must be unique")
    neutral.insert(0, "candidate_id", [f"candidate_{i:04d}" for i in range(count)])
    return neutral


def main() -> None:
    args = parse_args()
    if args.candidate_count < 1:
        raise ValueError("--candidate-count must be positive")
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint}")
    required_checkpoint_paths = (
        "unet/config.json",
        "unet/diffusion_pytorch_model.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
        "tokenizer/tokenizer_config.json",
        "text_encoder/config.json",
        "scheduler/scheduler_config.json",
        "spatial_encoder.pt",
        "film_mlps.pt",
    )
    missing_checkpoint = [
        item for item in required_checkpoint_paths if not (checkpoint / item).is_file()
    ]
    if missing_checkpoint:
        raise FileNotFoundError(f"Checkpoint is incomplete: {missing_checkpoint}")

    selected = _selected_table(
        args.selected_candidates, args.spatial_maps_dir, args.candidate_count
    )
    morphology = pd.read_parquet(args.morphology_table)
    morphology.index = morphology.index.map(str)
    missing_features = sorted(
        set(MORPHOLOGY_FEATURE_NAMES).difference(morphology.columns)
    )
    if missing_features:
        raise ValueError(f"Morphology table is missing features: {missing_features}")
    missing_morphology = sorted(set(selected["stem"]).difference(morphology.index))
    if missing_morphology:
        raise KeyError(
            f"Selected stems lack morphology rows: {missing_morphology[:10]}"
        )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_zip = output_dir / "cpathogen_inflammatory_mass_1000_data.zip"
    checkpoint_zip = output_dir / "pathogen_phase2_checkpoint_30000.zip"
    for path in (data_zip, checkpoint_zip):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")

    with tempfile.TemporaryDirectory(prefix="cpathogen-artifacts-") as temporary:
        staging = Path(temporary)
        data_root = staging / "data_bundle" / "data"
        maps_output = data_root / "spatial_maps"
        maps_output.mkdir(parents=True)
        selected.to_csv(data_root / "selected_candidates.csv", index=False)
        morphology.loc[selected["stem"], list(MORPHOLOGY_FEATURE_NAMES)].to_parquet(
            data_root / "morphology_stats.parquet"
        )

        map_hashes: dict[str, str] = {}
        for stem in selected["stem"]:
            source = args.spatial_maps_dir / f"{stem}.npz"
            if not source.is_file():
                raise FileNotFoundError(f"Selected spatial map not found: {source}")
            _load_map(source)
            destination = maps_output / source.name
            shutil.copy2(source, destination)
            map_hashes[source.name] = _sha256(destination)

        metadata = {
            "schema_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "selection": {
                "source": str(args.selected_candidates.expanduser().resolve()),
                "source_sha256": _sha256(args.selected_candidates),
                "rule": (
                    "green_sd == 0 and baseline inflammatory mass > 0; "
                    "spatial_score descending; stem ascending tie-break"
                ),
                "candidate_count": len(selected),
            },
            "spatial_map": {
                "archive_key": "map",
                "layout": "HWC",
                "channel_order": list(CELL_TYPE_NAMES),
            },
            "morphology_feature_order": list(MORPHOLOGY_FEATURE_NAMES),
            "files": {
                "selected_candidates.csv": _sha256(
                    data_root / "selected_candidates.csv"
                ),
                "morphology_stats.parquet": _sha256(
                    data_root / "morphology_stats.parquet"
                ),
                "spatial_maps": map_hashes,
            },
        }
        (data_root / "dataset_manifest.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _write_zip(staging / "data_bundle", data_zip)

        checkpoint_root = staging / "checkpoint_bundle" / "models" / checkpoint.name
        shutil.copytree(
            checkpoint, checkpoint_root, ignore=shutil.ignore_patterns(".DS_Store")
        )
        checkpoint_files = {
            str(path.relative_to(checkpoint_root)): _sha256(path)
            for path in sorted(checkpoint_root.rglob("*"))
            if path.is_file()
        }
        (checkpoint_root / "checkpoint_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "created_at": datetime.now(UTC).isoformat(),
                    "files": checkpoint_files,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        _write_zip(staging / "checkpoint_bundle", checkpoint_zip)

    print(data_zip)
    print(checkpoint_zip)


if __name__ == "__main__":
    main()
