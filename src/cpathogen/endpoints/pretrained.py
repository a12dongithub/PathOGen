"""Shared contracts for published slide/bag-model transfer evaluation."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .clinical import load_clinical_matrix, normalize_patient_id
from .encoders import build_encoder, extract_embeddings_sharded, release_encoder
from .metrics import concordance_index


def baseline_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "augmentation_code" in frame:
        frame = frame[frame["augmentation_code"].fillna(0).astype(int) == 0]
    required = {"tile_id", "patient_id", "image_path"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Tile manifest lacks {sorted(missing)}")
    return frame.reset_index(drop=True)


def load_or_extract_features(
    encoder_name: str,
    manifest: pd.DataFrame,
    *,
    cache_dir: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    cache_tag: str = "pretrained_bags",
) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{encoder_name}_{cache_tag}.npz"
    tile_ids = manifest["tile_id"].to_numpy(dtype=str)
    if cache.is_file():
        payload = np.load(cache, allow_pickle=False)
        if np.array_equal(payload["tile_ids"].astype(str), tile_ids):
            print(f"[{encoder_name}] reusing {cache}", flush=True)
            return payload["embeddings"].astype(np.float32, copy=False)
    bundle = build_encoder(encoder_name, device=device)
    try:
        embeddings = extract_embeddings_sharded(
            bundle,
            [Path(value) for value in manifest["image_path"]],
            tile_ids,
            shard_root=cache_dir / "shards",
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            description=f"{encoder_name} published-model features",
        )
    finally:
        release_encoder(bundle)
    temporary = cache.with_suffix(".npz.part")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, tile_ids=tile_ids, embeddings=embeddings)
    temporary.replace(cache)
    return embeddings


def patient_bags(
    manifest: pd.DataFrame, embeddings: np.ndarray
) -> dict[str, np.ndarray]:
    if len(manifest) != len(embeddings):
        raise ValueError("Manifest and embedding rows differ")
    return {
        patient: embeddings[np.asarray(indices)]
        for patient, indices in manifest.groupby(
            "patient_id", sort=True
        ).indices.items()
    }


def patient_coordinates(manifest: pd.DataFrame) -> dict[str, np.ndarray]:
    """Recover CPMP tile-grid coordinates from ``_x#_y#`` tile stems."""
    coordinates: dict[str, np.ndarray] = {}
    for patient, group in manifest.groupby("patient_id", sort=True):
        rows = []
        for tile_id in group["tile_id"].astype(str):
            match = re.search(r"_x(-?\d+)_y(-?\d+)(?:_|$)", tile_id)
            if match is None:
                raise ValueError(
                    f"Cannot parse x/y coordinates from tile ID: {tile_id}"
                )
            rows.append((int(match.group(1)) // 512, int(match.group(2)) // 512))
        coordinates[patient] = np.asarray(rows, dtype=np.float32)
    return coordinates


def replacement_bag(
    manifest: pd.DataFrame,
    bags: dict[str, np.ndarray],
    *,
    patient_id: str,
    source_tile_id: str,
    replacement: np.ndarray,
) -> np.ndarray | None:
    """Replace exactly one source tile while preserving the model's bag contract."""
    if patient_id not in bags:
        return None
    patient_rows = manifest[manifest["patient_id"] == patient_id].reset_index(drop=True)
    matches = np.flatnonzero(
        patient_rows["tile_id"].astype(str).to_numpy() == source_tile_id
    )
    if len(matches) != 1:
        return None
    bag = bags[patient_id].copy()
    bag[int(matches[0])] = replacement
    return bag


def wide_fold_mapping(
    split_csv: Path,
    *,
    patient_column: str,
    held_out_values: tuple[str, ...] = ("val", "test"),
) -> dict[str, int]:
    frame = pd.read_csv(split_csv)
    mapping: dict[str, int] = {}
    for fold in range(5):
        column = f"Fold {fold}"
        if column not in frame:
            continue
        selected = frame[column].astype(str).str.lower().isin(held_out_values)
        for patient in frame.loc[selected, patient_column]:
            normalized = normalize_patient_id(patient)
            if normalized in mapping:
                raise ValueError(
                    f"Patient {normalized} is held out in more than one fold"
                )
            mapping[normalized] = fold
    if not mapping:
        raise ValueError(f"No held-out patients found in {split_csv}")
    return mapping


def survival_metrics_from_predictions(
    predictions: pd.DataFrame,
    clinical_tsi: Path,
    *,
    risk_column: str,
    model: str,
    endpoint_alignment: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    clinical = load_clinical_matrix(clinical_tsi)
    result = predictions.merge(
        clinical, on="patient_id", how="left", validate="one_to_one"
    )
    evaluable = (
        result[["survival_time_days", "survival_event", risk_column]]
        .notna()
        .all(axis=1)
    )
    metrics: dict[str, Any] = {
        "schema_version": 1,
        "model": model,
        "evaluation": "published checkpoint on available patient tile bags",
        "endpoint": "overall_survival",
        "endpoint_alignment": endpoint_alignment,
        "patients_predicted": len(result),
        "patients_evaluable": int(evaluable.sum()),
        "events_evaluable": int(result.loc[evaluable, "survival_event"].sum()),
        "c_index": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "limitation": (
            "The 512 dataset is a sampled tile collection, not each model's original full-WSI "
            "tiling pipeline; this is an available-tile-bag transfer evaluation."
        ),
    }
    if evaluable.sum() >= 2:
        metrics["c_index"] = concordance_index(
            result.loc[evaluable, "survival_time_days"].to_numpy(float),
            result.loc[evaluable, "survival_event"].to_numpy(int),
            result.loc[evaluable, risk_column].to_numpy(float),
        )
    return result, metrics


def checkpoint_state(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in payload and isinstance(payload[key], dict):
                payload = payload[key]
                break
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint does not contain a state dictionary")
    return {str(key).removeprefix("module."): value for key, value in payload.items()}
