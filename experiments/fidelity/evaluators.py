"""Reusable nucleus-evaluator adapters for the paper fidelity tables."""

from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np

from .cellvit import CellViTRunner
from .constants import CELL_COLORS, CELL_TYPES
from .data import CellObservation
from .workflow import load_rgb_with_retry, path_is_file_with_retry

PANNUKE_TYPE_MAP = {
    1: "Neoplastic",
    2: "Inflammatory",
    3: "Connective",
    4: "Dead",
    5: "Epithelial",
}


def batches(values: list, batch_size: int) -> Iterable[list]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def save_observations_geojson(
    cells: list[CellObservation], destination: Path, evaluator: str
) -> None:
    features = []
    for index, cell in enumerate(cells):
        contour = np.asarray(cell.contour, dtype=float)
        if contour.ndim != 2 or contour.shape[0] < 3 or contour.shape[1] != 2:
            continue
        coordinates = contour.round(3).tolist()
        if coordinates[0] != coordinates[-1]:
            coordinates.append(coordinates[0])
        color = CELL_COLORS.get(cell.cell_type, (160, 160, 160))
        properties = {
            "classification": {"name": cell.cell_type, "color": list(color)},
            "centroid": [round(float(value), 3) for value in cell.centroid],
            "evaluator": evaluator,
        }
        if cell.type_probability is not None:
            properties["type_probability"] = round(float(cell.type_probability), 6)
        features.append(
            {
                "type": "Feature",
                "id": str(index),
                "geometry": {"type": "Polygon", "coordinates": [coordinates]},
                "properties": properties,
            }
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}),
        encoding="utf-8",
    )


def cellvit_dict_to_observations(cells: dict) -> list[CellObservation]:
    observations = []
    for cell in cells.values():
        cell_type = CELL_TYPES.get(int(cell["type"]) - 1)
        if cell_type is None:
            continue
        contour = np.asarray(cell["contour"], dtype=np.float32)
        centroid = np.asarray(cell["centroid"], dtype=float).reshape(-1)
        if contour.ndim != 2 or contour.shape[0] < 3 or centroid.size < 2:
            continue
        raw_probability = cell.get("type_prob")
        probability = (
            float(raw_probability)
            if raw_probability is not None and np.isfinite(float(raw_probability))
            else None
        )
        observations.append(
            CellObservation(
                cell_type=cell_type,
                centroid=(float(centroid[0]), float(centroid[1])),
                contour=contour,
                type_probability=probability,
            )
        )
    return observations


def ensure_cellvit_predictions(
    images: dict[str, Path],
    output_dir: Path,
    project_root: Path,
    model_path: Path,
    batch_size: int,
    precision: str,
    existing: dict[str, Path] | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    existing = existing or {}
    outputs: dict[str, Path] = {}
    missing = []
    for artifact_id in images:
        reusable = existing.get(artifact_id)
        destination = output_dir / f"{artifact_id}.geojson"
        if reusable is not None and path_is_file_with_retry(reusable) and not overwrite:
            outputs[artifact_id] = reusable
        elif path_is_file_with_retry(destination) and not overwrite:
            outputs[artifact_id] = destination
        else:
            missing.append(artifact_id)
    if not missing:
        return outputs

    runner = CellViTRunner(project_root, model_path, precision=precision)
    try:
        for group in batches(missing, batch_size):
            loaded = [load_rgb_with_retry(images[artifact_id]) for artifact_id in group]
            predictions = runner.infer_batch(loaded)
            for artifact_id, cells in zip(group, predictions):
                destination = output_dir / f"{artifact_id}.geojson"
                save_observations_geojson(
                    cellvit_dict_to_observations(cells), destination, "CellViT++"
                )
                outputs[artifact_id] = destination
            print(f"[CellViT++] cached {len(outputs)}/{len(images)}", flush=True)
    finally:
        runner.unload()
    return outputs


def _stardist_observations(labels: np.ndarray) -> list[CellObservation]:
    try:
        from skimage.measure import regionprops
    except ImportError as error:
        raise RuntimeError("StarDist conversion requires scikit-image") from error

    observations = []
    for region in regionprops(np.asarray(labels, dtype=np.int32)):
        local_mask = np.asarray(region.image, dtype=np.uint8)
        contours, _ = cv2.findContours(
            local_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
        min_row, min_col, _, _ = region.bbox
        contour[:, 0] += float(min_col)
        contour[:, 1] += float(min_row)
        if len(contour) < 3:
            continue
        centroid_y, centroid_x = region.centroid
        observations.append(
            CellObservation(
                cell_type="Unclassified",
                centroid=(float(centroid_x), float(centroid_y)),
                contour=contour,
            )
        )
    return observations


def ensure_stardist_predictions(
    images: dict[str, Path],
    output_dir: Path,
    model_name: str = "2D_versatile_he",
    overwrite: bool = False,
) -> dict[str, Path]:
    outputs = {}
    missing = []
    for artifact_id in images:
        destination = output_dir / f"{artifact_id}.geojson"
        if path_is_file_with_retry(destination) and not overwrite:
            outputs[artifact_id] = destination
        else:
            missing.append(artifact_id)
    if not missing:
        return outputs

    try:
        from csbdeep.utils import normalize
        from stardist.models import StarDist2D
    except ImportError as error:
        raise RuntimeError(
            "StarDist is missing. In Colab run `pip install stardist csbdeep`."
        ) from error

    model = StarDist2D.from_pretrained(model_name)
    for index, artifact_id in enumerate(missing, start=1):
        image = np.asarray(load_rgb_with_retry(images[artifact_id]), dtype=np.float32)
        labels, _ = model.predict_instances(normalize(image, 1, 99.8, axis=(0, 1)))
        destination = output_dir / f"{artifact_id}.geojson"
        save_observations_geojson(
            _stardist_observations(labels), destination, "StarDist"
        )
        outputs[artifact_id] = destination
        if index % 10 == 0 or index == len(missing):
            print(f"[StarDist] cached {len(outputs)}/{len(images)}", flush=True)
    del model
    gc.collect()
    return outputs


def load_hovernet_json(path: Path) -> list[CellObservation]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    nuclei = payload.get("nuc", payload) if isinstance(payload, dict) else {}
    observations = []
    for cell in nuclei.values():
        if not isinstance(cell, dict):
            continue
        cell_type = PANNUKE_TYPE_MAP.get(int(cell.get("type", -1)))
        if cell_type is None:
            continue
        contour = np.asarray(cell.get("contour", []), dtype=np.float32)
        centroid = np.asarray(cell.get("centroid", []), dtype=float).reshape(-1)
        if contour.ndim != 2 or contour.shape[0] < 3 or centroid.size < 2:
            continue
        probability = cell.get("type_prob")
        observations.append(
            CellObservation(
                cell_type=cell_type,
                centroid=(float(centroid[0]), float(centroid[1])),
                contour=contour,
                type_probability=float(probability)
                if probability is not None
                else None,
            )
        )
    return observations


def _write_hovernet_type_info(path: Path) -> None:
    payload = {"0": ["nolabel", [0, 0, 0]]}
    for type_id, name in PANNUKE_TYPE_MAP.items():
        payload[str(type_id)] = [name, list(CELL_COLORS[name])]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _stage_images(images: dict[str, Path], stage_dir: Path) -> None:
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)
    for artifact_id, source in images.items():
        destination = stage_dir / f"{artifact_id}{source.suffix.lower()}"
        try:
            os.symlink(source.resolve(), destination)
        except OSError:
            shutil.copy2(source, destination)


def _locate_hovernet_raw(root: Path, artifact_id: str) -> Path | None:
    direct = (
        root / "json" / f"{artifact_id}.json",
        root / f"{artifact_id}.json",
    )
    for path in direct:
        if path_is_file_with_retry(path):
            return path
    matches = list(root.rglob(f"{artifact_id}.json")) if root.is_dir() else []
    return matches[0] if matches else None


def ensure_hovernet_predictions(
    images: dict[str, Path],
    output_dir: Path,
    scratch_dir: Path,
    project_root: Path | None,
    model_path: Path | None,
    predictions_dir: Path | None,
    batch_size: int,
    model_mode: str = "fast",
    overwrite: bool = False,
    memory_fraction: float = 0.8,
) -> dict[str, Path]:
    if not 0 < memory_fraction <= 1:
        raise ValueError("HoVer-Net memory_fraction must be in (0, 1]")
    outputs = {}
    missing = []
    for artifact_id in images:
        destination = output_dir / f"{artifact_id}.geojson"
        if path_is_file_with_retry(destination) and not overwrite:
            outputs[artifact_id] = destination
        else:
            missing.append(artifact_id)

    unresolved = []
    if predictions_dir is not None:
        for artifact_id in missing:
            raw = _locate_hovernet_raw(predictions_dir, artifact_id)
            if raw is None:
                unresolved.append(artifact_id)
                continue
            destination = output_dir / f"{artifact_id}.geojson"
            save_observations_geojson(
                load_hovernet_json(raw), destination, "HoVer-Net PanNuke"
            )
            outputs[artifact_id] = destination
    else:
        unresolved = missing

    if not unresolved:
        return outputs
    if project_root is None or model_path is None:
        raise ValueError(
            "Missing HoVer-Net predictions require --hovernet-root and --hovernet-model"
        )
    project_root = project_root.expanduser().resolve()
    model_path = model_path.expanduser().resolve()
    if not (project_root / "run_infer.py").is_file():
        raise FileNotFoundError(
            f"HoVer-Net source missing: {project_root / 'run_infer.py'}"
        )
    if not path_is_file_with_retry(model_path):
        raise FileNotFoundError(f"HoVer-Net checkpoint missing: {model_path}")

    pending_images = {artifact_id: images[artifact_id] for artifact_id in unresolved}
    stage_dir = scratch_dir / "hovernet_input"
    raw_dir = scratch_dir / "hovernet_raw"
    _stage_images(pending_images, stage_dir)
    type_info = scratch_dir / "hovernet_type_info.json"
    _write_hovernet_type_info(type_info)
    compatibility_runner = (
        Path(__file__).resolve().parents[1] / "hovernet_compat_runner.py"
    )
    command = [
        sys.executable,
        str(compatibility_runner),
        str(project_root),
        "--gpu=0",
        "--nr_types=6",
        f"--type_info_path={type_info}",
        f"--model_path={model_path}",
        f"--model_mode={model_mode}",
        "--nr_inference_workers=0",
        "--nr_post_proc_workers=0",
        f"--batch_size={batch_size}",
        "tile",
        f"--input_dir={stage_dir}",
        f"--output_dir={raw_dir}",
        f"--mem_usage={memory_fraction:g}",
    ]
    print(f"[HoVer-Net] processing {len(unresolved)} missing images", flush=True)
    subprocess.run(command, cwd=project_root, check=True)
    for artifact_id in unresolved:
        raw = _locate_hovernet_raw(raw_dir, artifact_id)
        if raw is None:
            raise FileNotFoundError(f"HoVer-Net output missing for {artifact_id}")
        destination = output_dir / f"{artifact_id}.geojson"
        save_observations_geojson(
            load_hovernet_json(raw), destination, "HoVer-Net PanNuke"
        )
        outputs[artifact_id] = destination
    return outputs
