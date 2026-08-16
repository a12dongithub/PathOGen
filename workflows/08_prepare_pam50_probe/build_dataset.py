#!/usr/bin/env python3
"""Build a patient-aware TCGA-BRCA PAM50 Basal-vs-LumA tile manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

CLASSES = ("LumA", "Basal")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--clinical-tsi", type=Path, required=True)
    result.add_argument("--images-dir", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--max-tiles-per-patient", type=int, default=12)
    result.add_argument("--outer-folds", type=int, default=5)
    result.add_argument("--write-images", action="store_true")
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def patient_from_stem(stem: str) -> str | None:
    parts = stem.split("_")[0].split("-")
    if len(parts) >= 3 and parts[0] == "TCGA":
        return "-".join(parts[:3])
    return None


def clinical_labels(path: Path) -> pd.Series:
    matrix = pd.read_csv(path, sep="\t", index_col=0, dtype=str)
    if "PAM50" not in matrix.index:
        raise ValueError("Clinical matrix has no PAM50 row")
    labels = matrix.loc["PAM50"].replace("NA", pd.NA).dropna()
    labels.index = labels.index.str.replace(".", "-", regex=False)
    labels = labels.loc[labels.isin(CLASSES)]
    if set(labels.unique()) != set(CLASSES):
        raise ValueError("Both Basal and LumA labels are required")
    return labels


def stable_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def fold_map(labels: pd.Series, folds: int) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for label in CLASSES:
        patients = sorted(labels.index[labels.eq(label)], key=stable_key)
        mapping.update({patient: index % folds for index, patient in enumerate(patients)})
    return mapping


def main() -> None:
    args = parser().parse_args()
    if args.max_tiles_per_patient < 1:
        raise ValueError("--max-tiles-per-patient must be positive")
    if args.outer_folds < 3:
        raise ValueError("--outer-folds must be at least three")
    labels = clinical_labels(args.clinical_tsi)
    folds = fold_map(labels, args.outer_folds)

    available: dict[str, list[Path]] = {patient: [] for patient in labels.index}
    for path in args.images_dir.glob("*.png"):
        patient = patient_from_stem(path.stem)
        if patient in available:
            available[patient].append(path)

    missing = sorted(patient for patient, paths in available.items() if not paths)
    labels = labels.drop(index=missing)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    image_output = output / "images"
    if args.write_images:
        image_output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for patient in labels.index:
        paths = sorted(available[patient], key=lambda path: stable_key(path.name))
        for path in paths[: args.max_tiles_per_patient]:
            destination = image_output / path.name
            if args.write_images and not destination.is_file():
                shutil.copy2(path, destination)
            rows.append(
                {
                    "tile_id": path.stem,
                    "image_path": str(Path("images") / path.name),
                    "source_image_path": str(path.resolve()),
                    "patient_id": patient,
                    "label": labels[patient],
                    "outer_fold": folds[patient],
                }
            )
    tiles = pd.DataFrame(rows).sort_values(
        ["outer_fold", "label", "patient_id", "tile_id"], kind="stable"
    )
    tiles.to_csv(output / "tiles.csv", index=False)
    patients = (
        tiles[["patient_id", "label", "outer_fold"]]
        .drop_duplicates()
        .sort_values(["outer_fold", "label", "patient_id"], kind="stable")
    )
    patients.to_csv(output / "patients.csv", index=False)
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": "TCGA-BRCA PAM50 Basal vs Luminal A",
        "label_level": "patient",
        "aggregation": "mean tile embeddings within patient",
        "classes": list(CLASSES),
        "positive_class": "Basal",
        "outer_folds": args.outer_folds,
        "max_tiles_per_patient": args.max_tiles_per_patient,
        "clinical_tsi_sha256": sha256(args.clinical_tsi),
        "tile_count": len(tiles),
        "patient_count": len(patients),
        "patients_by_label": patients["label"].value_counts().sort_index().to_dict(),
        "tiles_by_label": tiles["label"].value_counts().sort_index().to_dict(),
        "patients_by_fold_and_label": {
            f"fold_{fold}/{label}": int(count)
            for (fold, label), count in patients.groupby(["outer_fold", "label"]).size().items()
        },
        "patients_without_local_tiles": missing,
        "images_copied": bool(args.write_images),
    }
    (output / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
