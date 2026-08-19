"""Clinical-matrix and tile-manifest contracts for TCGA-BRCA endpoints."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

PAM50_CLASSES = ("LumA", "LumB", "Basal", "HER2")
_PAM50_ALIASES = {
    "luma": "LumA",
    "lumb": "LumB",
    "basal": "Basal",
    "her2": "HER2",
}


def normalize_patient_id(value: str) -> str:
    """Normalize Xena-style dotted TCGA patient identifiers to hyphens."""
    return str(value).strip().replace(".", "-")


def patient_from_tile_stem(stem: str) -> str | None:
    """Extract a TCGA participant barcode from a CPathOGen tile stem."""
    prefix = stem.split("_", 1)[0]
    parts = prefix.split("-")
    if len(parts) >= 3 and parts[0].upper() == "TCGA":
        return "-".join(parts[:3]).upper()
    return None


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace({"NA": pd.NA, "": pd.NA}), errors="coerce")


def load_clinical_matrix(path: str | Path) -> pd.DataFrame:
    """Load the transposed ``Clinical.tsi`` matrix into one row per patient."""
    path = Path(path).expanduser().resolve()
    matrix = pd.read_csv(path, sep="\t", index_col=0, dtype=str)
    required = {"PAM50", "overall_survival", "status"}
    missing = sorted(required - set(matrix.index))
    if missing:
        raise ValueError(f"Clinical matrix lacks required rows: {missing}")

    clinical = matrix.T.copy()
    clinical.index = clinical.index.map(normalize_patient_id)
    clinical.index.name = "patient_id"
    clinical = clinical.reset_index()
    clinical["pam50"] = (
        clinical["PAM50"]
        .replace({"NA": pd.NA, "": pd.NA})
        .map(
            lambda value: (
                _PAM50_ALIASES.get(str(value).strip().lower())
                if pd.notna(value)
                else pd.NA
            )
        )
    )
    clinical["survival_time_days"] = _numeric(clinical["overall_survival"])
    clinical["survival_event"] = _numeric(clinical["status"])
    valid_event = clinical["survival_event"].isin((0, 1))
    valid_time = clinical["survival_time_days"].gt(0)
    clinical.loc[
        ~(valid_event & valid_time), ["survival_time_days", "survival_event"]
    ] = pd.NA

    columns = [
        "patient_id",
        "pam50",
        "survival_time_days",
        "survival_event",
    ]
    for optional in ("years_to_birth", "pathologic_stage"):
        if optional in clinical:
            columns.append(optional)
    result = clinical[columns].sort_values("patient_id", kind="stable")
    if result["patient_id"].duplicated().any():
        duplicates = result.loc[
            result["patient_id"].duplicated(), "patient_id"
        ].tolist()
        raise ValueError(f"Duplicate normalized clinical patient IDs: {duplicates[:5]}")
    return result.reset_index(drop=True)


def _stable_key(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def build_tile_manifest(
    images_dir: str | Path,
    clinical: pd.DataFrame,
    *,
    max_tiles_per_patient: int,
    seed: int,
) -> pd.DataFrame:
    """Build a deterministic, patient-capped manifest from the real tile folder."""
    if max_tiles_per_patient < 1:
        raise ValueError("max_tiles_per_patient must be positive")
    images_dir = Path(images_dir).expanduser().resolve()
    if not images_dir.is_dir():
        raise FileNotFoundError(images_dir)
    known = set(clinical["patient_id"])
    rows: list[dict[str, object]] = []
    for path in images_dir.rglob("*.png"):
        patient = patient_from_tile_stem(path.stem)
        if patient in known:
            rows.append(
                {
                    "tile_id": path.stem,
                    "patient_id": patient,
                    "image_path": str(path),
                }
            )
    if not rows:
        raise ValueError(f"No TCGA tiles under {images_dir} matched Clinical.tsi")
    frame = pd.DataFrame(rows)
    frame["selection_key"] = frame["tile_id"].map(
        lambda value: _stable_key(value, seed)
    )
    frame = (
        frame.sort_values(["patient_id", "selection_key"], kind="stable")
        .groupby("patient_id", group_keys=False)
        .head(max_tiles_per_patient)
        .drop(columns="selection_key")
        .reset_index(drop=True)
    )
    counts = frame.groupby("patient_id")["tile_id"].transform("size")
    frame["patient_tile_count"] = counts.astype(int)
    return frame.sort_values(["patient_id", "tile_id"], kind="stable").reset_index(
        drop=True
    )
