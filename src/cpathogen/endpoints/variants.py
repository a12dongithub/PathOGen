"""Normalize CPathOGen experiment manifests into one counterfactual schema."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .clinical import patient_from_tile_stem


def _dose(condition: str) -> float | None:
    lowered = condition.lower()
    if lowered == "baseline" or "baseline" in lowered:
        return 0.0
    match = re.search(r"(minus|plus)_([0-9]+(?:p[0-9]+)?)sd", lowered)
    if match:
        value = float(match.group(2).replace("p", "."))
        return -value if match.group(1) == "minus" else value
    match = re.search(r"(?:^|_)(m|p)([0-9]+(?:p[0-9]+)?)(?:sd)?(?:_|$)", lowered)
    if match:
        value = float(match.group(2).replace("p", "."))
        return -value if match.group(1) == "m" else value
    return None


def _resolve_path(row: pd.Series, manifest_path: Path) -> Path:
    root = manifest_path.parent
    for column in ("local_path", "relative_destination", "image_path"):
        if column not in row or pd.isna(row[column]):
            continue
        value = Path(str(row[column]))
        candidates = [value, root / value]
        normalized_parts = list(value.parts)
        if "images" in normalized_parts:
            index = len(normalized_parts) - 1 - normalized_parts[::-1].index("images")
            candidates.append(root / Path(*normalized_parts[index:]))
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
    stem = str(row.get("stem", ""))
    condition = str(row.get("condition", ""))
    tile_layout = root / stem / f"{condition}.png"
    if tile_layout.is_file():
        return tile_layout.resolve()
    raise FileNotFoundError(
        f"Cannot resolve image for {stem}/{condition} from {manifest_path}"
    )


def normalize_variant_manifests(paths: list[Path]) -> pd.DataFrame:
    rows = []
    seen: set[str] = set()
    seen_variant_ids: set[str] = set()
    for manifest_path in paths:
        manifest_path = manifest_path.expanduser().resolve()
        frame = pd.read_csv(manifest_path)
        required = {"stem", "condition"}
        if not required.issubset(frame):
            raise ValueError(f"{manifest_path} lacks {sorted(required - set(frame))}")
        for _, row in frame.iterrows():
            image_path = _resolve_path(row, manifest_path)
            augmentation_code = int(row.get("augmentation_code", 0))
            canonical = f"{str(image_path).lower()}::augmentation={augmentation_code}"
            if canonical in seen:
                continue
            seen.add(canonical)
            stem = str(row["stem"])
            experiment = str(
                row.get("experiment", row.get("task", manifest_path.parent.name))
            )
            condition = str(row["condition"])
            variant_id = f"{experiment}::{stem}::{condition}"
            if variant_id in seen_variant_ids:
                continue
            seen_variant_ids.add(variant_id)
            rows.append(
                {
                    "variant_id": variant_id,
                    "experiment": experiment,
                    "source_tile_id": stem,
                    "patient_id": patient_from_tile_stem(stem),
                    "condition": condition,
                    "dose_sd": _dose(condition),
                    "seed": row.get("seed", pd.NA),
                    "augmentation_code": augmentation_code,
                    "image_path": str(image_path),
                    "source_manifest": str(manifest_path),
                }
            )
    if not rows:
        raise ValueError("No counterfactual variants found")
    result = pd.DataFrame(rows)
    return result.sort_values(
        ["experiment", "source_tile_id", "condition"], kind="stable"
    ).reset_index(drop=True)


def discover_variant_manifests(root: Path) -> list[Path]:
    root = root.expanduser().resolve()
    preferred = sorted(root.rglob("organized_bucket_images.csv"))
    generated = sorted(root.rglob("images.csv"))
    audit = sorted(root.rglob("audit_manifest.csv"))
    paths = [*preferred, *generated, *audit]
    if not paths:
        raise FileNotFoundError(f"No supported counterfactual manifests under {root}")
    return paths
