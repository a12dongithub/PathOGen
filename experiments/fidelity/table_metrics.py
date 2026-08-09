"""Reviewer-facing spatial and morphology table statistics."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .constants import CELL_TYPES
from .data import CellObservation
from .spatial import cell_counts, match_coordinates
from .statistics import spearman_with_bootstrap

TYPED_CELL_NAMES = list(CELL_TYPES.values())
TABLE_MORPH_FEATURES = {
    "Nuclear size ρ": "area_mean",
    "Eccentricity ρ": "eccentricity_mean",
    "Solidity ρ": "solidity_mean",
    "Gradient ρ": "grad_mean",
    "Red ρ": "r_mean",
    "Green ρ": "g_mean",
    "Blue ρ": "b_mean",
}


def patient_id(stem: str) -> str:
    return str(stem).split("_", 1)[0]


def _rho(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def _quantile_ci(values: list[float]) -> tuple[float, float]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if len(finite) < 20:
        return float("nan"), float("nan")
    low, high = np.quantile(finite, [0.025, 0.975])
    return float(low), float(high)


def _group_bootstrap_indices(
    groups: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    unique = np.unique(groups)
    sampled = rng.choice(unique, size=len(unique), replace=True)
    return np.concatenate([np.flatnonzero(groups == group) for group in sampled])


def centroid_counts(
    source: list[CellObservation],
    predicted: list[CellObservation],
    radius: float,
    type_aware: bool,
) -> tuple[int, int, int]:
    if type_aware:
        true_positive = sum(
            len(match_coordinates(source, predicted, cell_type, radius).distances)
            for cell_type in TYPED_CELL_NAMES
        )
        source_count = sum(cell.cell_type in TYPED_CELL_NAMES for cell in source)
        predicted_count = sum(cell.cell_type in TYPED_CELL_NAMES for cell in predicted)
    else:
        true_positive = len(
            match_coordinates(source, predicted, "Total", radius).distances
        )
        source_count = len(source)
        predicted_count = len(predicted)
    return true_positive, predicted_count - true_positive, source_count - true_positive


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return float(2 * tp / denominator) if denominator else float("nan")


def summarize_spatial(
    pairs: list[tuple[str, list[CellObservation], list[CellObservation]]],
    evaluator: str,
    typed: bool,
    bootstrap: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not pairs:
        raise ValueError(f"No spatial pairs supplied for {evaluator}")
    case_rows = []
    per_type_rows = []
    for stem, source, predicted in pairs:
        source_counts = cell_counts(source)
        predicted_counts = cell_counts(predicted)
        row: dict[str, Any] = {
            "evaluator": evaluator,
            "stem": stem,
            "patient": patient_id(stem),
            "source_total": source_counts["Total"],
            "predicted_total": predicted_counts["Total"],
        }
        for cell_type in TYPED_CELL_NAMES:
            row[f"source_{cell_type}"] = source_counts[cell_type]
            row[f"predicted_{cell_type}"] = predicted_counts[cell_type]
        for radius in (25.0, 50.0):
            tp, fp, fn = centroid_counts(source, predicted, radius, typed)
            suffix = int(radius)
            row[f"tp_{suffix}"] = tp
            row[f"fp_{suffix}"] = fp
            row[f"fn_{suffix}"] = fn
        case_rows.append(row)
    cases = pd.DataFrame(case_rows)
    groups = cases["patient"].to_numpy(str)
    total = spearman_with_bootstrap(
        cases["source_total"].to_numpy(float),
        cases["predicted_total"].to_numpy(float),
        bootstrap=bootstrap,
        seed=seed,
        groups=groups,
    )

    type_results = []
    if typed:
        for type_index, cell_type in enumerate(TYPED_CELL_NAMES):
            result = spearman_with_bootstrap(
                cases[f"source_{cell_type}"].to_numpy(float),
                cases[f"predicted_{cell_type}"].to_numpy(float),
                bootstrap=bootstrap,
                seed=seed + 10 + type_index,
                groups=groups,
            )
            record = {"evaluator": evaluator, "cell_type": cell_type, **asdict(result)}
            type_results.append(result.rho)
            per_type_rows.append(record)
        macro_type = float(np.nanmean(type_results))
    else:
        macro_type = None

    rng = np.random.default_rng(seed + 100)
    bootstrap_rows = []
    for _ in range(bootstrap):
        index = _group_bootstrap_indices(groups, rng)
        record: dict[str, float] = {
            "total_rho": _rho(
                cases["source_total"].to_numpy(float)[index],
                cases["predicted_total"].to_numpy(float)[index],
            )
        }
        if typed:
            record["per_type_macro_rho"] = float(
                np.nanmean(
                    [
                        _rho(
                            cases[f"source_{name}"].to_numpy(float)[index],
                            cases[f"predicted_{name}"].to_numpy(float)[index],
                        )
                        for name in TYPED_CELL_NAMES
                    ]
                )
            )
        for radius in (25, 50):
            tp = int(cases[f"tp_{radius}"].to_numpy(int)[index].sum())
            fp = int(cases[f"fp_{radius}"].to_numpy(int)[index].sum())
            fn = int(cases[f"fn_{radius}"].to_numpy(int)[index].sum())
            record[f"centroid_f1_{radius}"] = f1_from_counts(tp, fp, fn)
        bootstrap_rows.append(record)

    estimates: dict[str, float | None] = {
        "Total Count ρ": total.rho,
        "Per Type Count ρ": macro_type,
    }
    for radius in (25, 50):
        estimates[f"Centroid F1 @ {radius} px"] = f1_from_counts(
            int(cases[f"tp_{radius}"].sum()),
            int(cases[f"fp_{radius}"].sum()),
            int(cases[f"fn_{radius}"].sum()),
        )
    detail = {
        "evaluator": evaluator,
        "typed_centroid_matching": typed,
        "n_tiles": len(cases),
        "n_patients": int(cases["patient"].nunique()),
    }
    boot = pd.DataFrame(bootstrap_rows)
    for column, value in estimates.items():
        key = {
            "Total Count ρ": "total_rho",
            "Per Type Count ρ": "per_type_macro_rho",
            "Centroid F1 @ 25 px": "centroid_f1_25",
            "Centroid F1 @ 50 px": "centroid_f1_50",
        }[column]
        low, high = (
            _quantile_ci(boot[key].tolist())
            if key in boot
            else (float("nan"), float("nan"))
        )
        detail[f"{key}_estimate"] = value
        detail[f"{key}_ci_low"] = low
        detail[f"{key}_ci_high"] = high
    return (
        {"Method": evaluator, **estimates},
        cases,
        pd.DataFrame(per_type_rows),
        detail,
    )


def summarize_across_morphology(
    measurements: pd.DataFrame,
    row_name: str,
    bootstrap: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    details = []
    row: dict[str, Any] = {"Method": row_name}
    groups = measurements["patient"].to_numpy(str)
    values = []
    for index, (column, feature) in enumerate(TABLE_MORPH_FEATURES.items()):
        result = spearman_with_bootstrap(
            measurements[f"input_{feature}"].to_numpy(float),
            measurements[f"measured_{feature}"].to_numpy(float),
            bootstrap=bootstrap,
            seed=seed + index,
            groups=groups,
        )
        row[column] = result.rho
        values.append(result.rho)
        details.append(
            {
                "row": row_name,
                "design": "across_images",
                "feature": feature,
                **asdict(result),
            }
        )
    row["Macro ρ"] = float(np.nanmean(values))
    return row, details


def summarize_controlled_morphology(
    measurements: pd.DataFrame,
    row_name: str,
    bootstrap: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame]:
    row: dict[str, Any] = {"Method": row_name}
    details = []
    per_source_rows = []
    for feature_index, (column, feature) in enumerate(TABLE_MORPH_FEATURES.items()):
        feature_frame = measurements[measurements["feature"] == feature]
        for stem, group in feature_frame.groupby("stem"):
            rho = _rho(
                group["input_value"].to_numpy(float),
                group["measured_value"].to_numpy(float),
            )
            per_source_rows.append(
                {
                    "row": row_name,
                    "feature": feature,
                    "stem": stem,
                    "patient": patient_id(stem),
                    "rho": rho,
                }
            )
        source_frame = pd.DataFrame(
            [item for item in per_source_rows if item["feature"] == feature]
        )
        finite = source_frame["rho"].to_numpy(float)
        estimate = float(np.nanmedian(finite))
        row[column] = estimate
        rng = np.random.default_rng(seed + feature_index)
        groups = source_frame["patient"].to_numpy(str)
        boot = []
        for _ in range(bootstrap):
            index = _group_bootstrap_indices(groups, rng)
            boot.append(float(np.nanmedian(finite[index])))
        low, high = _quantile_ci(boot)
        details.append(
            {
                "row": row_name,
                "design": "within_image_controlled",
                "feature": feature,
                "n_tiles": len(source_frame),
                "n_patients": int(source_frame["patient"].nunique()),
                "rho": estimate,
                "ci_low": low,
                "ci_high": high,
                "aggregation": "median within-tile Spearman across five dose levels",
            }
        )
    row["Macro ρ"] = float(np.nanmean([row[column] for column in TABLE_MORPH_FEATURES]))
    return row, details, pd.DataFrame(per_source_rows)
