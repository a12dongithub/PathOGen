"""Paper-facing XAI metrics for the five CPathOGen interventions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PAM50_CLASSES = ("Basal", "HER2", "LumA", "LumB")
PAPER_COLUMNS = (
    "Model",
    "Performance",
    "Stain Brightness",
    "Nuclear Enlargement",
    "Nuclear Shape Irregularity",
    "Immune Burden",
    "Tumor-Immune Mixing",
    "BNR",
)


@dataclass(frozen=True)
class PaperExperiment:
    experiment_id: str
    table_column: str
    reference_condition: str
    target_conditions: tuple[str, ...]

    @property
    def retained_conditions(self) -> frozenset[str]:
        return frozenset((self.reference_condition, *self.target_conditions))


PAPER_EXPERIMENTS = (
    PaperExperiment(
        "stain_brightness",
        "Stain Brightness",
        "baseline",
        (
            "stain_brightness_plus_0p5sd",
            "stain_brightness_plus_1p0sd",
            "stain_brightness_plus_1p5sd",
        ),
    ),
    PaperExperiment(
        "nuclear_enlargement",
        "Nuclear Enlargement",
        "baseline",
        (
            "nuclear_enlargement_plus_0p5sd",
            "nuclear_enlargement_plus_1p0sd",
            "nuclear_enlargement_plus_1p5sd",
        ),
    ),
    PaperExperiment(
        "nuclear_shape_irregularity",
        "Nuclear Shape Irregularity",
        "baseline",
        (
            "nuclear_shape_irregularity_plus_0p5sd",
            "nuclear_shape_irregularity_plus_1p0sd",
            "nuclear_shape_irregularity_plus_1p5sd",
        ),
    ),
    PaperExperiment(
        "peritumoral_immune_ring_diameter40px",
        "Immune Burden",
        "peritumoral_ring_plus_80",
        ("peritumoral_ring_plus_160", "peritumoral_ring_plus_320"),
    ),
    PaperExperiment(
        "tumor_immune_separation_diameter40px",
        "Tumor-Immune Mixing",
        "tumor_immune_maximal_mixing",
        (
            "tumor_immune_low_separation",
            "tumor_immune_intermediate",
            "tumor_immune_high_separation",
            "tumor_immune_maximal_segregation",
        ),
    ),
)


def filter_paper_variants(frame: pd.DataFrame) -> pd.DataFrame:
    """Retain only images that contribute to the five paper-table columns."""
    required = {"experiment", "condition"}
    if not required.issubset(frame):
        raise ValueError(f"Variant frame lacks {sorted(required - set(frame))}")
    keep = pd.Series(False, index=frame.index)
    for spec in PAPER_EXPERIMENTS:
        keep |= frame["experiment"].eq(spec.experiment_id) & frame["condition"].isin(
            spec.retained_conditions
        )
    filtered = frame.loc[keep].copy()
    present = set(filtered["experiment"].unique())
    missing = [
        spec.experiment_id
        for spec in PAPER_EXPERIMENTS
        if spec.experiment_id not in present
    ]
    if missing:
        raise ValueError(f"Paper intervention manifests are missing: {missing}")
    return filtered.sort_values(
        ["experiment", "source_tile_id", "condition"], kind="stable"
    ).reset_index(drop=True)


def load_prediction_records(model_dir: Path) -> pd.DataFrame:
    source = model_dir / "counterfactual_predictions.jsonl"
    if not source.is_file():
        raise FileNotFoundError(f"Missing Virchow2 predictions: {source}")
    with source.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records:
        raise ValueError(f"No prediction records in {source}")
    return pd.DataFrame(records)


def _distribution(
    endpoint: str, prediction: dict[str, Any] | None
) -> tuple[np.ndarray, str]:
    if not prediction:
        raise ValueError("Missing prediction")
    if endpoint == "PAM50_four_class":
        probabilities = prediction.get("probabilities", {})
        vector = np.asarray([probabilities.get(name, np.nan) for name in PAM50_CLASSES])
        if (
            not np.all(np.isfinite(vector))
            or np.any(vector < -1e-8)
            or np.any(vector > 1.0 + 1e-8)
            or not np.isclose(vector.sum(), 1.0, atol=1e-5)
        ):
            raise ValueError("Invalid PAM50 probability vector")
        return vector, PAM50_CLASSES[int(vector.argmax())]
    if endpoint == "overall_survival":
        survival = float(prediction["survival_probability_5y"])
        if not 0.0 <= survival <= 1.0:
            raise ValueError("Invalid five-year survival probability")
        return (
            np.asarray((survival, 1.0 - survival)),
            "survive_5y" if survival >= 0.5 else "event_by_5y",
        )
    raise ValueError(f"Unsupported endpoint: {endpoint}")


def _total_variation(first: np.ndarray, second: np.ndarray) -> float:
    return float(0.5 * np.abs(first - second).sum())


def _experiment_metric(
    frame: pd.DataFrame, endpoint: str, spec: PaperExperiment
) -> tuple[float, float]:
    subset = frame[
        frame["endpoint"].eq(endpoint)
        & frame["experiment"].eq(spec.experiment_id)
        & frame["status"].eq("ok")
    ]
    tile_metrics: list[tuple[float, float]] = []
    for source_tile, group in subset.groupby("source_tile_id", sort=False):
        references = group[group["condition"].eq(spec.reference_condition)]
        targets = group[group["condition"].isin(spec.target_conditions)]
        if len(references) != 1 or len(targets) != len(spec.target_conditions):
            continue
        if set(targets["condition"]) != set(spec.target_conditions):
            continue
        seeds = pd.concat((references["seed"], targets["seed"])).dropna().unique()
        if len(seeds) > 1:
            raise ValueError(
                f"Seed mismatch for {spec.experiment_id}/{source_tile}: {seeds.tolist()}"
            )
        reference_vector, reference_decision = _distribution(
            endpoint, references.iloc[0]["prediction"]
        )
        target_tvds: list[float] = []
        target_flips: list[float] = []
        for _, target in targets.iterrows():
            target_vector, target_decision = _distribution(
                endpoint, target["prediction"]
            )
            target_tvds.append(_total_variation(reference_vector, target_vector))
            target_flips.append(float(target_decision != reference_decision))
        tile_metrics.append((float(np.mean(target_tvds)), float(np.mean(target_flips))))
    if not tile_metrics:
        raise RuntimeError(
            f"No complete {spec.experiment_id} panels for endpoint {endpoint}"
        )
    metrics = np.asarray(tile_metrics, dtype=float)
    return float(metrics[:, 0].mean()), float(metrics[:, 1].mean())


def _performance(model_dir: Path, endpoint: str) -> float:
    if endpoint == "PAM50_four_class":
        metrics = json.loads(
            (model_dir / "pam50_metrics.json").read_text(encoding="utf-8")
        )
        return float(metrics["overall_oof"]["macro_roc_auc_ovr"])
    metrics = json.loads(
        (model_dir / "survival_metrics.json").read_text(encoding="utf-8")
    )
    return float(metrics["overall_oof_c_index"])


def build_paper_rows(model_dir: Path) -> dict[str, dict[str, str]]:
    """Build the two display-ready Virchow2 rows used by the paper table."""
    frame = load_prediction_records(model_dir)
    model_ids = set(frame.get("model_id", pd.Series(dtype=str)).dropna().unique())
    if model_ids != {"virchow2"}:
        raise ValueError(f"Expected only virchow2 records, found {sorted(model_ids)}")

    rows: dict[str, dict[str, str]] = {}
    for task, endpoint in (
        ("PAM50 Classification", "PAM50_four_class"),
        ("Overall Survival", "overall_survival"),
    ):
        raw_metrics = {
            spec.table_column: _experiment_metric(frame, endpoint, spec)
            for spec in PAPER_EXPERIMENTS
        }
        stain_tvd = raw_metrics["Stain Brightness"][0]
        biological_tvd = float(
            np.mean(
                [
                    raw_metrics["Nuclear Enlargement"][0],
                    raw_metrics["Nuclear Shape Irregularity"][0],
                    raw_metrics["Immune Burden"][0],
                    raw_metrics["Tumor-Immune Mixing"][0],
                ]
            )
        )
        if stain_tvd <= 0:
            raise ValueError(f"Cannot calculate BNR with stain TVD={stain_tvd}")
        row = {
            "Model": "Virchow2",
            "Performance": f"{_performance(model_dir, endpoint):.4f}",
        }
        for spec in PAPER_EXPERIMENTS:
            tvd, flip = raw_metrics[spec.table_column]
            row[spec.table_column] = f"{tvd:.4f} / {flip:.4f}"
        row["BNR"] = f"{biological_tvd / stain_tvd:.4f}"
        rows[task] = {column: row[column] for column in PAPER_COLUMNS}
    return rows
