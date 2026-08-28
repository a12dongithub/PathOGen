#!/usr/bin/env python3
"""Bootstrap reviewer-facing confidence intervals from saved endpoint outputs.

This script performs no model inference. Counterfactual effects are resampled
by patient while retaining every source-tile panel belonging to that patient.
Predictive-performance intervals resample patient-level out-of-fold records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PAM50_CLASSES = ("Basal", "HER2", "LumA", "LumB")
DISPLAY_MODELS = {
    "resnet50": "ResNet-50",
    "ctranspath": "CTransPath",
    "uni2h": "UNI2-h",
    "virchow2": "Virchow2",
    "pathlupi_conch": "PathLUPI + CONCH",
}
MODEL_IDS = {display: model_id for model_id, display in DISPLAY_MODELS.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument(
        "--additional-results-root",
        type=Path,
        action="append",
        default=[],
        help="Optional additional root, for example results_virchow2.",
    )
    parser.add_argument("--table-csv", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--bag-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260828)
    return parser.parse_args()


def stable_rng(seed: int, *parts: str) -> np.random.Generator:
    payload = "\x1f".join((str(seed), *parts)).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return np.random.default_rng(value)


def interval(values: np.ndarray, confidence: float) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return math.nan, math.nan
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(finite, [alpha, 1.0 - alpha])
    return float(low), float(high)


def existing_paths(roots: Iterable[Path], filename: str) -> list[Path]:
    return [root / filename for root in roots if (root / filename).is_file()]


def load_pair_metrics(roots: list[Path]) -> pd.DataFrame:
    names = (
        "pair_metrics_with_pathlupi.parquet",
        "pair_metrics.parquet",
        "rotation_pair_metrics.parquet",
        "pathlupi_rotation_pair_metrics.parquet",
    )
    paths: list[Path] = []
    for name in names:
        paths.extend(existing_paths(roots, name))
    if not paths:
        raise FileNotFoundError("No pair-metric Parquet files were found")
    detail = pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)
    required = {
        "model_id",
        "model",
        "endpoint",
        "bag_size",
        "experiment",
        "display_experiment",
        "family",
        "source_tile_id",
        "patient_id",
        "target_condition",
        "tvd",
        "flip",
    }
    missing = sorted(required - set(detail.columns))
    if missing:
        raise ValueError(f"Pair metrics are missing columns: {missing}")
    detail["bag_key"] = (
        pd.to_numeric(detail["bag_size"], errors="coerce").fillna(-1).astype(int)
    )
    key = [
        "model_id",
        "endpoint",
        "bag_key",
        "experiment",
        "source_tile_id",
        "target_condition",
    ]
    detail = detail.drop_duplicates(key, keep="last").copy()
    detail["tvd"] = pd.to_numeric(detail["tvd"], errors="coerce")
    detail["flip"] = pd.to_numeric(detail["flip"], errors="coerce")
    detail = detail[np.isfinite(detail["tvd"]) & np.isfinite(detail["flip"])]
    detail["patient_id"] = detail["patient_id"].astype("string")
    detail["cluster_id"] = detail["patient_id"].fillna(
        "tile:" + detail["source_tile_id"].astype(str)
    )
    return detail


def tile_panels(detail: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "model_id",
        "model",
        "endpoint",
        "bag_key",
        "experiment",
        "display_experiment",
        "family",
        "source_tile_id",
        "cluster_id",
    ]
    return (
        detail.groupby(keys, as_index=False, dropna=False)[["tvd", "flip"]]
        .mean()
        .sort_values(keys, kind="stable")
    )


def cluster_mean_bootstrap(
    frame: pd.DataFrame,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    grouped = frame.groupby("cluster_id", sort=True)
    sums = grouped[["tvd", "flip"]].sum().to_numpy(float)
    counts = grouped.size().to_numpy(float)
    n_clusters = len(counts)
    weights = rng.multinomial(
        n_clusters,
        np.full(n_clusters, 1.0 / n_clusters),
        size=replicates,
    )
    denominator = weights @ counts
    estimates = (weights @ sums) / denominator[:, None]
    return estimates[:, 0], estimates[:, 1]


def bootstrap_effects(
    panels: pd.DataFrame,
    selected: pd.DataFrame,
    replicates: int,
    confidence: float,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for table_row in selected.itertuples(index=False):
        model_id = MODEL_IDS[str(table_row.Model)]
        endpoint = str(table_row.Task)
        bag_key = -1 if endpoint == "PAM50" else int(table_row.bag_key)
        subset = panels[
            panels["model_id"].eq(model_id)
            & panels["endpoint"].eq(endpoint)
            & panels["bag_key"].eq(bag_key)
        ]
        if subset.empty:
            raise RuntimeError(f"No pair metrics for {model_id} / {endpoint}")
        for experiment, group in subset.groupby("experiment", sort=True):
            rng = stable_rng(seed, model_id, endpoint, str(experiment), "effects")
            tvd_boot, flip_boot = cluster_mean_bootstrap(group, replicates, rng)
            for metric, point, values in (
                ("mean_tvd", float(group["tvd"].mean()), tvd_boot),
                ("flip_rate", float(group["flip"].mean()), flip_boot),
            ):
                low, high = interval(values, confidence)
                rows.append(
                    {
                        "Task": endpoint,
                        "Model": str(table_row.Model),
                        "model_id": model_id,
                        "bag_size": None if bag_key < 0 else bag_key,
                        "metric": metric,
                        "experiment": experiment,
                        "display_experiment": str(group["display_experiment"].iloc[0]),
                        "family": str(group["family"].iloc[0]),
                        "estimate": point,
                        "ci_low": low,
                        "ci_high": high,
                        "tiles": int(group["source_tile_id"].nunique()),
                        "patients": int(group["cluster_id"].nunique()),
                        "valid_replicates": int(np.isfinite(values).sum()),
                    }
                )
    return pd.DataFrame(rows)


def bootstrap_bnr(
    panels: pd.DataFrame,
    selected: pd.DataFrame,
    replicates: int,
    confidence: float,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for table_row in selected.itertuples(index=False):
        model_id = MODEL_IDS[str(table_row.Model)]
        endpoint = str(table_row.Task)
        bag_key = -1 if endpoint == "PAM50" else int(table_row.bag_key)
        subset = panels[
            panels["model_id"].eq(model_id)
            & panels["endpoint"].eq(endpoint)
            & panels["bag_key"].eq(bag_key)
        ]
        metadata = subset[["experiment", "family"]].drop_duplicates()
        biological = metadata.loc[
            metadata["family"].eq("biological"), "experiment"
        ].tolist()
        nuisance = metadata.loc[
            metadata["family"].eq("nuisance"), "experiment"
        ].tolist()
        if len(biological) != 4 or len(nuisance) != 2:
            raise RuntimeError(
                f"{model_id} / {endpoint}: BNR needs four biological and two nuisance experiments; "
                f"found {len(biological)} and {len(nuisance)}"
            )
        experiments = biological + nuisance
        clusters = np.sort(subset["cluster_id"].unique())
        cluster_lookup = {cluster: index for index, cluster in enumerate(clusters)}
        sums = np.zeros((len(experiments), len(clusters)), dtype=float)
        counts = np.zeros_like(sums)
        for row_index, experiment in enumerate(experiments):
            group = subset[subset["experiment"].eq(experiment)]
            aggregated = group.groupby("cluster_id")["tvd"].agg(["sum", "count"])
            indices = np.asarray(
                [cluster_lookup[item] for item in aggregated.index], dtype=int
            )
            sums[row_index, indices] = aggregated["sum"].to_numpy(float)
            counts[row_index, indices] = aggregated["count"].to_numpy(float)
        rng = stable_rng(seed, model_id, endpoint, "bnr")
        weights = rng.multinomial(
            len(clusters),
            np.full(len(clusters), 1.0 / len(clusters)),
            size=replicates,
        )
        denominators = weights @ counts.T
        with np.errstate(divide="ignore", invalid="ignore"):
            experiment_means = (weights @ sums.T) / denominators
            boot = experiment_means[:, : len(biological)].mean(
                axis=1
            ) / experiment_means[:, len(biological) :].mean(axis=1)
        point_means = subset.groupby("experiment")["tvd"].mean()
        point = float(point_means[biological].mean() / point_means[nuisance].mean())
        low, high = interval(boot, confidence)
        rows.append(
            {
                "Task": endpoint,
                "Model": str(table_row.Model),
                "model_id": model_id,
                "bag_size": None if bag_key < 0 else bag_key,
                "metric": "BNR",
                "experiment": "BNR",
                "display_experiment": "BNR",
                "family": "ratio",
                "estimate": point,
                "ci_low": low,
                "ci_high": high,
                "tiles": int(subset["source_tile_id"].nunique()),
                "patients": int(subset["cluster_id"].nunique()),
                "valid_replicates": int(np.isfinite(boot).sum()),
                "ci_excludes_one": bool(low > 1.0 or high < 1.0),
            }
        )
    return pd.DataFrame(rows)


def find_file(roots: list[Path], filename: str) -> Path | None:
    for root in roots:
        candidate = root / filename
        if candidate.is_file():
            return candidate
    return None


def bootstrap_auc(
    frame: pd.DataFrame,
    replicates: int,
    confidence: float,
    rng: np.random.Generator,
) -> tuple[float, float, float, int]:
    probability_columns = [f"probability_{name}" for name in PAM50_CLASSES]
    frame = frame.dropna(subset=["pam50", *probability_columns]).drop_duplicates(
        "patient_id"
    )
    label_map = {name: index for index, name in enumerate(PAM50_CLASSES)}
    labels = frame["pam50"].map(label_map).to_numpy(int)
    probabilities = frame[probability_columns].to_numpy(float)
    point = float(
        roc_auc_score(
            labels,
            probabilities,
            multi_class="ovr",
            average="macro",
            labels=np.arange(len(PAM50_CLASSES)),
        )
    )
    values = []
    n = len(frame)
    for weights in rng.multinomial(n, np.full(n, 1.0 / n), size=replicates):
        class_weight = np.bincount(
            labels, weights=weights, minlength=len(PAM50_CLASSES)
        )
        if np.any(class_weight == 0):
            continue
        values.append(
            roc_auc_score(
                labels,
                probabilities,
                multi_class="ovr",
                average="macro",
                labels=np.arange(len(PAM50_CLASSES)),
                sample_weight=weights,
            )
        )
    low, high = interval(np.asarray(values), confidence)
    return point, low, high, len(values)


def weighted_concordance(
    times: np.ndarray,
    events: np.ndarray,
    risks: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Harrell's C-index with integer bootstrap multiplicities."""
    order = np.argsort(-times, kind="stable")
    sorted_times = times[order]
    risk_rank = np.unique(risks, return_inverse=True)[1]
    rank_count = int(risk_rank.max()) + 1
    tree = np.zeros(rank_count + 1, dtype=float)

    def add(rank: int, value: float) -> None:
        index = rank + 1
        while index <= rank_count:
            tree[index] += value
            index += index & -index

    def prefix(rank: int) -> float:
        if rank < 0:
            return 0.0
        result = 0.0
        index = rank + 1
        while index:
            result += tree[index]
            index -= index & -index
        return result

    numerator = 0.0
    denominator = 0.0
    later_weight = 0.0
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and sorted_times[stop] == sorted_times[start]:
            stop += 1
        group = order[start:stop]
        for index in group:
            weight = float(weights[index])
            if weight <= 0 or not events[index]:
                continue
            rank = int(risk_rank[index])
            less = prefix(rank - 1)
            equal = prefix(rank) - less
            numerator += weight * (less + 0.5 * equal)
            denominator += weight * later_weight
        for index in group:
            weight = float(weights[index])
            if weight > 0:
                add(int(risk_rank[index]), weight)
                later_weight += weight
        start = stop
    return math.nan if denominator <= 0 else numerator / denominator


def bootstrap_cindex(
    frame: pd.DataFrame,
    replicates: int,
    confidence: float,
    rng: np.random.Generator,
) -> tuple[float, float, float, int]:
    columns = ["patient_id", "survival_time_days", "survival_event", "risk_score"]
    frame = frame.dropna(subset=columns).drop_duplicates("patient_id")
    times = frame["survival_time_days"].to_numpy(float)
    events = frame["survival_event"].to_numpy(bool)
    risks = frame["risk_score"].to_numpy(float)
    n = len(frame)
    point = weighted_concordance(times, events, risks, np.ones(n, dtype=int))
    values = np.empty(replicates, dtype=float)
    draws = rng.multinomial(n, np.full(n, 1.0 / n), size=replicates)
    for index, weights in enumerate(draws):
        values[index] = weighted_concordance(times, events, risks, weights)
    low, high = interval(values, confidence)
    return point, low, high, int(np.isfinite(values).sum())


def bootstrap_performance(
    roots: list[Path],
    selected: pd.DataFrame,
    replicates: int,
    confidence: float,
    seed: int,
    bag_size: int,
) -> pd.DataFrame:
    rows = []
    for table_row in selected.itertuples(index=False):
        model = MODEL_IDS[str(table_row.Model)]
        endpoint = str(table_row.Task)
        if endpoint == "PAM50":
            path = find_file(roots, f"{model}_pam50_patient_oof.parquet")
            if path is None:
                raise FileNotFoundError(f"Missing PAM50 OOF predictions for {model}")
            frame = pd.read_parquet(path)
            point, low, high, valid = bootstrap_auc(
                frame,
                replicates,
                confidence,
                stable_rng(seed, model, endpoint, "performance"),
            )
        else:
            filename = (
                f"pathlupi_survival_fixedbag{bag_size}_predictions.parquet"
                if model == "pathlupi_conch"
                else f"{model}_survival_fixedbag{bag_size}_oof.parquet"
            )
            path = find_file(roots, filename)
            if path is None:
                raise FileNotFoundError(f"Missing survival OOF predictions for {model}")
            frame = pd.read_parquet(path)
            point, low, high, valid = bootstrap_cindex(
                frame,
                replicates,
                confidence,
                stable_rng(seed, model, endpoint, "performance"),
            )
        rows.append(
            {
                "Task": endpoint,
                "Model": str(table_row.Model),
                "model_id": model,
                "bag_size": None if endpoint == "PAM50" else bag_size,
                "metric": "Performance",
                "experiment": "Performance",
                "display_experiment": "Performance",
                "family": "performance",
                "estimate": point,
                "ci_low": low,
                "ci_high": high,
                "tiles": math.nan,
                "patients": int(frame["patient_id"].nunique()),
                "valid_replicates": valid,
            }
        )
    return pd.DataFrame(rows)


def format_interval(row: pd.Series) -> str:
    return f"{row.estimate:.4f} [{row.ci_low:.4f}, {row.ci_high:.4f}]"


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines) + "\n"


def build_ci_table(table: pd.DataFrame, ci: pd.DataFrame) -> pd.DataFrame:
    output = table.copy()
    # The source table stores point estimates as numeric values. CI rendering
    # replaces them with strings, so opt into object dtype explicitly.
    for column in output.columns:
        output[column] = output[column].astype(object)
    for row_index, table_row in output.iterrows():
        match = ci[
            ci["Task"].eq(table_row["Task"]) & ci["Model"].eq(table_row["Model"])
        ]
        performance = match[match["metric"].eq("Performance")].iloc[0]
        output.at[row_index, "Performance"] = format_interval(performance)
        for column in (
            "Stain brightness",
            "Image rotation",
            "Nuclear enlargement",
            "Shape irregularity",
            "Immune burden",
            "Tumor-immune mixing",
        ):
            experiment = match[match["display_experiment"].eq(column)]
            tvd = experiment[experiment["metric"].eq("mean_tvd")].iloc[0]
            flip = experiment[experiment["metric"].eq("flip_rate")].iloc[0]
            output.at[row_index, column] = (
                f"{format_interval(tvd)} / {format_interval(flip)}"
            )
        bnr = match[match["metric"].eq("BNR")].iloc[0]
        marker = "*" if bool(bnr.get("ci_excludes_one", False)) else ""
        output.at[row_index, "BNR"] = format_interval(bnr) + marker
    return output


def main() -> None:
    args = parse_args()
    if args.replicates < 200:
        raise ValueError("Use at least 200 bootstrap replicates")
    if not 0.0 < args.confidence < 1.0:
        raise ValueError("--confidence must lie between zero and one")
    primary = args.results_root.resolve()
    roots = [primary, *(path.resolve() for path in args.additional_results_root)]
    output = (args.output_root or primary).resolve()
    output.mkdir(parents=True, exist_ok=True)
    table_path = (
        args.table_csv or primary / "table4_rotation_without_virchow.csv"
    ).resolve()
    table = pd.read_csv(table_path)
    unknown = sorted(set(table["Model"]) - set(MODEL_IDS))
    if unknown:
        raise ValueError(f"Unknown table models: {unknown}")
    selected = table[["Task", "Model"]].copy()
    selected["bag_key"] = np.where(selected["Task"].eq("PAM50"), -1, args.bag_size)

    panels = tile_panels(load_pair_metrics(roots))
    effects = bootstrap_effects(
        panels, selected, args.replicates, args.confidence, args.seed
    )
    bnr = bootstrap_bnr(panels, selected, args.replicates, args.confidence, args.seed)
    performance = bootstrap_performance(
        roots,
        selected,
        args.replicates,
        args.confidence,
        args.seed,
        args.bag_size,
    )
    ci = pd.concat([performance, effects, bnr], ignore_index=True, sort=False)
    ci.to_csv(output / "table4_confidence_intervals_long.csv", index=False)
    bnr.to_csv(output / "table4_bnr_confidence_intervals.csv", index=False)
    formatted = build_ci_table(table, ci)
    formatted.to_csv(
        output / "table4_rotation_without_virchow_with_ci.csv", index=False
    )
    (output / "table4_rotation_without_virchow_with_ci.md").write_text(
        markdown_table(formatted), encoding="utf-8"
    )
    audit = {
        "method": "nonparametric patient-clustered percentile bootstrap",
        "confidence": args.confidence,
        "replicates": args.replicates,
        "seed": args.seed,
        "performance_note": "OOF predictions are resampled without retraining the fitted model",
        "counterfactual_unit": "source-tile panels clustered by patient",
        "table_source": str(table_path),
        "pair_metric_rows": len(panels),
        "all_intervals_finite": bool(
            np.isfinite(ci[["estimate", "ci_low", "ci_high"]].to_numpy(float)).all()
        ),
    }
    (output / "table4_confidence_interval_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(formatted.to_string(index=False))
    print(f"\nSaved confidence intervals to: {output}")


if __name__ == "__main__":
    main()
