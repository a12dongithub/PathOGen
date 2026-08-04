"""Cell-count and coordinate-distribution fidelity utilities."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .constants import CELL_NAMES_WITH_TOTAL, CELL_TYPES
from .data import CellObservation
from .statistics import spearman_with_bootstrap


def cell_counts(cells: list[CellObservation]) -> dict[str, int]:
    counts = Counter(cell.cell_type for cell in cells)
    return {
        "Total": len(cells),
        **{name: int(counts.get(name, 0)) for name in CELL_TYPES.values()},
    }


def points_for_type(
    cells: list[CellObservation], cell_type: str
) -> np.ndarray:
    selected = cells if cell_type == "Total" else [c for c in cells if c.cell_type == cell_type]
    if not selected:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray([cell.centroid for cell in selected], dtype=np.float64)


def density_grid(
    cells: list[CellObservation],
    cell_type: str,
    grid_size: int,
    width: int = 512,
    height: int = 512,
) -> np.ndarray:
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    points = points_for_type(cells, cell_type)
    if not len(points):
        return np.zeros((grid_size, grid_size), dtype=np.float64)
    x = np.clip(points[:, 0], 0, width - np.finfo(float).eps)
    y = np.clip(points[:, 1], 0, height - np.finfo(float).eps)
    histogram, _, _ = np.histogram2d(
        y,
        x,
        bins=(grid_size, grid_size),
        range=((0, height), (0, width)),
    )
    return histogram.astype(np.float64)


@dataclass(frozen=True)
class CoordinateMatch:
    source: np.ndarray
    predicted: np.ndarray
    distances: np.ndarray


def match_coordinates(
    source_cells: list[CellObservation],
    predicted_cells: list[CellObservation],
    cell_type: str,
    max_distance: float,
) -> CoordinateMatch:
    source = points_for_type(source_cells, cell_type)
    predicted = points_for_type(predicted_cells, cell_type)
    if not len(source) or not len(predicted):
        empty = np.empty((0, 2), dtype=np.float64)
        return CoordinateMatch(empty, empty.copy(), np.empty(0, dtype=np.float64))
    distances = np.linalg.norm(source[:, None, :] - predicted[None, :, :], axis=2)
    source_index, predicted_index = linear_sum_assignment(distances)
    pair_distances = distances[source_index, predicted_index]
    keep = pair_distances <= max_distance
    return CoordinateMatch(
        source=source[source_index[keep]],
        predicted=predicted[predicted_index[keep]],
        distances=pair_distances[keep],
    )


def coordinate_metrics(
    source_cells: list[CellObservation],
    predicted_cells: list[CellObservation],
    cell_type: str,
    grid_size: int,
    max_distance: float,
    bootstrap: int,
    seed: int,
) -> tuple[dict[str, float], CoordinateMatch, np.ndarray, np.ndarray]:
    source_grid = density_grid(source_cells, cell_type, grid_size)
    predicted_grid = density_grid(predicted_cells, cell_type, grid_size)
    grid_correlation = spearman_with_bootstrap(
        source_grid.ravel(), predicted_grid.ravel(), bootstrap=bootstrap, seed=seed
    )
    match = match_coordinates(source_cells, predicted_cells, cell_type, max_distance)
    x_corr = spearman_with_bootstrap(
        match.source[:, 0] if len(match.source) else np.array([]),
        match.predicted[:, 0] if len(match.predicted) else np.array([]),
        bootstrap=bootstrap,
        seed=seed + 1,
    )
    y_corr = spearman_with_bootstrap(
        match.source[:, 1] if len(match.source) else np.array([]),
        match.predicted[:, 1] if len(match.predicted) else np.array([]),
        bootstrap=bootstrap,
        seed=seed + 2,
    )
    source_count = len(points_for_type(source_cells, cell_type))
    predicted_count = len(points_for_type(predicted_cells, cell_type))
    match_denominator = max(source_count, predicted_count, 1)
    metrics = {
        "source_count": source_count,
        "predicted_count": predicted_count,
        "grid_rho": grid_correlation.rho,
        "grid_p_value": grid_correlation.p_value,
        "grid_ci_low": grid_correlation.ci_low,
        "grid_ci_high": grid_correlation.ci_high,
        "matched_pairs": len(match.distances),
        "matched_fraction": len(match.distances) / match_denominator,
        "median_match_distance": (
            float(np.median(match.distances)) if len(match.distances) else float("nan")
        ),
        "rho_x": x_corr.rho,
        "p_x": x_corr.p_value,
        "rho_y": y_corr.rho,
        "p_y": y_corr.p_value,
    }
    return metrics, match, source_grid, predicted_grid


def validate_cell_type(cell_type: str) -> None:
    if cell_type not in CELL_NAMES_WITH_TOTAL:
        raise KeyError(f"Unknown cell type: {cell_type}")
