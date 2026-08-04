"""Robust Spearman summaries and bootstrap confidence intervals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr


@dataclass(frozen=True)
class CorrelationResult:
    n: int
    rho: float
    p_value: float
    ci_low: float
    ci_high: float


def _finite_pairs(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.shape != y.shape:
        raise ValueError(f"Correlation arrays differ in shape: {x.shape} vs {y.shape}")
    valid = np.isfinite(x) & np.isfinite(y)
    return x[valid], y[valid]


def spearman_with_bootstrap(
    x: np.ndarray,
    y: np.ndarray,
    bootstrap: int = 1000,
    seed: int = 42,
    groups: np.ndarray | None = None,
) -> CorrelationResult:
    raw_x = np.asarray(x, dtype=np.float64).reshape(-1)
    raw_y = np.asarray(y, dtype=np.float64).reshape(-1)
    if raw_x.shape != raw_y.shape:
        raise ValueError(f"Correlation arrays differ in shape: {raw_x.shape} vs {raw_y.shape}")
    valid = np.isfinite(raw_x) & np.isfinite(raw_y)
    x, y = raw_x[valid], raw_y[valid]
    n = len(x)
    if n < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return CorrelationResult(n, float("nan"), float("nan"), float("nan"), float("nan"))
    result = spearmanr(x, y)
    rho = float(result.statistic)
    p_value = float(result.pvalue)
    if bootstrap < 1:
        return CorrelationResult(n, rho, p_value, float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    if groups is None:
        groups = np.arange(n)
    else:
        groups = np.asarray(groups).reshape(-1)
        if len(groups) != len(raw_x):
            raise ValueError("groups must align with the original x/y arrays")
        groups = groups[valid]
    unique_groups = np.unique(groups)
    for _ in range(bootstrap):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == group) for group in sampled_groups])
        if np.unique(x[indices]).size < 2 or np.unique(y[indices]).size < 2:
            continue
        estimate = float(spearmanr(x[indices], y[indices]).statistic)
        if np.isfinite(estimate):
            estimates.append(estimate)
    if len(estimates) < max(20, bootstrap // 10):
        low = high = float("nan")
    else:
        low, high = np.quantile(estimates, [0.025, 0.975]).tolist()
    return CorrelationResult(n, rho, p_value, float(low), float(high))


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=np.float64)
    adjusted = np.full(values.shape, np.nan, dtype=np.float64)
    finite = np.flatnonzero(np.isfinite(values))
    if not len(finite):
        return adjusted.tolist()
    order = finite[np.argsort(values[finite])]
    ranked = values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.clip(ranked, 0, 1)
    return adjusted.tolist()
