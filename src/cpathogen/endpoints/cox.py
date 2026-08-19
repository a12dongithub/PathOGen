"""Small, dependency-light Cox proportional-hazards probe."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class CoxProbe:
    """Linear Cox probe with train-fitted scaling, PCA, and Breslow baseline."""

    scaler: StandardScaler
    pca: PCA
    coefficient: np.ndarray
    event_times: np.ndarray
    cumulative_hazard: np.ndarray

    def transform(self, features: np.ndarray) -> np.ndarray:
        return self.pca.transform(self.scaler.transform(features))

    def predict_risk(self, features: np.ndarray) -> np.ndarray:
        return self.transform(features) @ self.coefficient

    def predict_survival(
        self, features: np.ndarray, horizons_days: np.ndarray
    ) -> np.ndarray:
        horizons = np.asarray(horizons_days, dtype=float)
        indices = np.searchsorted(self.event_times, horizons, side="right") - 1
        baseline = np.zeros(len(horizons), dtype=float)
        valid = indices >= 0
        baseline[valid] = self.cumulative_hazard[indices[valid]]
        relative_risk = np.exp(np.clip(self.predict_risk(features), -30, 30))
        return np.exp(-np.outer(relative_risk, baseline))


def fit_cox_probe(
    features: np.ndarray,
    event_time: np.ndarray,
    event_observed: np.ndarray,
    *,
    pca_components: int = 64,
    l2_penalty: float = 1.0,
    seed: int = 42,
) -> CoxProbe:
    """Fit a regularized Cox probe using Breslow handling of tied events."""
    features = np.asarray(features, dtype=np.float64)
    time = np.asarray(event_time, dtype=np.float64)
    event = np.asarray(event_observed, dtype=bool)
    if features.ndim != 2 or len(features) != len(time) or len(time) != len(event):
        raise ValueError("Cox inputs have incompatible shapes")
    if not np.isfinite(features).all() or not np.isfinite(time).all():
        raise ValueError("Cox inputs contain non-finite values")
    if event.sum() < 2:
        raise ValueError("At least two observed events are required")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    components = max(1, min(pca_components, len(features) - 1, features.shape[1]))
    pca = PCA(n_components=components, random_state=seed)
    reduced = pca.fit_transform(scaled)
    unique_event_times = np.unique(time[event])
    event_groups = [
        np.flatnonzero(event & (time == value)) for value in unique_event_times
    ]
    risk_groups = [np.flatnonzero(time >= value) for value in unique_event_times]
    event_count = int(event.sum())

    def objective(coefficient: np.ndarray) -> tuple[float, np.ndarray]:
        linear = reduced @ coefficient
        loss = 0.0
        gradient = np.zeros_like(coefficient)
        for event_indices, risk_indices in zip(event_groups, risk_groups, strict=True):
            risk_linear = linear[risk_indices]
            log_denominator = logsumexp(risk_linear)
            weights = np.exp(risk_linear - log_denominator)
            tied_events = len(event_indices)
            loss += tied_events * log_denominator - linear[event_indices].sum()
            gradient += tied_events * (weights @ reduced[risk_indices]) - reduced[
                event_indices
            ].sum(axis=0)
        loss = loss / event_count + 0.5 * l2_penalty * coefficient.dot(coefficient)
        gradient = gradient / event_count + l2_penalty * coefficient
        return float(loss), gradient

    result = minimize(
        objective,
        np.zeros(components, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 1_000, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"Cox optimization failed: {result.message}")
    coefficient = np.asarray(result.x, dtype=np.float64)

    linear = reduced @ coefficient
    increments = []
    for event_indices, risk_indices in zip(event_groups, risk_groups, strict=True):
        denominator = np.exp(np.clip(linear[risk_indices], -30, 30)).sum()
        increments.append(len(event_indices) / denominator)
    cumulative_hazard = np.cumsum(np.asarray(increments, dtype=np.float64))
    return CoxProbe(
        scaler=scaler,
        pca=pca,
        coefficient=coefficient,
        event_times=unique_event_times,
        cumulative_hazard=cumulative_hazard,
    )
