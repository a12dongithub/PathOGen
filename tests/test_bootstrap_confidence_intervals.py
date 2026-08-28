from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = (
    Path(__file__).parents[1]
    / "workflows"
    / "11_tile_local_xai_rotation_virchow2"
    / "bootstrap_confidence_intervals.py"
)
SPEC = importlib.util.spec_from_file_location("bootstrap_confidence_intervals", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def brute_concordance(times, events, risks):
    concordant = 0.0
    comparable = 0
    for first in range(len(times)):
        for second in range(first + 1, len(times)):
            if times[first] == times[second]:
                continue
            if times[first] < times[second] and events[first]:
                earlier, later = first, second
            elif times[second] < times[first] and events[second]:
                earlier, later = second, first
            else:
                continue
            comparable += 1
            concordant += (
                1.0
                if risks[earlier] > risks[later]
                else 0.5
                if risks[earlier] == risks[later]
                else 0.0
            )
    return concordant / comparable


def test_weighted_concordance_matches_explicit_bootstrap_duplicates():
    times = np.asarray([2.0, 4.0, 6.0, 6.0, 9.0])
    events = np.asarray([1, 1, 0, 1, 0], dtype=bool)
    risks = np.asarray([0.9, 0.3, 0.1, 0.3, -0.2])
    weights = np.asarray([2, 0, 3, 1, 2])
    indices = np.repeat(np.arange(len(times)), weights)
    expected = brute_concordance(times[indices], events[indices], risks[indices])
    observed = MODULE.weighted_concordance(times, events, risks, weights)
    assert np.isclose(observed, expected)


def test_bnr_uses_four_biological_and_two_nuisance_experiments():
    experiments = [
        ("bio1", "biological", 0.2),
        ("bio2", "biological", 0.2),
        ("bio3", "biological", 0.2),
        ("bio4", "biological", 0.2),
        ("noise1", "nuisance", 0.1),
        ("noise2", "nuisance", 0.1),
    ]
    rows = []
    for patient in ("P1", "P2", "P3"):
        for experiment, family, value in experiments:
            rows.append(
                {
                    "model_id": "resnet50",
                    "model": "ResNet-50",
                    "endpoint": "PAM50",
                    "bag_key": -1,
                    "experiment": experiment,
                    "display_experiment": experiment,
                    "family": family,
                    "source_tile_id": f"{patient}_tile",
                    "cluster_id": patient,
                    "tvd": value,
                    "flip": 0.0,
                }
            )
    selected = pd.DataFrame([{"Task": "PAM50", "Model": "ResNet-50", "bag_key": -1}])
    result = MODULE.bootstrap_bnr(
        pd.DataFrame(rows), selected, replicates=250, confidence=0.95, seed=42
    )
    assert np.isclose(result.iloc[0].estimate, 2.0)
    assert np.isclose(result.iloc[0].ci_low, 2.0)
    assert np.isclose(result.iloc[0].ci_high, 2.0)
    assert bool(result.iloc[0].ci_excludes_one)
