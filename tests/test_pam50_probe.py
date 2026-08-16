import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def load_run():
    path = Path(__file__).parents[1] / "workflows" / "09_train_evaluate_pam50_probe" / "run.py"
    spec = importlib.util.spec_from_file_location("pam50_probe_run", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_pool_patients_means_then_normalizes() -> None:
    module = load_run()
    tiles = pd.DataFrame(
        {
            "patient_id": ["A", "A", "B"],
            "label": ["Basal", "Basal", "LumA"],
            "outer_fold": [0, 0, 1],
        }
    )
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 2.0]], dtype=np.float32)
    patients, pooled = module.pool_patients(tiles, embeddings)
    assert patients["patient_id"].tolist() == ["A", "B"]
    np.testing.assert_allclose(pooled[0], [2**-0.5, 2**-0.5])
    np.testing.assert_allclose(pooled[1], [0.0, 1.0])


def test_counterfactual_scoring_uses_held_out_head_or_ensemble() -> None:
    module = load_run()
    x = np.array([[-2.0], [2.0]])
    labels = np.array([0, 1])
    heads = {}
    for fold, scale in ((0, 1.0), (1, 0.5)):
        head = LogisticRegression(C=1000).fit(x * scale, labels)
        heads[fold] = head
    probe = np.array([[1.0], [1.0]])
    probabilities, methods, folds = module.score_counterfactuals(
        probe, pd.Series(["known", "unknown"]), {"known": 1}, heads
    )
    expected_known = heads[1].predict_proba(probe)[:, 1][0]
    expected_unknown = np.mean([head.predict_proba(probe)[:, 1][1] for head in heads.values()])
    np.testing.assert_allclose(probabilities, [expected_known, expected_unknown])
    assert methods == ["source_patient_held_out_fold", "all_fold_ensemble_source_not_in_binary_cohort"]
    assert folds == ["1", "0|1"]
