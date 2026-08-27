import json
from pathlib import Path

import pandas as pd

from cpathogen.endpoints.paper_xai import (
    PAPER_COLUMNS,
    PAPER_EXPERIMENTS,
    build_paper_rows,
    filter_paper_variants,
)


def _pam50(value: float) -> dict:
    return {
        "probabilities": {
            "Basal": value,
            "HER2": 1.0 - value,
            "LumA": 0.0,
            "LumB": 0.0,
        },
        "predicted_class": "Basal" if value >= 0.5 else "HER2",
    }


def _survival(value: float) -> dict:
    return {
        "log_risk": 0.0,
        "survival_probability_5y": value,
        "survival_probability_10y": value * 0.8,
    }


def test_filter_and_export_virchow2_paper_rows(tmp_path: Path) -> None:
    variant_rows = []
    records = []
    for spec_index, spec in enumerate(PAPER_EXPERIMENTS, start=1):
        conditions = (spec.reference_condition, *spec.target_conditions)
        for condition_index, condition in enumerate(conditions):
            variant_rows.append(
                {
                    "experiment": spec.experiment_id,
                    "condition": condition,
                    "source_tile_id": "tile_001",
                }
            )
            for endpoint in ("PAM50_four_class", "overall_survival"):
                prediction = (
                    _pam50(0.8 - 0.02 * spec_index * condition_index)
                    if endpoint == "PAM50_four_class"
                    else _survival(0.8 - 0.03 * spec_index * condition_index)
                )
                records.append(
                    {
                        "variant_id": f"{spec.experiment_id}::tile_001::{condition}",
                        "experiment": spec.experiment_id,
                        "source_tile_id": "tile_001",
                        "patient_id": "TCGA-AA-0001",
                        "condition": condition,
                        "seed": 17,
                        "model_id": "virchow2",
                        "endpoint": endpoint,
                        "status": "ok",
                        "prediction": prediction,
                        "baseline_prediction": prediction,
                    }
                )
    variant_rows.append(
        {
            "experiment": "nuclear_enlargement",
            "condition": "nuclear_enlargement_minus_2p0sd",
            "source_tile_id": "tile_001",
        }
    )
    filtered = filter_paper_variants(pd.DataFrame(variant_rows))
    assert len(filtered) == sum(
        1 + len(spec.target_conditions) for spec in PAPER_EXPERIMENTS
    )

    model_dir = tmp_path / "models" / "virchow2"
    model_dir.mkdir(parents=True)
    with (model_dir / "counterfactual_predictions.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    (model_dir / "pam50_metrics.json").write_text(
        json.dumps({"overall_oof": {"macro_roc_auc_ovr": 0.87654}}),
        encoding="utf-8",
    )
    (model_dir / "survival_metrics.json").write_text(
        json.dumps({"overall_oof_c_index": 0.65432}), encoding="utf-8"
    )

    rows = build_paper_rows(model_dir)
    assert tuple(rows["PAM50 Classification"]) == PAPER_COLUMNS
    assert rows["PAM50 Classification"]["Performance"] == "0.8765"
    assert rows["Overall Survival"]["Performance"] == "0.6543"
    assert " / " in rows["PAM50 Classification"]["Immune Burden"]
    assert float(rows["PAM50 Classification"]["BNR"]) > 0
