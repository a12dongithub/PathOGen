import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from cpathogen.endpoints.clinical import PAM50_CLASSES, load_clinical_matrix
from cpathogen.endpoints.cox import fit_cox_probe
from cpathogen.endpoints.metrics import concordance_index, multiclass_metrics
from cpathogen.endpoints.pretrained import patient_coordinates
from cpathogen.endpoints.probes import (
    assign_pam50_folds,
    fit_pam50_crossfit,
)
from cpathogen.endpoints.variants import normalize_variant_manifests


def test_clinical_matrix_and_metrics_contract(tmp_path: Path) -> None:
    matrix = pd.DataFrame(
        {
            "TCGA.AA.0001": ["LumA", "100", "1"],
            "TCGA.AA.0002": ["Her2", "200", "0"],
        },
        index=["PAM50", "overall_survival", "status"],
    )
    path = tmp_path / "Clinical.tsi"
    matrix.to_csv(path, sep="\t")
    clinical = load_clinical_matrix(path)
    assert clinical["patient_id"].tolist() == ["TCGA-AA-0001", "TCGA-AA-0002"]
    assert clinical["pam50"].tolist() == ["LumA", "HER2"]

    labels = np.repeat(np.arange(4), 2)
    probabilities = np.full((8, 4), 0.02)
    probabilities[np.arange(8), labels] = 0.94
    metrics = multiclass_metrics(labels, probabilities, PAM50_CLASSES)
    assert metrics["overall_accuracy"] == 1.0
    assert metrics["macro_roc_auc_ovr"] == 1.0


def test_crossfit_and_cox_smoke() -> None:
    rng = np.random.default_rng(7)
    labels = np.repeat(np.arange(4), 10)
    features = np.eye(4)[labels] + rng.normal(0, 0.03, size=(40, 4))
    patients = pd.DataFrame(
        {
            "patient_id": [f"P{i:03d}" for i in range(40)],
            "pam50": [PAM50_CLASSES[index] for index in labels],
        }
    )
    folds = assign_pam50_folds(patients, n_folds=5, seed=42)
    _, probabilities, metrics = fit_pam50_crossfit(
        features, labels, folds, seed=42, c_value=1.0
    )
    assert probabilities.shape == (40, 4)
    assert metrics["overall_oof"]["macro_f1"] > 0.9

    risk_signal = np.linspace(-2, 2, 60)
    survival_features = np.column_stack((risk_signal, risk_signal**2))
    event_time = 1_000 * np.exp(-risk_signal)
    event = np.ones(60, dtype=int)
    probe = fit_cox_probe(
        survival_features,
        event_time,
        event,
        pca_components=2,
        l2_penalty=0.01,
    )
    risk = probe.predict_risk(survival_features)
    assert concordance_index(event_time, event, risk) > 0.9
    assert probe.predict_survival(
        survival_features[:3], np.array([365, 1826])
    ).shape == (
        3,
        2,
    )


def test_variant_manifest_normalization(tmp_path: Path) -> None:
    image = tmp_path / "nuclear_enlargement" / "TCGA-AA-0001_x0_y0_TL" / "baseline.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8), (120, 50, 90)).save(image)
    manifest = tmp_path / "organized_bucket_images.csv"
    pd.DataFrame(
        [
            {
                "experiment": "nuclear_enlargement",
                "stem": "TCGA-AA-0001_x0_y0_TL",
                "condition": "baseline",
                "relative_destination": str(image.relative_to(tmp_path)),
                "seed": 3,
            }
        ]
    ).to_csv(manifest, index=False)
    normalized = normalize_variant_manifests([manifest])
    assert normalized.loc[0, "patient_id"] == "TCGA-AA-0001"
    assert normalized.loc[0, "dose_sd"] == 0.0
    assert Path(normalized.loc[0, "image_path"]).is_file()

    coordinates = patient_coordinates(
        pd.DataFrame(
            {
                "patient_id": ["TCGA-AA-0001"],
                "tile_id": ["TCGA-AA-0001_x1024_y1536_TL"],
            }
        )
    )
    np.testing.assert_array_equal(coordinates["TCGA-AA-0001"], [[2, 3]])


def test_foundation_workflow_end_to_end_smoke(tmp_path: Path, monkeypatch) -> None:
    patients = [f"TCGA-AA-{index:04d}" for index in range(1, 41)]
    labels = [PAM50_CLASSES[index % 4] for index in range(40)]
    matrix = pd.DataFrame(
        {
            patient.replace("-", "."): [
                labels[index],
                str(300 + index * 30),
                str(index % 2),
            ]
            for index, patient in enumerate(patients)
        },
        index=["PAM50", "overall_survival", "status"],
    )
    clinical_path = tmp_path / "Clinical.tsi"
    matrix.to_csv(clinical_path, sep="\t")
    images = tmp_path / "dataset" / "images"
    images.mkdir(parents=True)
    colors = {
        "LumA": (220, 30, 30),
        "LumB": (30, 220, 30),
        "Basal": (30, 30, 220),
        "HER2": (180, 180, 30),
    }
    for patient, label in zip(patients, labels, strict=True):
        Image.new("RGB", (32, 32), colors[label]).save(
            images / f"{patient}_x0_y0_TL.png"
        )
    output = tmp_path / "output"
    script = (
        Path(__file__).parents[1]
        / "workflows"
        / "10_train_evaluate_endpoint_models"
        / "run_foundation.py"
    )
    spec = importlib.util.spec_from_file_location("endpoint_foundation_smoke", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "--clinical-tsi",
            str(clinical_path),
            "--images-dir",
            str(images),
            "--output-root",
            str(output),
            "--encoders",
            "debug_rgb",
            "--batch-size",
            "8",
            "--num-workers",
            "0",
            "--survival-pca-components",
            "2",
        ],
    )
    module.main()
    assert (output / "models" / "debug_rgb" / "pam50_metrics.json").is_file()
    assert (output / "models" / "debug_rgb" / "survival_metrics.json").is_file()
    predictions = pd.read_csv(
        output / "models" / "debug_rgb" / "pam50_patient_oof_predictions.csv"
    )
    assert len(predictions) == 40

    variant_root = tmp_path / "counterfactuals" / "nuclear_enlargement"
    variant_root.mkdir(parents=True)
    variant_image = (
        variant_root / "TCGA-AA-0001_x0_y0_TL" / "nuclear_enlargement_plus_1p0sd.png"
    )
    variant_image.parent.mkdir()
    Image.new("RGB", (32, 32), (255, 10, 10)).save(variant_image)
    pd.DataFrame(
        [
            {
                "candidate_id": "candidate_0000",
                "stem": "TCGA-AA-0001_x0_y0_TL",
                "seed": 1,
                "condition": "nuclear_enlargement_plus_1p0sd",
                "image_path": str(variant_image),
            }
        ]
    ).to_csv(variant_root / "images.csv", index=False)
    score_script = (
        Path(__file__).parents[1]
        / "workflows"
        / "10_train_evaluate_endpoint_models"
        / "score_foundation_variants.py"
    )
    score_spec = importlib.util.spec_from_file_location(
        "endpoint_variant_smoke", score_script
    )
    assert score_spec and score_spec.loader
    score_module = importlib.util.module_from_spec(score_spec)
    sys.modules[score_spec.name] = score_module
    score_spec.loader.exec_module(score_module)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(score_script),
            "--clinical-tsi",
            str(clinical_path),
            "--output-root",
            str(output),
            "--variant-root",
            str(tmp_path / "counterfactuals"),
            "--encoders",
            "debug_rgb",
            "--batch-size",
            "1",
            "--num-workers",
            "0",
        ],
    )
    score_module.main()
    jsonl = output / "models" / "debug_rgb" / "counterfactual_predictions.jsonl"
    assert len(jsonl.read_text(encoding="utf-8").splitlines()) == 2
