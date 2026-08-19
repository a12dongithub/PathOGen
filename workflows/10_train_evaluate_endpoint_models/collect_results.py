#!/usr/bin/env python3
"""Collect endpoint metric JSON files into reviewer-friendly long tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from cpathogen.endpoints.jsonio import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.output_root.expanduser().resolve()
    rows = []
    documents = []
    for path in sorted((root / "models").rglob("*metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        documents.append({"path": str(path.relative_to(root)), "metrics": payload})
        model = payload.get("model", path.parent.name)
        if path.name == "pam50_metrics.json":
            overall = payload["overall_oof"]
            rows.append(
                {
                    "model": model,
                    "endpoint": "PAM50",
                    "scope": "macro/overall",
                    "auc": overall["macro_roc_auc_ovr"],
                    "f1": overall["macro_f1"],
                    "accuracy": overall["overall_accuracy"],
                    "class_accuracy": overall["macro_class_accuracy"],
                    "c_index": None,
                    "support": payload["patients"],
                }
            )
            for class_name, metrics in overall["per_class"].items():
                rows.append(
                    {
                        "model": model,
                        "endpoint": "PAM50",
                        "scope": class_name,
                        "auc": metrics["roc_auc_ovr"],
                        "f1": metrics["f1_ovr"],
                        "accuracy": None,
                        "class_accuracy": metrics["class_accuracy"],
                        "c_index": None,
                        "support": metrics["support"],
                    }
                )
        elif "survival" in path.name:
            rows.append(
                {
                    "model": model,
                    "endpoint": payload.get("endpoint", "overall_survival"),
                    "scope": "global",
                    "auc": None,
                    "f1": None,
                    "accuracy": None,
                    "class_accuracy": None,
                    "c_index": payload.get(
                        "overall_oof_c_index", payload.get("c_index")
                    ),
                    "support": payload.get(
                        "patients", payload.get("patients_evaluable")
                    ),
                }
            )
        elif path.name == "recurrence_risk_metrics.json":
            rows.append(
                {
                    "model": model,
                    "endpoint": payload.get("endpoint"),
                    "scope": "not_evaluable",
                    "auc": payload.get("auc"),
                    "f1": payload.get("f1"),
                    "accuracy": payload.get("accuracy"),
                    "class_accuracy": None,
                    "c_index": None,
                    "support": payload.get("patients_predicted"),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(root / "all_model_performance_long.csv", index=False)
    write_json(
        root / "all_model_performance.json",
        {"schema_version": 1, "metrics_documents": documents},
    )
    print(
        f"Collected {len(documents)} metric files into {len(table)} table rows",
        flush=True,
    )


if __name__ == "__main__":
    main()
