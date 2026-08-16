# Artifact layout

Use task, model, and experiment as separate dimensions. A counterfactual image
dataset is reusable across every classifier and must not live inside a model
run directory.

## Google Cloud Storage

```text
gs://cpathogen_artifacts/
├── inputs/
│   ├── classification_tasks/<task_id>/<version>/
│   │   ├── dataset.zip
│   │   └── dataset.zip.sha256
│   └── counterfactuals/<experiment_id>/<version>/<cohort_id>/
│       ├── generated.zip
│       └── generated.zip.sha256
├── models/
│   ├── encoders/<model_id>/<version>/
│   └── probes/<task_id>/<model_id>/<run_id>/
│       ├── classifier.joblib
│       ├── head_weights.npz
│       ├── metrics.json
│       └── run_manifest.json
└── outputs/
    └── probes/<task_id>/<model_id>/<experiment_id>/<run_id>/
        ├── counterfactual_predictions.csv
        ├── counterfactual_predictions.parquet
        ├── counterfactual_summary.csv
        ├── metrics.json
        ├── run_manifest.json
        └── status.json
```

`run_id` is an immutable UTC timestamp or content-derived identifier. Never
overwrite a completed run in place. Promote a validated run by copying it to a
`latest.json` pointer or recording it in the experiment log.

## Git repository

```text
configs/probes/
├── tasks/          # label source, classes, label and aggregation unit
├── models/         # encoder/checkpoint and head definition
└── experiments/    # knob semantics and four levels
workflows/
├── 05_generate_counterfactuals/
├── 06_prepare_probe_dataset/
├── 07_train_evaluate_probe/
└── 08_prepare_pam50_probe/
docs/
├── artifact-layout.md
├── cpathogen_counterfactual_probe_matrix.md
└── experiment-run-log.md
```

Git contains code, small configuration files, schemas, and documentation. GCS
contains images, embeddings, checkpoints, trained heads, and prediction tables.

## Required prediction-table identity columns

Every prediction row must retain:

- `task`, `model_id`, `candidate_id`, `stem`, `source_patient_id`, and `seed`;
- `condition`, numeric knob value, and intervention parameters;
- one probability column per class, predicted class index, and label;
- matched baseline probability and paired delta;
- source archive URI plus archive member, or a direct image URI;
- run and checkpoint provenance in the adjacent manifest.
