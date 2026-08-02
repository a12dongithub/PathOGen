# Morphology/stain conditions

The repository currently bundles:

- `standardized.parquet`, the original 38,978-row model-compatible condition
  table used with `artifacts/models/pathogen_phase2/checkpoint_30000`; and
- `feature_manifest.json`, documenting its exact 16-column order and provenance.

The fitted original full-cohort scaler and raw table were not present in the
source dataset and are therefore not represented as if they belonged to this
table.

Workflow 02 can produce a new preprocessing bundle containing:

- `raw.parquet`;
- `standardized.parquet`;
- `scaler.joblib` fitted on training patients only; and
- `feature_manifest.json` with ordered columns and split/config hashes.

Any new scaler must be fitted on training patients only and reused unchanged for
validation, test, and inference data. A scaler fitted only to the six repository
samples is suitable for workflow testing, not for the bundled Phase-2 checkpoint.
