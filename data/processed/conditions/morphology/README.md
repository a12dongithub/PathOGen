# Morphology/stain conditions

Expected outputs:

- `raw.parquet`;
- `standardized.parquet`;
- `scaler.joblib` fitted on training patients only; and
- `feature_manifest.json` with ordered columns and split/config hashes.

The scaler must be fitted on training patients only and reused unchanged for
validation, test, and inference data.
