# Workflow 07: train and evaluate a pathology probe

This workflow freezes the released CTransPath encoder, extracts L2-normalized
embeddings from real BCSS/TIGER tiles, selects an L2-logistic head on the
patient-disjoint validation split, refits that head on train plus validation,
and evaluates it once on the held-out real test split. The fixed model then
scores all 1,200 generated images from the 300-candidate inflammatory-centroid
density experiment.

The 44 `probe_holdout` patients are never used for head fitting, model
selection, or the conventional real-image test. Counterfactual images are used
only after the head is fixed.

## One-command VM run

After cloning the repository on an authenticated CUDA VM:

```bash
uv sync --extra probe
uv run python workflows/07_train_evaluate_probe/cloud_run.py
```

The defaults point to:

- `gs://cpathogen_artifacts/inputs/bcss_tumor_stroma_v1/bcss_tumor_stroma_v1.zip`
- `gs://cpathogen_artifacts/models/ctranspath/ctranspath.pth`
- `gs://cpathogen_artifacts/outputs/inflammatory_centroid_density_sd_v1_20260815-1508`

Supply `--output-uri` when a stable run name is preferred. Otherwise a UTC
timestamp is appended automatically. Add `--dry-run` to download, verify, and
count every input without loading the model.

## Uploaded outputs

- `classifier.joblib`: fitted scikit-learn head and class metadata
- `head_weights.npz`: portable coefficient, intercept, and class arrays
- `metrics.json`: validation selection and held-out real-tile metrics
- `counterfactual_predictions.csv`: one row per generated tile and knob level
- `counterfactual_predictions.parquet`: typed copy of the prediction table
- `counterfactual_summary.csv`: simple condition-level probability summary
- `real_tile_embeddings.npz`: cached embeddings and split/label arrays
- `run_manifest.json`: hashes and provenance for every output
- `status.json`: current/completed cloud-run status

`counterfactual_predictions.csv` retains candidate, seed, condition, source
stem, and intervention parameters. It adds both class probabilities, predicted
label, knob value in SD units, source patient barcode when available, matched
baseline probability, within-candidate probability change, relative image path,
and the durable GCS URI of every scored image.

The CTransPath architecture and checkpoint are from the official
[TransPath repository](https://github.com/Xiyue-Wang/TransPath) and are limited
to the license and research-use conditions stated there.
