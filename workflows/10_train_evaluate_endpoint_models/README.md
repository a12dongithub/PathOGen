# Patient endpoint models and XAI prediction artifacts

This workflow evaluates PAM50 and overall survival at the **patient level** and
then applies the fitted or released models to CPathOGen counterfactual panels.
It deliberately does not turn overall survival into a recurrence label.
`Clinical.tsi` has PAM50, overall-survival duration, and censoring status, but no
recurrence time/event and no MammaPrint Low/High ground truth.

## Protocol

- Frozen encoders: UNI2-h, CTransPath, Virchow2, and ImageNet ResNet50.
- One patient is one statistical sample. L2-normalized tile features are mean
  pooled, then normalized again.
- Five patient-disjoint folds are used for all trained heads.
- PAM50 is four-class (`LumA`, `LumB`, `Basal`, `HER2`) with inverse-frequency
  class weighting. Outputs include one-vs-rest AUC and F1 per class, class
  recall, macro summaries, and overall accuracy.
- Survival remains censor-aware. An L2 Cox probe fitted after train-only scaling
  and PCA reports held-out Harrell C-index. Five- and ten-year AUC/F1/accuracy
  are secondary summaries that exclude patients censored before each horizon.
- Low-tile deterministic flip/rotation views are optional via
  `--minimum-views-per-patient`; the default is one (disabled). Class weighting,
  not tile duplication, is the primary balancing method.
- A counterfactual replaces its source tile inside the patient's baseline bag.
  The source patient's held-out head scores the resulting bag. JSONL stores the
  full prediction and its matched baseline, so later XAI metrics do not require
  rerunning a model.

## Released models

- PathLUPI uses released BRCA survival folds and CONCH 512-dimensional tile
  features.
- OTSurv uses released BRCA folds and original UNI 1024-dimensional features.
  Its released BRCA evaluation config names disease-specific survival, whereas
  this clinical matrix supplies overall survival; the output marks this endpoint
  mismatch rather than presenting it as an exact reproduction.
- CPMP uses original UNI features and outputs MammaPrint risk scores. Since this
  clinical file has no MammaPrint or recurrence labels, predictions are saved
  but AUC/F1/accuracy are explicitly `null` / not evaluable. Its positional
  encoding receives tile-grid coordinates recovered from each `_x#_y#` stem.
- The 512 dataset is a selected tile collection rather than each method's full
  WSI tiling pipeline. Released-model results are therefore named
  **available-tile-bag transfer evaluations**.

## Outputs

```text
endpoint_models/
├── clinical_normalized.csv
├── tile_manifest.csv
├── foundation_performance.{csv,json}
├── all_model_performance_long.csv
├── all_model_performance.json
├── embedding_cache/
└── models/<model>/
    ├── pam50_metrics.json
    ├── pam50_patient_oof_predictions.{csv,parquet}
    ├── survival_metrics.json
    ├── survival_patient_oof_predictions.{csv,parquet}
    └── counterfactual_predictions.jsonl
```

Each JSONL counterfactual record contains model, endpoint, experiment,
condition, dose (when parseable), source tile, patient, held-out fold, status,
prediction, and matched-baseline prediction. CSV/Parquet hold rectangular cohort
data; JSON and JSONL hold metrics, provenance, and nested multiclass/survival
predictions.

## Virchow2 paper-table run

Virchow2 can be run independently after its Hugging Face license has been
accepted. Use `--encoders virchow2` for both foundation-model commands. Add
`--paper-five-only` while scoring variants to retain only the conditions used
by the paper columns: stain brightness, nuclear enlargement, nuclear shape
irregularity, immune burden, and tumor--immune mixing. This avoids embedding
unused signed-dose images.

After scoring, export the two insertion-ready rows with:

```bash
python workflows/10_train_evaluate_endpoint_models/export_virchow2_paper_rows.py \
  --output-root /path/to/endpoint_models_virchow2
```

The command prints only the PAM50 and overall-survival paper columns. It also
writes separate one-row CSV files and a LaTeX fragment under
`models/virchow2/paper_table/`. Performance is macro one-vs-rest AUC for PAM50
and C-index for survival. Intervention cells are `TVD / prediction-flip rate`,
and BNR uses the mean TVD of the four biological interventions divided by the
stain-brightness TVD, matching the existing table.

## Access requirements

UNI2-h, Virchow2, CONCH, and original UNI are gated Hugging Face repositories.
Accept their licenses before the Colab run and expose `HF_TOKEN`. CTransPath is
loaded from its released checkpoint. The Colab cell in the handoff clones
PathLUPI, OTSurv, and CPMP into `/content/external` and downloads the CPMP
checkpoint without modifying those repositories.
