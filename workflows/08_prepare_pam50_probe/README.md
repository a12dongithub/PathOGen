# Workflow 08: prepare the TCGA-BRCA PAM50 probe

The first PAM50 task is **Basal vs Luminal A**. PAM50 is a patient-level
RNA-derived label inherited by each sampled tile, so tile labels are weak rather
than direct morphology annotations. Heads are trained at tile level, while the
five outer folds remain patient-disjoint. Tile probabilities can subsequently
be averaged within patient.

The builder reads the LinkedOmics clinical matrix and the existing real
TCGA-BRCA tile directory. It deterministically caps tiles per patient to bound
GPU and storage cost without changing patient weights.

```bash
uv run --extra probe python workflows/08_prepare_pam50_probe/build_dataset.py \
  --clinical-tsi data/labels/linkedomics_tcga_brca/clinical_firehose_2016_01_28.tsi \
  --images-dir ../refactored/data/images \
  --output-dir artifacts/tcga_brca_pam50_basal_vs_luma_v1 \
  --max-tiles-per-patient 12
```

Add `--write-images` when building the GCS archive. The source matrix is the
LinkedOmics TCGA-BRCA clinical download; it is not committed to Git.

```bash
uv run --extra probe python workflows/08_prepare_pam50_probe/build_dataset.py \
  --clinical-tsi data/labels/linkedomics_tcga_brca/clinical_firehose_2016_01_28.tsi \
  --images-dir ../refactored/data/images \
  --output-dir artifacts/tcga_brca_pam50_basal_vs_luma_v1/dataset \
  --max-tiles-per-patient 12 --write-images \
  --archive artifacts/tcga_brca_pam50_basal_vs_luma_v1/dataset.zip
```
