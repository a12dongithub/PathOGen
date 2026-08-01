# Setup and workflows

## Current reproducibility level

The repository now has one path contract and chronological workflow entry points, but it is not yet a one-command reproduction. The main TCGA-BRCA tiles, CellViT++ implementation/annotations, and generated condition data are missing. TODO entry points fail explicitly rather than pretending those stages exist.

The historical generator environment used Python 3.10, PyTorch/CUDA, Diffusers 0.30.2, Transformers 4.44.2, Accelerate, Pandas/PyArrow, OpenCV, scikit-learn, TorchMetrics, bitsandbytes, and related packages. Downstream experiments additionally use `timm`, `peft`, XGBoost, Matplotlib/Seaborn, UMAP, and Flask. Model dependencies include Stable Diffusion 2.1, UNI2-h, CTransPath, and torchvision ResNet50.

Set `CPATHOGEN_ENV_PREFIX` before using the cloud/overfit shell helpers if the environment is not at `.envs/pathogen` below the project root.

## Canonical inputs

Matching file stems are required across:

```text
data/interim/tiles/tcga_brca/<stem>.png
data/interim/annotations/tcga_brca/geojson/<stem>.geojson
data/processed/generator/spatial_maps/<stem>.npz
data/processed/generator/morphology_features/morphology_standardized.parquet
data/processed/generator/manifests/metadata.jsonl
```

Each NPZ must contain `map` with shape `(512, 512, 5)`. The standardized morphology table must use tile stems as its index and the documented 16-column order. Fit the scaler only on the training split; the script writes raw values, standardized values, `scaler.joblib`, and `feature_manifest.json`, but the caller is responsible for supplying a split-safe training subset.

## Run order

Run commands from the repository root after installing the package (for example `pip install -e .`).

### 1. Tile slides

```bash
python workflows/01_tile_slides/run.py
```

This is currently a TODO because the authoritative WSI tiler is absent. The historical tile-curation UI is under `tools/tile_curator/`.

### 2. Annotate nuclei

```bash
python workflows/02_annotate_nuclei/run.py
```

This is a TODO until CellViT++ and its pinned checkpoint/configuration are added. Expected output is per-tile GeoJSON under `data/interim/annotations/tcga_brca/geojson/`.

### 3. Build generator conditions

The orchestration wrapper is a TODO, but the three implementations can be called directly:

```bash
python -m cpathogen.preprocessing.spatial_maps \
  --geojson-dir data/interim/annotations/tcga_brca/geojson \
  --output-dir data/processed/generator/spatial_maps \
  --n_jobs 32

python -m cpathogen.preprocessing.morphology_features \
  --image-dir data/interim/tiles/tcga_brca \
  --geojson-dir data/interim/annotations/tcga_brca/geojson \
  --raw-output data/processed/generator/morphology_features/morphology_raw.parquet \
  --output data/processed/generator/morphology_features/morphology_standardized.parquet \
  --scaler-output data/processed/generator/morphology_features/scaler.joblib \
  --manifest-output data/processed/generator/morphology_features/feature_manifest.json \
  --n_jobs 32

python -m cpathogen.preprocessing.metadata \
  --tiles-dir data/interim/tiles/tcga_brca \
  --output data/processed/generator/manifests/metadata.jsonl
```

Before training, verify map shape/dtype/key, exact stem intersections, feature order, training-only scaler fit, split manifest, and preprocessing hashes.

### 4. Train phase 1

```bash
python workflows/04_train_phase1/run.py --help
```

The wrapper delegates to `cpathogen.generation.phase1`. The reference configuration uses the constant prompt `"he"` and writes to `artifacts/runs/phase1_domain_adapt/`. Consult `configs/training/phase1_reference.yaml` and the CLI help for the complete arguments.

### 5. Train phase 2

```bash
python workflows/05_train_phase2/run.py --help
```

Phase 2 accepts explicit `--train-tiles-dir`, `--train-spatial-maps-dir`, and `--train-morphology-table` arguments. Outputs belong in `artifacts/runs/phase2_concat_film/`. The architecture is direct concatenation of encoded spatial maps plus FiLM conditioning, not the older ControlNet variant.

The historical cloud helper is `workflows/05_train_phase2/run_cloud.sh`. Review package installation and Accelerate settings before running it in a dedicated environment.

### 6. Sanity check and inference

For a small architecture check:

```bash
bash tests/integration/run_overfit_test.sh
```

It expects a phase-1 checkpoint at `artifacts/runs/phase1_domain_adapt/checkpoints/checkpoint-30000/` and populated TCGA tile/map directories. Treat its small FID estimate as a debugging signal only.

Counterfactual orchestration remains TODO in `workflows/06_generate_counterfactuals/run.py`. Reusable sampling code lives in `cpathogen.generation.inference`; historical checkpoint-specific examples live in that workflow's `misc/` directory.

### 7-9. Embeddings, classifiers, evaluation

The intended sequence is:

```text
workflows/07_extract_embeddings/run.py
  -> workflows/08_train_classifiers/run.py
  -> workflows/09_evaluate_counterfactuals/run.py
```

These are TODO wrappers. Historical implementations are grouped under `experiments/`. Existing classification data is under `data/processed/classification/`, including BACH UNI2 features and combined TCGA benchmark features. Classifier models and other learned weights belong in `artifacts/models/` or an immutable `artifacts/runs/<run-id>/models/` directory.

## Required run record

For every new training or evaluation run, record the code revision/diff, architecture, model revisions, checkpoint hash, dataset/split manifest, preprocessing bundle and scaler, sample stems, seeds, generation settings, library versions, and raw/filtered metrics. Do not use training-data fallback metrics as held-out results.

## Legacy material

Numbered experiments preserve research chronology. Their CPathoGen data/model/output paths were normalized, but they have not been rerun after migration. Vendored paths inside `third_party/nuhtc/` are upstream examples and were intentionally left untouched.
