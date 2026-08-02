# Setup and workflows

## Current reproducibility level

The repository now has one path contract and chronological workflow entry points,
but it is not yet a one-command reproduction. CellViT++ tile annotation,
condition building, and Phase-2 counterfactual generation are implemented.
Downstream workflows 06-08 remain missing; their TODO entry points fail
explicitly rather than pretending those stages exist.

The historical generator environment used Python 3.10, PyTorch/CUDA, Diffusers 0.30.2, Transformers 4.44.2, Accelerate, Pandas/PyArrow, OpenCV, scikit-learn, TorchMetrics, bitsandbytes, and related packages. Downstream experiments additionally use `timm`, `peft`, XGBoost, Matplotlib/Seaborn, UMAP, and Flask. Model dependencies include Stable Diffusion 2.1, UNI2-h, CTransPath, and torchvision ResNet50.

Set `CPATHOGEN_ENV_PREFIX` before using the cloud/overfit shell helpers if the environment is not at `.envs/pathogen` below the project root.

## Canonical inputs

Matching file stems are required across:

```text
data/interim/tiles/tcga_brca/<stem>.png
data/interim/annotations/tcga_brca/geojson/<stem>.geojson
data/processed/conditions/spatial_maps/<stem>.npz
data/processed/conditions/morphology/standardized.parquet
data/processed/conditions/metadata.jsonl
```

Each NPZ must contain `map` with shape `(512, 512, 5)`. The standardized morphology table must use tile stems as its index and the documented 16-column order. Fit the scaler only on the training split; the script writes raw values, standardized values, `scaler.joblib`, and `feature_manifest.json`, but the caller is responsible for supplying a split-safe training subset.

## Run order

Run commands from the repository root. Install CellViT++ tile inference with
`pip install -e ".[annotation]"`, preprocessing with
`pip install -e ".[preprocessing]"`, and counterfactual inference with
`pip install -e ".[inference]"`; generator training still requires the additional
historical PyTorch/Diffusers environment described above.

### 1. Annotate nuclei

```bash
python workflows/01_annotate_nuclei/run.py
```

This runs the pinned CellViT-SAM-H x40 model over prepared tiles in
`data/interim/tiles/tcga_brca/` and writes per-nucleus GeoJSON to
`data/interim/annotations/tcga_brca/geojson/`. Existing outputs are validated
unless `--overwrite` is passed. The default model is
`artifacts/models/cellvit_plus_plus/cellvit_sam_h_x40_amp_001/model.pth`.

Workflow 01 also consumes Workflow 05's matched-pair manifest and annotates the
generated images while preserving the tile, seed, and intervention join keys:

```bash
python workflows/01_annotate_nuclei/run.py \
  --pairs-manifest artifacts/runs/<generation-run>/pairs.jsonl
```

The default paired output is
`artifacts/runs/<generation-run>/cellvit_plus_plus_annotations/`. This stage
produces repeat measurements for counterfactual validation; it does not itself
decide whether an intervention succeeded. See the
[Workflow 01 guide](../../workflows/01_annotate_nuclei/README.md) for the output
tree, GeoJSON contract, devices, and provenance.

### 2. Build conditions

The workflow validates matching tile/GeoJSON stems and creates the complete
condition bundle:

```bash
python workflows/02_build_conditions/run.py \
  --tiles-dir data/interim/tiles/tcga_brca \
  --geojson-dir data/interim/annotations/tcga_brca/geojson \
  --output-dir data/processed/conditions \
  --n-jobs 32
```

By default, unmatched stems are an error. `--allow-unmatched` explicitly limits
processing to the intersection. Before training, verify map shape/dtype/key,
feature order, training-only scaler fit, split manifest, and preprocessing hashes.

### 3. Train phase 1

```bash
python workflows/03_train_phase1/run.py --help
```

The wrapper delegates to `cpathogen.generation.phase1`. The reference configuration uses the constant prompt `"he"` and writes to `artifacts/runs/phase1_domain_adapt/`. Consult `configs/training/phase1_reference.yaml` and the CLI help for the complete arguments.

### 4. Train phase 2

```bash
python workflows/04_train_phase2/run.py --help
```

Phase 2 accepts explicit `--train-tiles-dir`, `--train-spatial-maps-dir`, and `--train-morphology-table` arguments. Outputs belong in `artifacts/runs/phase2_concat_film/`. The architecture is direct concatenation of encoded spatial maps plus FiLM conditioning, not the older ControlNet variant.

The historical cloud helper is `workflows/04_train_phase2/run_cloud.sh`. Review package installation and Accelerate settings before running it in a dedicated environment.

### 5. Sanity check and counterfactual inference

For a small architecture check:

```bash
bash tests/integration/run_overfit_test.sh
```

It expects a phase-1 checkpoint at `artifacts/runs/phase1_domain_adapt/checkpoints/checkpoint-30000/` and populated TCGA tile/map directories. Treat its small FID estimate as a debugging signal only.

List the intervention variants in a Python experiment module:

```bash
python workflows/05_generate_counterfactuals/run.py \
  --experiment experiments.spatial.relabel_all_cells \
  --list-interventions
```

Generate a matched baseline/counterfactual pair without writing modified
condition files:

```bash
python workflows/05_generate_counterfactuals/run.py \
  --experiment experiments.spatial.relabel_all_cells \
  --intervention all_cells_inflammatory \
  --seed 42
```

The workflow writes `run_manifest.json`, `pairs.jsonl`, and generated PNGs under
one immutable output directory. Use `--dry-run` to validate transformations
without loading the model. Historical checkpoint-specific examples remain in
the workflow's `misc/` directory. By default workflow 05 uses the six repository
sample tiles and conditions plus
`artifacts/models/pathogen_phase2/checkpoint_30000/`. That checkpoint also
contains the frozen SD 2.1 tokenizer, text encoder, and scheduler, so local
inference can run offline.

### 6-8. Embeddings, classifiers, evaluation

The intended sequence is:

```text
workflows/06_extract_embeddings/run.py
  -> workflows/07_train_classifiers/run.py
  -> workflows/08_evaluate_counterfactuals/run.py
```

These are TODO wrappers. Historical implementations are preserved under
`archive/experiments/`; reviewed experiments will be promoted into
`experiments/` one at a time. Existing classification data is under
`data/processed/classification/`, including BACH UNI2 features and combined TCGA
benchmark features. Classifier models and other learned weights belong in
`artifacts/models/` or an immutable `artifacts/runs/<run-id>/models/` directory.

## Required run record

For every new training or evaluation run, record the code revision/diff, architecture, model revisions, checkpoint hash, dataset/split manifest, preprocessing bundle and scaler, sample stems, seeds, generation settings, library versions, and raw/filtered metrics. Do not use training-data fallback metrics as held-out results.

## Legacy material

Numbered experiments under `archive/experiments/` preserve research chronology.
Their CPathoGen data/model/output paths were normalized, but they have not been
rerun after migration. Vendored paths inside `third_party/nuhtc/` are upstream
examples and were intentionally left untouched.
