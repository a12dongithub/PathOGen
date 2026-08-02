# Workflow 05: generate counterfactuals

This is the stable inference workflow. It loads one frozen Phase-2 checkpoint,
reads each tile's original spatial map and standardized morphology vector, applies
experiment-defined transformations **in memory**, and generates baseline and
counterfactual images from identical initial diffusion noise.

Experiments are Python modules containing `build_interventions()`. They do not
load checkpoints, copy datasets, invoke the denoising loop, or write transformed
conditions. The workflow validates their output and writes:

```text
<output-dir>/
├── run_manifest.json
├── pairs.jsonl
└── images/
    └── <tile-stem>/
        └── seed_<seed>/
            ├── baseline.png
            └── <intervention-slug>.png
```

List an experiment's available variants without loading data or models:

```bash
python workflows/05_generate_counterfactuals/run.py \
  --experiment experiments.spatial.relabel_all_cells \
  --list-interventions
```

Validate the data and transformed tensors without loading the diffusion model:

```bash
python workflows/05_generate_counterfactuals/run.py \
  --experiment experiments.morphology.shape_variance_sweep \
  --dry-run
```

Generate one matched pair:

```bash
python workflows/05_generate_counterfactuals/run.py \
  --experiment experiments.spatial.relabel_all_cells \
  --intervention all_cells_inflammatory \
  --seed 42
```

## Run a complete external test set

For Workflow 05, the required per-tile controls are the five-channel NPZ map
and the corresponding row in the *standardized* 16-column morphology table.
The source tile is optional for image generation itself, but should be present
so `pairs.jsonl` records the source-image path and Workflow 01 can later
re-annotate and compare each result. GeoJSON is not read by this workflow; it
is required only to rebuild conditions with Workflow 02.

On the other machine, provide this layout (the model weights are intentionally
not stored in Git):

```text
data/interim/tiles/tcga_brca/<stem>.png                         # recommended
data/processed/conditions/spatial_maps/<stem>.npz               # required
data/processed/conditions/morphology/standardized.parquet       # required
artifacts/models/pathogen_phase2/checkpoint_30000/              # required
```

The parquet must use the training-time feature order and standardization; copy
`feature_manifest.json` with it for auditability. The map/class order must also
match the Phase-2 contract. Copy the entire checkpoint directory, including
the UNet, VAE, tokenizer, text encoder, scheduler, `spatial_encoder.pt`, and
`film_mlps.pt`.

First validate every selected condition without loading the diffusion model:

```bash
python workflows/05_generate_counterfactuals/run.py \
  --experiment experiments.spatial.relabel_all_cells \
  --intervention all_cells_inflammatory \
  --all-tiles --dry-run \
  --output-dir artifacts/runs/tcga_test_relabel_inflammatory_dry_run
```

Then generate all matched pairs on a CUDA machine:

```bash
python workflows/05_generate_counterfactuals/run.py \
  --experiment experiments.spatial.relabel_all_cells \
  --intervention all_cells_inflammatory \
  --all-tiles --seed 42 --steps 20 --batch-size 1 \
  --device cuda --dtype float16 --local-files-only \
  --output-dir artifacts/runs/tcga_test_relabel_inflammatory_seed42
```

The output directory must be new or empty. For `N` aligned tiles, one seed, and
one intervention, this produces `N` baseline images, `N` counterfactual images,
and `N` rows in `pairs.jsonl`.

The defaults are repository-local:

```text
data/processed/conditions/spatial_maps/
data/processed/conditions/morphology/standardized.parquet
data/interim/tiles/tcga_brca/
artifacts/models/pathogen_phase2/checkpoint_30000/
```

The checkpoint bundles its frozen SD 2.1 tokenizer, text encoder, and scheduler,
so the supported workflow does not require the original external dataset,
checkpoint path, Hugging Face cache, or a network connection.

The real H&E tile is recorded as a reference path in `pairs.jsonl`; it is not an
input to this noise-to-image Phase-2 checkpoint. Historical one-off scripts in
`misc/` remain reference material and are not the supported interface.
