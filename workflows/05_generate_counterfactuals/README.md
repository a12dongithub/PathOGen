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
├── images.csv
└── images/
    └── <candidate-id>/
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

## Run a complete dataset

For Workflow 05, the required per-tile controls are the five-channel NPZ map
and the corresponding row in the *standardized* 16-column morphology table.
The source tile is optional for image generation itself, but should be present
so `pairs.jsonl` records its source-image path. This lean branch does not
annotate images or rebuild conditions.

On the other machine, provide this layout (the model weights are intentionally
not stored in Git):

```text
data/images/<stem>.png                                           # recommended
data/spatial_maps/<stem>.npz                                     # required
data/morphology_stats.parquet                                    # required
models/pathogen_phase2/checkpoint_30000/                         # required
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
  --output-dir data/evaluations/tcga_test_relabel_inflammatory_dry_run
```

Then generate all matched pairs on a CUDA machine:

```bash
python workflows/05_generate_counterfactuals/run.py \
  --experiment experiments.spatial.relabel_all_cells \
  --intervention all_cells_inflammatory \
  --all-tiles --seed 42 --steps 30 --spatial-strength 2 --batch-size 1 \
  --device cuda --dtype float16 --local-files-only \
  --output-dir data/evaluations/tcga_test_relabel_inflammatory_seed42
```

The output directory must be new or empty. For `N` aligned tiles, one seed, and
one intervention, this produces `N` baseline images, `N` counterfactual images,
and `N` rows in `pairs.jsonl`.

The defaults are repository-local:

```text
data/spatial_maps/
data/morphology_stats.parquet
data/images/
models/pathogen_phase2/checkpoint_30000/
```

The checkpoint bundles its frozen SD 2.1 tokenizer, text encoder, and scheduler,
so the supported workflow does not require the original external dataset,
checkpoint path, Hugging Face cache, or a network connection.

The real H&E tile is recorded as a reference path in `pairs.jsonl`; it is not an
input to this noise-to-image Phase-2 checkpoint. See `docs/` for the study
protocol and model/data contract.

## Inflammatory-signal mass experiment on Google Cloud

The experiment `experiments.spatial.inflammatory_signal_mass` defines four
prespecified conditions: baseline, +10%, +20%, and +30% inflammatory signal
mass. It changes only spatial channel 1 (`inflammatory`) and preserves the
morphology vector and selected diffusion seed.

Build two portable Zip64 archives locally. The data archive contains exactly
1,000 neutral-green candidates with nonzero baseline inflammatory signal,
selected deterministically by `spatial_score`; their matching five-channel
maps; their 16-value morphology rows; checksums; and the canonical
candidate-to-seed manifest. The model archive contains the complete Phase-2
checkpoint.

```bash
uv run --extra inference python workflows/05_generate_counterfactuals/build_artifacts.py \
  --checkpoint ../refactored/models/pathogen_phase2/checkpoint_30000 \
  --output-dir artifacts/inflammatory_mass_v1
```

On a CUDA VM with this repository cloned and `gcloud` authenticated, run one
shard and upload the resulting PNGs and CSV/JSON manifests. The runner
automatically downloads and verifies each archive's `.sha256` sidecar before
extracting it.

```bash
uv sync --extra inference
uv run python workflows/05_generate_counterfactuals/cloud_run.py \
  --data-uri gs://cpathogen_artifacts/inputs/inflammatory_mass_v1/cpathogen_inflammatory_mass_1000_data.zip \
  --checkpoint-uri gs://cpathogen_artifacts/models/pathogen_phase2_checkpoint_30000.zip \
  --output-uri gs://cpathogen_artifacts/outputs/inflammatory_mass_v1/shard_00 \
  --workspace /mnt/disks/cpathogen --shard-index 0 --num-shards 1
```

For multiple VMs, give each a distinct `--shard-index` in
`0..num-shards-1`, the same `--num-shards`, a distinct workspace disk, and a
distinct output URI. Add `--dry-run` to validate all transformations without
loading the diffusion model.
