# PathOGen experiments

Experiment code lives here; reusable model/training components live in `training/`.

## Experiment 01: conditional inference smoke test

Recommended Colab GPU: **L4**. It has enough VRAM for 512×512 PathOGen inference and is generally a better value than using A100/H100 for a one-image smoke test. T4 is supported but slower; A100/H100 are supported but unnecessary for this test.

Open `PathOGen_Inference_L4.ipynb` in Colab, select a GPU runtime, and choose **Runtime → Run all**. The notebook:

1. Clones the selected Git branch/commit.
2. Verifies the assigned GPU.
3. Installs pinned inference dependencies without replacing Colab's CUDA PyTorch.
4. Downloads the dataset and checkpoint ZIPs from Google Drive.
5. Safely extracts them and deletes the archives to conserve disk.
6. Finds the nested dataset/checkpoint directories by their contents.
7. Generates a deterministic conditional H&E sample.
8. Saves the generated image, source image, spatial map, comparison grid, and JSON manifest.

For final paper runs, set `GIT_REF` to a commit SHA rather than `main`.

The Drive links must be shared so the Colab account can download them. The full extracted assets need roughly 31 GiB; the script checks free disk before downloading.

Command-line use inside Colab:

```bash
python experiments/01_inference_smoke.py \
  --data-url "https://drive.google.com/file/d/FILE_ID/view" \
  --model-url "https://drive.google.com/file/d/FILE_ID/view" \
  --num-images 1 \
  --steps 20 \
  --seed 42
```

## Fidelity experiments 02–04

The fidelity framework provides three resumable, end-to-end experiments:

- `02_morphology_fidelity.py`: paired, same-seed interventions in which exactly one standardized morphology coordinate is increased. Generated images are re-segmented with CellViT++, the original 16 morphology statistics are recomputed, and pooled/perturbed/delta Spearman statistics are reported.
- `03_spatial_count_fidelity.py`: Spearman correlation between exact source-GeoJSON cell counts used to construct each condition and CellViT++ counts from generated H&E, for total, neoplastic, inflammatory, connective, dead, and epithelial cells.
- `04_spatial_coordinate_fidelity.py`: class-specific centroid-density grid Spearman correlation (primary) plus distance-matched x/y coordinate correlation (secondary).

The source GeoJSON is used for count and coordinate targets because every saved spatial-map channel was independently normalized to peak intensity 255 during preprocessing. Consequently, map intensity does not preserve absolute cell count.

### Morphology intervention validity

The original preprocessing fitted `StandardScaler` but did not persist the scaler. Interventions therefore remain in the standardized training space and use the empirical morphology parquet directly. For each case and feature, the script:

1. estimates the case's empirical percentile;
2. moves it upward by `--quantile-shift` (default 0.20);
3. clamps it to the observed 1st–99th percentile range;
4. asserts that exactly one of the 16 coordinates changed;
5. uses the identical spatial map and noise seed for baseline and intervention.

The default features are the eight means (`area_mean`, `eccentricity_mean`, `solidity_mean`, `perimeter_mean`, `grad_mean`, and RGB means). Pass `--features` to test any subset of all 16 coordinates.

### Example end-to-end commands

Use the same output directory for experiments 03 and 04 so coordinate analysis reuses the generated images and CellViT++ predictions from count fidelity.

```bash
COMMON="--data-dir /content/data/512_final_dataset \
--checkpoint-dir /content/model/checkpoint-30000 \
--cellvit-root /content/CellViT-plus-plus \
--cellvit-model /content/models/CellViT-256-x40-AMP.pth \
--num-images 50 --steps 20 --seed 42"

python experiments/02_morphology_fidelity.py $COMMON \
  --output-dir /content/results/morphology \
  --features area_mean eccentricity_mean solidity_mean perimeter_mean grad_mean r_mean g_mean b_mean

python experiments/03_spatial_count_fidelity.py $COMMON \
  --output-dir /content/results/spatial

python experiments/04_spatial_coordinate_fidelity.py $COMMON \
  --output-dir /content/results/spatial \
  --grid-size 16 --max-match-distance 32
```

Run `--dry-run` first to validate aligned data, requested features, morphology ranges, and case plans without loading either model. Runs resume from existing generated PNGs and CellViT++ GeoJSONs unless `--overwrite` is passed.

### Future guided sampling

All generation calls pass through `experiments.fidelity.guidance.GuidanceHook`. Supply `--guidance-hook package.module:factory` and optional `--guidance-config config.json`. A hook can:

- adjust morphology or spatial controls before generation;
- modify latents at each denoising step for gradient/classifier guidance;
- score decoded candidates with a classifier or CellViT++;
- reject weak candidates and request resampling;
- return morphology deltas or a spatial-strength update for the next attempt.

Use `--max-guidance-attempts` to permit rejection sampling. Rejected final candidates are not saved unless `--keep-rejected` is explicitly set.

### Local/Colab dependencies

Install CUDA PyTorch/torchvision for the runtime first, then:

```bash
pip install -r experiments/requirements_fidelity.txt
```

On GTX 16-series cards, both PathOGen and CellViT++ automatically use FP32 because FP16 can produce non-finite tensors. Modern Colab/H200 GPUs use FP16 by default.

## Complete Colab workflow

The `colab-fidelity-experiments` branch includes a large-asset folder layout, automatic setup and download script, environment verifier, combined experiment launcher and a button-by-button notebook. See [`experiments/colab/README.md`](colab/README.md) or open [`PathOGen_Fidelity_Colab.ipynb`](PathOGen_Fidelity_Colab.ipynb).

For the end-to-end baseline FID/KID versus CellViT++ best-of-16 comparison, run `experiments/05_cellvit_rerank_fid_kid.py`. It searches neutral and −1 SD green across eight seeds, fixes spatial strength at 2 and denoising at 30 steps, and uses only the requested `+1/0/-1` cell-type and 50-pixel position score. Candidate batches are flattened across source cases, so batch size 32 processes two complete 16-candidate cases together without changing the experiment design.
