# PathOGen fidelity experiments on Colab

This branch provides a reproducible Colab pipeline for the morphology, spatial-count and spatial-coordinate experiments. Use an L4 as the cost-effective default. A100, H100 and H200 runtimes are also supported.

## 1. Start a GPU runtime and clone this branch

```python
!nvidia-smi
!git clone --branch colab-fidelity-experiments --single-branch https://github.com/a12dongithub/PathOGen.git /content/PathOGen
%cd /content/PathOGen
```

The repository contains tracked folders under `assets/` for the large runtime files, but it never commits their contents.

## 2. Provide the CellViT++ checkpoint

The data and FID58 checkpoint Drive links are already configured. CellViT++ source is cloned automatically. You must provide a CellViT++ `.pth` checkpoint by one of these methods:

```python
# A checkpoint already stored on mounted Google Drive
!python experiments/colab/setup_colab.py \
  --cellvit-model /content/drive/MyDrive/PathOGenAssets/CellViT-256-x40-AMP.pth \
  --output-root /content/drive/MyDrive/PathOGenResults
```

```python
# A shareable Google Drive URL for either a raw .pth or a ZIP containing it
!python experiments/colab/setup_colab.py \
  --cellvit-model-url "PASTE_DRIVE_URL" \
  --output-root /content/drive/MyDrive/PathOGenResults
```

You can also upload the file directly to `assets/checkpoints/cellvit/` and run setup without either checkpoint argument.

Setup performs the following steps:

1. installs the pinned fidelity dependencies without replacing Colab's CUDA PyTorch;
2. downloads and safely extracts the dataset, FID58 and optional CellViT++ checkpoint ZIPs;
3. skips downloads whenever valid extracted assets already exist;
4. sparse-clones the required `cellvit/` source package into `assets/external/CellViT-plus-plus/repository/`;
5. discovers nested ZIP layouts and writes `assets/runtime_paths.json`;
6. leaves at least the requested large files outside Git tracking.

Preparing both ZIPs requires approximately 55 GiB of free disk. Archives are deleted after successful extraction by default.

Public Drive files can temporarily hit Google's download quota. When the ZIP is
available on mounted Drive, bypass `gdown` with its local path. Setup preserves
mounted archives and still reuses an already extracted dataset:

```python
!python experiments/colab/setup_colab.py \
  --data-archive "/content/drive/MyDrive/PTRI/CVPR/512_final_dataset.zip" \
  --cellvit-model "/content/drive/MyDrive/PathOGenAssets/cellvit-model.zip" \
  --output-root "/content/drive/MyDrive/PathOGenResults"
```

The equivalent option for the diffusion checkpoint is `--model-archive`.

The default CellViT++ source is pinned to a tested upstream commit. Override it only when intentionally validating a different revision with `--cellvit-git-ref`.

## 3. Verify before spending GPU time

```python
!python experiments/colab/verify_colab.py

!python experiments/colab/run_fidelity_suite.py \
  --dry-run \
  --num-images 3
```

Verification checks CUDA, CuPy, the aligned dataset, checkpoint structure, CellViT++ imports and checkpoint, then runs the unit tests. The dry run validates all three experiment plans and morphology ranges without loading either model.

## 4. End-to-end smoke test

```python
!python experiments/colab/run_fidelity_suite.py --smoke-test
```

This generates one deliberately under-denoised two-step sample, runs CellViT++, and writes a spatial-count result. It verifies wiring only and is not an image-quality measurement.

## 5. Run the paper experiments

```python
!python experiments/colab/run_fidelity_suite.py \
  --experiments all \
  --num-images 25 \
  --steps 20 \
  --bootstrap 1000 \
  --seed 42
```

With the default eight morphology interventions, `N` cases require approximately `10N` generations: baseline plus eight paired interventions for morphology, and one spatial baseline reused by both spatial analyses. Start with a small `--num-images`, inspect outputs, and then scale up.

Run individual experiments when needed:

```python
!python experiments/colab/run_fidelity_suite.py \
  --experiments morphology \
  --num-images 50 \
  --features area_mean eccentricity_mean solidity_mean perimeter_mean grad_mean r_mean g_mean b_mean

!python experiments/colab/run_fidelity_suite.py \
  --experiments spatial-count spatial-coordinate \
  --num-images 50
```

The spatial experiments deliberately share one output directory, so coordinate analysis reuses generated PNGs and CellViT++ GeoJSONs from count fidelity. All scripts resume existing artifacts unless `--overwrite` is supplied.

## CellViT++ best-of-16 FID/KID experiment

One entrypoint runs the complete comparison: it generates one fixed baseline image per input and calculates baseline FID/KID, generates and segments 16 candidates per input, selects the highest CellViT++ point score, and calculates FID/KID again on the selected set.

```python
!python experiments/05_cellvit_rerank_fid_kid.py \
  --dry-run \
  --num-images 3

!python experiments/05_cellvit_rerank_fid_kid.py \
  --num-images 100 \
  --seeds-per-config 8 \
  --match-radius 50 \
  --generation-batch-size 32 \
  --cellvit-batch-size 32 \
  --seed 42
```

The portable defaults remain a PathOGen batch of 4 and CellViT++ batch of 4.
Both backends automatically halve their current batch after a CUDA out-of-memory
error; use batch size 32 as a conservative A100 starting point. Reranking
flattens candidates across source cases: with 16 candidates per case, batch size
32 processes two source cases together. Each sample retains its own
deterministic `torch.Generator`, and batching therefore does not reuse noise across
candidates. Score CSVs are checkpointed every 10 completed inputs by default; use
`--save-every` to change that interval.

Before a large reranking run, verify the baseline metric without generating the
candidate grid. A later full run with identical arguments reuses these artifacts:

```python
!python experiments/05_cellvit_rerank_fid_kid.py \
  --num-images 5000 \
  --seeds-per-config 8 \
  --generation-batch-size 32 \
  --baseline-only \
  --seed 42
```

After a Colab reset, bypass the baseline phase completely when the corresponding
Drive-backed metric sets and `metrics_before_reranking.json` already exist:

```python
!python experiments/05_cellvit_rerank_fid_kid.py \
  --num-images 2000 \
  --seeds-per-config 8 \
  --generation-batch-size 32 \
  --cellvit-batch-size 32 \
  --skip-baseline \
  --seed 42
```

Generated candidate images are read with exponential-backoff retries to tolerate
transient mounted-Drive `Errno 5` failures.

The script reads `assets/runtime_paths.json`, so no asset paths are needed after setup. For each input cell, unique CellViT++ detections are assigned with the following only ranking score:

- same type within 50 pixels: `+1`;
- different type within 50 pixels: `0`;
- no detected cell within 50 pixels: `-1`.

There is no additional count, morphology or spatial-correlation score. Extra detections are reported but receive no penalty. Tied candidates retain the first fixed configuration/seed order.

The focused design fixes denoising at 30 steps and spatial strength at 2, then evaluates `2 green levels × 8 seeds = 16` candidates:

| Configuration | Green offset (SD) | ControlNet strength | Denoising steps |
|---|---:|---:|---:|
| `cfg00_g0_c2_s30` | 0 | 2 | 30 |
| `cfg01_gm1_c2_s30` | -1 | 2 | 30 |

The same eight deterministic noise seeds are reused across both configurations for each input. Use `--seeds-per-config` to reduce or increase that count. Green changes operate on standardized `g_mean` and are clamped to its empirical 1st–99th percentile range. The baseline FID/KID set always uses `cfg00_g0_c2_s30` and seed index zero. Use substantially more than 100 inputs for final paper FID; small runs are useful only for pipeline validation and preliminary KID estimates.

## Custom asset locations

The setup script accepts `--asset-root`, `--data-root`, `--checkpoint-root`, `--cellvit-root`, `--cellvit-model`, and `--output-root`. If `--asset-root` is changed, pass the generated configuration to later commands:

```python
!python experiments/colab/run_fidelity_suite.py \
  --config /custom/assets/runtime_paths.json \
  --experiments all
```

Future classifier or CellViT-guided sampling can be enabled with `--guidance-hook package.module:factory`, `--guidance-config`, and `--max-guidance-attempts` without changing the experiment scripts.
