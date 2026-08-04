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
# A shareable Google Drive checkpoint URL
!python experiments/colab/setup_colab.py \
  --cellvit-model-url "PASTE_DRIVE_URL" \
  --output-root /content/drive/MyDrive/PathOGenResults
```

You can also upload the file directly to `assets/checkpoints/cellvit/` and run setup without either checkpoint argument.

Setup performs the following steps:

1. installs the pinned fidelity dependencies without replacing Colab's CUDA PyTorch;
2. downloads and safely extracts the dataset and FID58 checkpoint ZIPs;
3. skips downloads whenever valid extracted assets already exist;
4. sparse-clones the required `cellvit/` source package into `assets/external/CellViT-plus-plus/repository/`;
5. discovers nested ZIP layouts and writes `assets/runtime_paths.json`;
6. leaves at least the requested large files outside Git tracking.

Preparing both ZIPs requires approximately 55 GiB of free disk. Archives are deleted after successful extraction by default.

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

## Custom asset locations

The setup script accepts `--asset-root`, `--data-root`, `--checkpoint-root`, `--cellvit-root`, `--cellvit-model`, and `--output-root`. If `--asset-root` is changed, pass the generated configuration to later commands:

```python
!python experiments/colab/run_fidelity_suite.py \
  --config /custom/assets/runtime_paths.json \
  --experiments all
```

Future classifier or CellViT-guided sampling can be enabled with `--guidance-hook package.module:factory`, `--guidance-config`, and `--max-guidance-attempts` without changing the experiment scripts.
