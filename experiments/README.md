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
