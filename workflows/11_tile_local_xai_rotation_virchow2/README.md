# A100 rotation and Virchow2 extension

This workflow produces the final counterfactual-probing table under the corrected
protocol:

- PAM50 counterfactual response is scored directly on one generated tile.
- Survival uses a fixed 16-tile bag: one generated tile and 15 unchanged real
  context tiles from the same patient.
- Rotation compares the unrotated generated tile with exact 90, 180, and 270
  degree views.
- BNR is the mean response to the four biological experiments divided by the
  mean response to stain brightness and rotation, with equal experiment weight.

The Drive output contains the resumable embedding caches and compact result
artifacts, but no duplicated real or counterfactual image folders.

## Required Drive inputs

The staging script searches both `/content/drive/MyDrive/PTRI/CVPR` and all of
`/content/drive/MyDrive` for:

- `512_final_dataset.zip`;
- exactly seven `CPathOGen_Counterfactuals*.zip` archives;
- either an extracted `endpoint_models` directory or `PathOGenResults*.zip`.

Add a Colab secret named `HF_TOKEN`, grant the notebook access to it, and select
an A100 runtime. The Hugging Face account must have access to UNI2-h, CONCH, and
Virchow2.

## Single Colab cell

```python
from google.colab import drive, userdata
from pathlib import Path
import json
import os
import shutil
import torch

drive.mount("/content/drive")

token = userdata.get("HF_TOKEN")
if not token:
    raise RuntimeError("Add HF_TOKEN in Colab Secrets and grant notebook access.")
os.environ["HF_TOKEN"] = token

if not torch.cuda.is_available():
    raise RuntimeError("Enable a GPU runtime before running this cell.")
gpu_name = torch.cuda.get_device_name(0)
print("GPU:", gpu_name)
if "A100" not in gpu_name:
    raise RuntimeError(f"This workflow is sized for an A100, but found {gpu_name}.")

REPO = Path("/content/PathOGen")
BRANCH = "codex/inflammatory-mass-generation"
MYDRIVE = Path("/content/drive/MyDrive")
CVPR_ROOT = MYDRIVE / "PTRI" / "CVPR"
WORK_ROOT = Path("/content/cpathogen_a100")
OUTPUT_ROOT = CVPR_ROOT / "CPathOGen_A100_Rotation_Virchow2"
WF = REPO / "workflows" / "11_tile_local_xai_rotation_virchow2"

if not (REPO / ".git").is_dir():
    !git clone --branch {BRANCH} --single-branch https://github.com/a12dongithub/PathOGen.git {REPO}
else:
    !git -C {REPO} fetch origin {BRANCH}
    !git -C {REPO} checkout {BRANCH}
    !git -C {REPO} pull --ff-only origin {BRANCH}

!pip install -q -e "{REPO}[endpoints]" hf_xet
!pip install -q git+https://github.com/Mahmoodlab/CONCH.git

from huggingface_hub import snapshot_download

PATHLUPI_ROOT = Path("/content/external/PathLUPI")
if not (PATHLUPI_ROOT / ".git").is_dir():
    !git clone https://github.com/ChengJin-git/PathLUPI.git {PATHLUPI_ROOT}

!python "{WF / 'prepare_colab_inputs.py'}" \
    --mydrive-root "{MYDRIVE}" \
    --cvpr-root "{CVPR_ROOT}" \
    --work-root "{WORK_ROOT}" \
    --output-root "{OUTPUT_ROOT}"

paths = json.loads((OUTPUT_ROOT / "resolved_paths.json").read_text())
DATASET_ROOT = Path(paths["dataset_root"])
REAL_IMAGES = Path(paths["real_images_dir"])
COUNTERFACTUAL_ROOT = Path(paths["counterfactual_root"])
ENDPOINT_ROOT = Path(paths["endpoint_root"])
RESULTS_ROOT = Path(paths["results_root"])
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

ROTATION_DIR = COUNTERFACTUAL_ROOT / "image_rotation"
!python "{REPO / 'workflows/10_train_evaluate_endpoint_models/prepare_rotation_nuisance.py'}" \
    --counterfactual-root "{COUNTERFACTUAL_ROOT}" \
    --tile-manifest "{ENDPOINT_ROOT / 'tile_manifest.csv'}" \
    --output-dir "{ROTATION_DIR}" \
    --num-images 1000 \
    --seed 42

!python "{REPO / 'workflows/10_train_evaluate_endpoint_models/extract_rotation_virchow2_embeddings.py'}" \
    --endpoint-root "{ENDPOINT_ROOT}" \
    --counterfactual-root "{COUNTERFACTUAL_ROOT}" \
    --real-images-dir "{REAL_IMAGES}" \
    --rotation-manifest "{ROTATION_DIR / 'images.csv'}" \
    --models resnet50 ctranspath uni2h virchow2 conch \
    --full-models virchow2 \
    --batch-size 64 \
    --virchow-batch-size 32 \
    --num-workers 8 \
    --shard-size 2048 \
    --device cuda

!python "{WF / 'run_local_xai_rerun.py'}" \
    --endpoint-root "{ENDPOINT_ROOT}" \
    --counterfactual-root "{COUNTERFACTUAL_ROOT}" \
    --output-root "{RESULTS_ROOT}" \
    --models resnet50 ctranspath uni2h virchow2 \
    --bag-sizes 16 \
    --primary-bag-size 16 \
    --seed 42

PATHLUPI_CHECKPOINTS = Path(
    snapshot_download(
        repo_id="peterjin0703/PathLUPI",
        allow_patterns=["survival/BRCA/*"],
        local_dir="/content/pathlupi_checkpoints",
        token=token,
    )
)

!python "{WF / 'run_pathlupi_fixedbag.py'}" \
    --endpoint-root "{ENDPOINT_ROOT}" \
    --pathlupi-root "{PATHLUPI_ROOT}" \
    --checkpoint-root "{PATHLUPI_CHECKPOINTS}" \
    --base-results-root "{RESULTS_ROOT}" \
    --output-root "{RESULTS_ROOT}" \
    --bag-size 16 \
    --seed 42 \
    --device cuda

!python "{WF / 'finalize_results.py'}" --results-root "{RESULTS_ROOT}"

!python "{WF / 'score_rotation_extension.py'}" \
    --endpoint-root "{ENDPOINT_ROOT}" \
    --base-results-root "{RESULTS_ROOT}" \
    --virchow-results-root "{RESULTS_ROOT}" \
    --output-root "{RESULTS_ROOT}" \
    --models resnet50 ctranspath uni2h virchow2 \
    --bag-size 16 \
    --seed 42

!python "{WF / 'score_pathlupi_rotation.py'}" \
    --endpoint-root "{ENDPOINT_ROOT}" \
    --pathlupi-root "{PATHLUPI_ROOT}" \
    --checkpoint-root "{PATHLUPI_CHECKPOINTS}" \
    --output-root "{RESULTS_ROOT}" \
    --bag-size 16 \
    --seed 42 \
    --device cuda

!python "{WF / 'merge_rotation_virchow2.py'}" \
    --base-results-root "{RESULTS_ROOT}" \
    --virchow-results-root "{RESULTS_ROOT}" \
    --rotation-summary "{RESULTS_ROOT / 'rotation_experiment_summary.csv'}" \
    --pathlupi-rotation-summary "{RESULTS_ROOT / 'pathlupi_rotation_summary.csv'}" \
    --output-root "{RESULTS_ROOT}" \
    --primary-bag-size 16

# Shards are resumable temporary files; final NPZ caches are retained in Drive.
shutil.rmtree(ENDPOINT_ROOT / "embedding_cache" / "shards", ignore_errors=True)

FINAL_MD = RESULTS_ROOT / "table4_rotation_virchow2.md"
FINAL_CSV = RESULTS_ROOT / "table4_rotation_virchow2.csv"
print("\nFinal Markdown table:", FINAL_MD)
print("Final CSV table:", FINAL_CSV)
print(FINAL_MD.read_text())
```

The final table is saved under
`MyDrive/PTRI/CVPR/CPathOGen_A100_Rotation_Virchow2/results/`. Re-running the
cell reuses complete embedding caches and resumable shards.
