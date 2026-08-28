# Rotation nuisance and Virchow2 extension

Run the workflow in two stages. Stage 1 reports rotation for ResNet-50,
CTransPath, UNI2-h, and PathLUPI+CONCH without extracting the real-image
dataset. Stage 2 adds Virchow2 and emits the complete table.

For PathLUPI, TVD and flip rate use cumulative survival from bin index 2, the
released BRCA interval spanning 42.4--78.9 months and containing five years.
Its official risk score remains unchanged for C-index calculation.

The counterfactuals are read from the already-extracted directory
`MyDrive/PTRI/CVPR/CPathOGen_Counterfactuals`. Both stages save resumable
artifacts under `MyDrive/PTRI/CVPR/CPathOGen_A100_Rotation_Virchow2`.

Add a Colab secret named `HF_TOKEN` and enable notebook access to it.

## Stage 1: rotation for all models except Virchow2

This stage does not locate or extract `512_final_dataset.zip`.

```python
from google.colab import drive, userdata
from pathlib import Path
import json
import os
import torch

drive.mount("/content/drive")
token = userdata.get("HF_TOKEN")
if not token:
    raise RuntimeError("Add HF_TOKEN in Colab Secrets and grant notebook access.")
os.environ["HF_TOKEN"] = token
if not torch.cuda.is_available():
    raise RuntimeError("Enable a GPU runtime before running this cell.")
print("GPU:", torch.cuda.get_device_name(0))

REPO = Path("/content/PathOGen")
BRANCH = "codex/inflammatory-mass-generation"
MYDRIVE = Path("/content/drive/MyDrive")
CVPR_ROOT = MYDRIVE / "PTRI" / "CVPR"
CF_SOURCE = CVPR_ROOT / "CPathOGen_Counterfactuals"
WORK_ROOT = Path("/content/cpathogen_rotation")
OUTPUT_ROOT = CVPR_ROOT / "CPathOGen_A100_Rotation_Virchow2"

if not (CF_SOURCE / "organized_bucket_images.csv").is_file():
    raise FileNotFoundError(f"Counterfactual folder is invalid: {CF_SOURCE}")
if not (REPO / ".git").is_dir():
    !git clone --branch {BRANCH} --single-branch https://github.com/a12dongithub/PathOGen.git {REPO}
else:
    !git -C {REPO} fetch origin {BRANCH}
    !git -C {REPO} checkout {BRANCH}
    !git -C {REPO} pull --ff-only origin {BRANCH}

!pip install -q -e "{REPO}[endpoints]" hf_xet
!pip install -q git+https://github.com/Mahmoodlab/CONCH.git

from huggingface_hub import snapshot_download

WF = REPO / "workflows" / "11_tile_local_xai_rotation_virchow2"
PATHLUPI_ROOT = Path("/content/external/PathLUPI")
if not (PATHLUPI_ROOT / ".git").is_dir():
    !git clone https://github.com/ChengJin-git/PathLUPI.git {PATHLUPI_ROOT}

!python "{WF / 'prepare_colab_inputs.py'}" \
    --mydrive-root "{MYDRIVE}" \
    --cvpr-root "{CVPR_ROOT}" \
    --work-root "{WORK_ROOT}" \
    --output-root "{OUTPUT_ROOT}" \
    --counterfactual-source "{CF_SOURCE}" \
    --skip-dataset

paths = json.loads((OUTPUT_ROOT / "resolved_paths.json").read_text())
COUNTERFACTUAL_ROOT = Path(paths["counterfactual_root"])
ENDPOINT_ROOT = Path(paths["endpoint_root"])
RESULTS_ROOT = Path(paths["results_root"])
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
ROTATION_DIR = WORK_ROOT / "rotation_manifest"

!python "{REPO / 'workflows/10_train_evaluate_endpoint_models/prepare_rotation_nuisance.py'}" \
    --counterfactual-root "{COUNTERFACTUAL_ROOT}" \
    --tile-manifest "{ENDPOINT_ROOT / 'tile_manifest.csv'}" \
    --output-dir "{ROTATION_DIR}" \
    --local-image-cache-dir "{WORK_ROOT / 'rotation_sources'}" \
    --num-images 1000 \
    --seed 42

!python "{REPO / 'workflows/10_train_evaluate_endpoint_models/extract_rotation_virchow2_embeddings.py'}" \
    --endpoint-root "{ENDPOINT_ROOT}" \
    --counterfactual-root "{COUNTERFACTUAL_ROOT}" \
    --real-images-dir "/content/unused" \
    --rotation-manifest "{ROTATION_DIR / 'images.csv'}" \
    --models resnet50 ctranspath uni2h conch \
    --full-models none \
    --batch-size 64 \
    --num-workers 8 \
    --shard-size 2048 \
    --device cuda

!python "{WF / 'run_local_xai_rerun.py'}" \
    --endpoint-root "{ENDPOINT_ROOT}" \
    --counterfactual-root "{COUNTERFACTUAL_ROOT}" \
    --output-root "{RESULTS_ROOT}" \
    --models resnet50 ctranspath uni2h \
    --bag-sizes 16 \
    --primary-bag-size 16 \
    --seed 42

PATHLUPI_CHECKPOINTS = Path(snapshot_download(
    repo_id="peterjin0703/PathLUPI",
    allow_patterns=["survival/BRCA/*"],
    local_dir="/content/pathlupi_checkpoints",
    token=token,
))

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
    --output-root "{RESULTS_ROOT}" \
    --models resnet50 ctranspath uni2h \
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

!python "{WF / 'merge_rotation_results.py'}" \
    --base-results-root "{RESULTS_ROOT}" \
    --rotation-summary "{RESULTS_ROOT / 'rotation_experiment_summary.csv'}" \
    --pathlupi-rotation-summary "{RESULTS_ROOT / 'pathlupi_rotation_summary.csv'}" \
    --output-root "{RESULTS_ROOT}" \
    --models resnet50 ctranspath uni2h pathlupi_conch \
    --primary-bag-size 16

FINAL = RESULTS_ROOT / "table4_rotation_without_virchow.md"
print("\nSaved:", FINAL)
print(FINAL.read_text())
```

## Stage 2: Virchow2 only, then final merge

Run this after Stage 1. It extracts the real-image dataset because Virchow2
does not yet have a real-tile cache. It also copies the counterfactual folder to
ephemeral Colab storage to avoid repeated small-file reads from Drive.

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
    raise RuntimeError("Enable an A100 GPU runtime before running this cell.")
print("GPU:", torch.cuda.get_device_name(0))

REPO = Path("/content/PathOGen")
BRANCH = "codex/inflammatory-mass-generation"
MYDRIVE = Path("/content/drive/MyDrive")
CVPR_ROOT = MYDRIVE / "PTRI" / "CVPR"
CF_SOURCE = CVPR_ROOT / "CPathOGen_Counterfactuals"
WORK_ROOT = Path("/content/cpathogen_virchow2")
OUTPUT_ROOT = CVPR_ROOT / "CPathOGen_A100_Rotation_Virchow2"

if not (REPO / ".git").is_dir():
    !git clone --branch {BRANCH} --single-branch https://github.com/a12dongithub/PathOGen.git {REPO}
else:
    !git -C {REPO} fetch origin {BRANCH}
    !git -C {REPO} checkout {BRANCH}
    !git -C {REPO} pull --ff-only origin {BRANCH}

!pip install -q -e "{REPO}[endpoints]" hf_xet

WF = REPO / "workflows" / "11_tile_local_xai_rotation_virchow2"
!python "{WF / 'prepare_colab_inputs.py'}" \
    --mydrive-root "{MYDRIVE}" \
    --cvpr-root "{CVPR_ROOT}" \
    --work-root "{WORK_ROOT}" \
    --output-root "{OUTPUT_ROOT}" \
    --counterfactual-source "{CF_SOURCE}"

paths = json.loads((OUTPUT_ROOT / "resolved_paths.json").read_text())
REAL_IMAGES = Path(paths["real_images_dir"])
ENDPOINT_ROOT = Path(paths["endpoint_root"])
RESULTS_ROOT = Path(paths["results_root"])
VIRCH_RESULTS = OUTPUT_ROOT / "results_virchow2"
VIRCH_RESULTS.mkdir(parents=True, exist_ok=True)

LOCAL_CF = WORK_ROOT / "CPathOGen_Counterfactuals"
LOCAL_CF_MARKER = LOCAL_CF / ".copy_complete"
if not LOCAL_CF_MARKER.is_file():
    if LOCAL_CF.exists():
        shutil.rmtree(LOCAL_CF)
    print("Copying counterfactual PNGs from Drive to local Colab storage...")
    shutil.copytree(CF_SOURCE, LOCAL_CF)
    LOCAL_CF_MARKER.write_text("complete\n")

ROTATION_DIR = WORK_ROOT / "rotation_manifest"
!python "{REPO / 'workflows/10_train_evaluate_endpoint_models/prepare_rotation_nuisance.py'}" \
    --counterfactual-root "{LOCAL_CF}" \
    --tile-manifest "{ENDPOINT_ROOT / 'tile_manifest.csv'}" \
    --output-dir "{ROTATION_DIR}" \
    --num-images 1000 \
    --seed 42

!python "{REPO / 'workflows/10_train_evaluate_endpoint_models/extract_rotation_virchow2_embeddings.py'}" \
    --endpoint-root "{ENDPOINT_ROOT}" \
    --counterfactual-root "{LOCAL_CF}" \
    --real-images-dir "{REAL_IMAGES}" \
    --rotation-manifest "{ROTATION_DIR / 'images.csv'}" \
    --models virchow2 \
    --full-models virchow2 \
    --virchow-batch-size 32 \
    --num-workers 8 \
    --shard-size 2048 \
    --device cuda

!python "{WF / 'run_local_xai_rerun.py'}" \
    --endpoint-root "{ENDPOINT_ROOT}" \
    --counterfactual-root "{LOCAL_CF}" \
    --output-root "{VIRCH_RESULTS}" \
    --models virchow2 \
    --bag-sizes 16 \
    --primary-bag-size 16 \
    --seed 42

!python "{WF / 'score_rotation_extension.py'}" \
    --endpoint-root "{ENDPOINT_ROOT}" \
    --base-results-root "{RESULTS_ROOT}" \
    --virchow-results-root "{VIRCH_RESULTS}" \
    --output-root "{VIRCH_RESULTS}" \
    --models virchow2 \
    --bag-size 16 \
    --seed 42

!python "{WF / 'merge_rotation_virchow2.py'}" \
    --base-results-root "{RESULTS_ROOT}" \
    --virchow-results-root "{VIRCH_RESULTS}" \
    --rotation-summary \
        "{RESULTS_ROOT / 'rotation_experiment_summary.csv'}" \
        "{VIRCH_RESULTS / 'rotation_experiment_summary.csv'}" \
    --pathlupi-rotation-summary "{RESULTS_ROOT / 'pathlupi_rotation_summary.csv'}" \
    --output-root "{RESULTS_ROOT}" \
    --primary-bag-size 16

FINAL = RESULTS_ROOT / "table4_rotation_virchow2.md"
print("\nSaved:", FINAL)
print(FINAL.read_text())
```
