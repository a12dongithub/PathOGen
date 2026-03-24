#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# restore_instance.sh — Restore from pathogen_snapshot on a new machine.
#
# Usage:
#   tar -xzf pathogen_snapshot.tar.gz
#   cd pathogen_snapshot
#   bash restore_instance.sh /workspace/PathOGen
# ═══════════════════════════════════════════════════════════════

set -e

DEST="${1:-/workspace/PathOGen}"
SNAPSHOT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "═══════════════════════════════════════════════"
echo "  Restoring PathOGen to: $DEST"
echo "═══════════════════════════════════════════════"

# ── 1. Create conda environment ──
echo ""
echo "[1/5] Creating conda environment..."
eval "$(conda shell.bash hook)"

if conda info --envs | grep -q 'pathogen'; then
    echo "  Environment 'pathogen' exists, skipping."
else
    conda create -n pathogen python=3.10 -y
fi
conda activate pathogen

# Install PyTorch CUDA first (the conda yml may not get this right across machines)
echo "[2/5] Installing PyTorch with CUDA..."
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Install remaining packages
echo "[3/5] Installing pip packages..."
pip install -q diffusers==0.30.2 transformers==4.44.2 accelerate scipy pandas pyarrow torchmetrics torch-fidelity \
    numba opencv-python-headless datasets bitsandbytes scikit-learn joblib tqdm "numpy<2"

# ── 2. Set up accelerate ──
echo "[4/5] Configuring accelerate..."
rm -f ~/.cache/huggingface/accelerate/default_config.yaml
accelerate config default

# ── 3. Copy files ──
echo "[5/5] Copying files to $DEST..."
mkdir -p "$DEST"

# Scripts
cp "$SNAPSHOT_DIR/scripts/"* "$DEST/" 2>/dev/null || true

# Checkpoints
if [ -d "$SNAPSHOT_DIR/checkpoints" ]; then
    cp -r "$SNAPSHOT_DIR/checkpoints" "$DEST/"
    echo "  → Checkpoints restored"
fi

# Data metadata
if [ -d "$SNAPSHOT_DIR/data_meta" ]; then
    mkdir -p "$DEST/data/morphology_features"
    cp "$SNAPSHOT_DIR/data_meta/morphology_stats.parquet" "$DEST/data/morphology_features/" 2>/dev/null || true
    cp "$SNAPSHOT_DIR/data_meta/metadata.jsonl" "$DEST/data/" 2>/dev/null || true
    echo "  → Data metadata restored"
fi

# TensorBoard logs
if [ -d "$SNAPSHOT_DIR/tensorboard" ]; then
    cp -r "$SNAPSHOT_DIR/tensorboard/"* "$DEST/checkpoints/" 2>/dev/null || true
    echo "  → TensorBoard logs restored"
fi

# Set LD_LIBRARY_PATH for PyTorch
SITE_PACKAGES=$(python -c 'import site; print(site.getsitepackages()[0])')
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/cudnn/lib:$SITE_PACKAGES/nvidia/cublas/lib:$LD_LIBRARY_PATH"

echo ""
echo "═══════════════════════════════════════════════"
echo "  RESTORE COMPLETE!"
echo "═══════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "  1. Copy your training tiles to $DEST/data/tiles/"
echo "  2. Copy spatial maps to $DEST/data/spatial_maps/"
echo "  3. cd $DEST && conda activate pathogen"
echo "  4. bash start_cloud_training.sh"
