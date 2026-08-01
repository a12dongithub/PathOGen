#!/bin/bash
# Quick Overfit Test for Concat Conditioning
# Trains on 200 images for 500 steps (~10 min, single GPU)
# to validate the architecture before scaling to full training.

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "=========================================================="
echo "PathOGen: Quick Overfit Test (Concat Conditioning)"
echo "=========================================================="

eval "$(conda shell.bash hook)"

# ── Activate Environment ──
ENV_PREFIX="${CPATHOGEN_ENV_PREFIX:-$PROJECT_ROOT/.envs/pathogen}"
if [ ! -d "$ENV_PREFIX" ] || [ ! -f "$ENV_PREFIX/bin/python" ]; then
    echo "ERROR: Conda environment not found at $ENV_PREFIX"
    echo "Set CPATHOGEN_ENV_PREFIX to the environment created for this project."
    exit 1
fi
conda activate "$ENV_PREFIX"
echo "Activated conda env: $ENV_PREFIX"
echo "Python: $(python --version)"

# ── GPU Check ──
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi not found"

# ── Run the quick overfitting test ──
echo ""
echo "Starting quick overfit test..."
echo "  - 200 random training images"
echo "  - 500 training steps"
echo "  - FID evaluated every 100 steps"
echo "  - Output: artifacts/runs/quick_overfit_test/"
echo ""

python tests/integration/quick_overfit_test.py \
    --pretrained_model_name_or_path='Manojb/stable-diffusion-2-1-base' \
    --phase1_unet_checkpoint='artifacts/runs/phase1_domain_adapt/checkpoints/checkpoint-30000' \
    --tiles_dir='data/interim/tiles/tcga_brca' \
    --spatial_maps_dir='data/processed/generator/spatial_maps' \
    --num_images=200 \
    --num_steps=500 \
    --eval_every=100 \
    --lr=1e-5 \
    --output_dir='artifacts/runs/quick_overfit_test'

echo ""
echo "=========================================================="
echo "Test complete! Check artifacts/runs/quick_overfit_test/ for results."
echo "  - step0_gen_*.png     = baseline (should look like Phase 1)"
echo "  - step500_gen_*.png   = after training (should show spatial structure)"
echo "  - Compare gen images with step0_real_*.png for correspondence"
echo "=========================================================="
