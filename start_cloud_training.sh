#!/bin/bash
# PathOGen Cloud Training (Conda)
# Target: 8x NVIDIA Tesla V100 32GB, ~1M H&E tiles
# Experiment: UNet + VAE + ControlNet (no FiLM/morphology)

set -e  # Exit on first error

echo "=========================================================="
echo "PathOGen: H&E Generation with Spatial Layout Control"
echo "=========================================================="

eval "$(conda shell.bash hook)"

# ── 1. Environment Setup ──
ENV_PREFIX="/home/samarth.singhal/pathogen"
echo "[1/6] Creating conda environment at $ENV_PREFIX ..."
if [ -d "$ENV_PREFIX" ] && [ -f "$ENV_PREFIX/bin/python" ]; then
    echo "Environment at $ENV_PREFIX already exists. Skipping creation."
else
    conda create --prefix "$ENV_PREFIX" python=3.10 -y
fi
conda activate "$ENV_PREFIX"

echo "[2/6] Installing dependencies..."
# PyTorch stable with CUDA 11.8 for V100 (Volta sm_70)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# Using latest diffusers/transformers. Removing xformers to rely on PyTorch 2.0+ native SDPA
pip install -q diffusers==0.30.2 transformers==4.44.2 accelerate scipy pandas pyarrow torchmetrics torch-fidelity \
    numba opencv-python-headless datasets bitsandbytes scikit-learn joblib tqdm "numpy<2"

# Clean up any residual config from newer versions to prevent unknown key errors
rm -f /workspace/.hf_home/accelerate/default_config.yaml
rm -f ~/.cache/huggingface/accelerate/default_config.yaml
accelerate config default

# ── 2. Data Preprocessing (precomputed, not during training) ──
echo "[3/6] Precomputing spatial maps..."
if [ ! -d "./data/spatial_maps" ] || [ -z "$(ls -A ./data/spatial_maps 2>/dev/null)" ]; then
    python generate_spatial_maps.py --data_dir=./data --n_jobs=32
else
    echo "Spatial maps already exist. Skipping."
fi

# Note: generate_spatial_maps.py has an internal skip check per file, so it safely resumes

if [ ! -f "./data/metadata.jsonl" ]; then
    python generate_metadata.py
else
    echo "Metadata already exists. Skipping."
fi

# HuggingFace requires a source build of diffusers because of a hardcoded version string check.
# We comment out that single line dynamically so it works on pip installations.
sed -i 's/check_min_version("0.37.0.dev0")/# check_min_version("0.37.0.dev0")/g' train_text_to_image_base.py

# ── 3. Phase 1: Domain Adaptation ──
# Fine-tunes SD2.1 UNet to generate H&E tissue textures.
# Text prompt is constant ("he") for ALL samples — the model learns
# unconditional H&E generation, NOT text-conditioned generation.
# Use all 8 V100 GPUs
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

echo "[4/6] Phase 1: Domain adaptation (unconditional H&E generation)..."
# Phase 1 is already complete — uncomment to re-run if needed
# accelerate launch --multi_gpu --num_processes=8 train_text_to_image_base.py \
#     --pretrained_model_name_or_path='Manojb/stable-diffusion-2-1-base' \
#     --train_data_dir='./data' \
#     --use_ema \
#     --resolution=512 \
#     --train_batch_size=8 \
#     --gradient_accumulation_steps=1 \
#     --gradient_checkpointing \
#     --max_train_steps=100000 \
#     --learning_rate=1e-5 \
#     --lr_scheduler='constant_with_warmup' \
#     --lr_warmup_steps=1000 \
#     --checkpointing_steps=10000 \
#     --resume_from_checkpoint='latest' \
#     --output_dir='./checkpoints/phase1_domain_adapt' \
#     --use_8bit_adam \
#     --allow_tf32 \
#     --dataloader_num_workers=32 \
#     --report_to='tensorboard' \
#     --tracker_project_name='pathogen-phase1'

# ── 4. Phase 2: Layout Conditioning (ControlNet + UNet + VAE, no FiLM) ──
# Trains full UNet + VAE (gentle LR) and ControlNet (5-ch spatial maps, full LR).
# Text stays constant "he" — model conditions on layout only.
# FRESH START from Phase 1 checkpoint-30000 (best FID).
echo "[5/6] Phase 2: ControlNet (spatial) + UNet + VAE training (no FiLM)..."
accelerate launch --multi_gpu --num_processes=8 train_pathogen.py \
    --pretrained_model_name_or_path='./checkpoints/phase1_domain_adapt' \
    --phase1_unet_checkpoint='./checkpoints/phase1_domain_adapt/checkpoint-30000' \
    --output_dir='./checkpoints/phase2_controlnet' \
    --train_data_dir='./data' \
    --resolution=512 \
    --learning_rate=1e-5 \
    --lr_scheduler='constant_with_warmup' \
    --lr_warmup_steps=1000 \
    --train_batch_size=6 \
    --gradient_accumulation_steps=2 \
    --gradient_checkpointing \
    --max_train_steps=120000 \
    --checkpointing_steps=5000 \
    --use_8bit_adam \
    --allow_tf32 \
    --dataloader_num_workers=32 \
    --report_to='tensorboard' \
    --tracker_project_name='pathogen-phase2-no-film'

echo "[6/6] Training complete! Checkpoints saved to ./checkpoints/"
echo "  Phase 1: ./checkpoints/phase1_domain_adapt/"
echo "  Phase 2: ./checkpoints/phase2_controlnet/"
echo "  TensorBoard logs: tensorboard --logdir ./checkpoints/"
