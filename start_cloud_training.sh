#!/bin/bash
# PathOGen Cloud Training (Conda)
# Target: 8x NVIDIA RTX PRO 6000 (Ada), ~1M H&E tiles

set -e  # Exit on first error

echo "=========================================================="
echo "PathOGen: H&E Generation with Layout + Morphology Control"
echo "=========================================================="

eval "$(conda shell.bash hook)"

# ── 1. Environment Setup ──
echo "[1/6] Creating conda environment..."
if conda info --envs | grep -q 'pathogen'; then
    echo "Environment 'pathogen' already exists. Skipping creation."
else
    conda create -n pathogen python=3.10 -y
fi
conda activate pathogen

echo "[2/6] Installing dependencies..."
# Use PyTorch nightly with CUDA 12.8 for native Blackwell (sm_120) support
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
# Using latest diffusers/transformers. Removing xformers to rely on PyTorch 2.0+ native SDPA
pip install -q diffusers==0.30.2 transformers==4.44.2 accelerate scipy pandas pyarrow torchmetrics torch-fidelity \
    numba opencv-python-headless datasets bitsandbytes scikit-learn joblib tqdm "numpy<2"

# Clean up any residual config from newer versions to prevent unknown key errors
rm -f /workspace/.hf_home/accelerate/default_config.yaml
rm -f ~/.cache/huggingface/accelerate/default_config.yaml
accelerate config default
# ── 2. Data Preprocessing (precomputed, not during training) ──
echo "[3/6] Precomputing morphology features + spatial maps..."
if [ ! -f "./data/morphology_features/morphology_stats.parquet" ]; then
    python generate_spatial_maps.py --data_dir=./data --n_jobs=128
    python generate_morphology_features.py \
        --dataset_path=./data \
        --data_dir=./data \
        --output=./data/morphology_features/morphology_stats.parquet \
        --n_jobs=128
else
    echo "Morphology features already exist. Skipping."
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
# Exclude faulty GPU 4 (ECC error) and use the 7 healthy ones
export CUDA_VISIBLE_DEVICES="0,1,2,3,5,6,7"

# PyTorch 2.4/2.5+ Nightly via pip sometimes fails to locate its own bundled cuDNN/cuBLAS 
# libraries on certain OS/Docker distributions, causing 'Cannot load symbol cudnnGetVersion'
# We explicitly add them to the linker path.
SITE_PACKAGES=$(python -c 'import site; print(site.getsitepackages()[0])')
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/cudnn/lib:$SITE_PACKAGES/nvidia/cublas/lib:$LD_LIBRARY_PATH"

echo "[4/6] Phase 1: Domain adaptation (unconditional H&E generation)..."
# accelerate launch --multi_gpu --num_processes=7 train_text_to_image_base.py \
#     --pretrained_model_name_or_path='Manojb/stable-diffusion-2-1-base' \
#     --train_data_dir='./data' \
#     --use_ema \
#     --resolution=512 \
#     --train_batch_size=12 \
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
#     --dataloader_num_workers=128 \
#     --report_to='tensorboard' \
#     --tracker_project_name='pathogen-phase1'

# ── 4. Phase 2: Layout + Morphology Conditioning ──
# Trains UNet (gentle LR), ControlNet (5-ch spatial maps), and FiLM MLPs (16D morphology).
# Text stays constant "he" — model conditions on layout & morphology, NOT text.
# FRESH START from Phase 1 checkpoint-30000 (best FID).
echo "[5/6] Phase 2: ControlNet (spatial) + FiLM (morphology) training..."
accelerate launch --multi_gpu --num_processes=7 train_pathogen.py \
    --pretrained_model_name_or_path='./checkpoints/phase1_domain_adapt' \
    --phase1_unet_checkpoint='./checkpoints/phase1_domain_adapt/checkpoint-30000' \
    --output_dir='./checkpoints/phase2_controlnet' \
    --train_data_dir='./data' \
    --resolution=512 \
    --learning_rate=1e-5 \
    --lr_scheduler='constant_with_warmup' \
    --lr_warmup_steps=1000 \
    --train_batch_size=12 \
    --gradient_accumulation_steps=1 \
    --gradient_checkpointing \
    --max_train_steps=120000 \
    --checkpointing_steps=10000 \
    --resume_from_checkpoint='./checkpoints/phase2_controlnet/checkpoint-10000' \
    --use_8bit_adam \
    --allow_tf32 \
    --dataloader_num_workers=128 \
    --report_to='tensorboard' \
    --tracker_project_name='pathogen-phase2'

echo "[6/6] Training complete! Checkpoints saved to ./checkpoints/"
echo "  Phase 1: ./checkpoints/phase1_domain_adapt/"
echo "  Phase 2: ./checkpoints/phase2_controlnet/"
echo "  TensorBoard logs: tensorboard --logdir ./checkpoints/"
