#!/bin/bash
# Snapshot the conda environment before destroying the instance.
# Run on the cloud machine:  bash snapshot_instance.sh

set -e
OUT="$HOME/pathogen_env_snapshot"
mkdir -p "$OUT"

eval "$(conda shell.bash hook)"
conda activate pathogen

echo "Exporting conda environment..."
conda env export > "$OUT/conda_environment.yml"
conda env export --no-builds > "$OUT/conda_environment_portable.yml"

echo "Exporting pip packages..."
pip freeze > "$OUT/pip_requirements.txt"

echo "Saving version info..."
python -c "
import torch, sys
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA (PyTorch): {torch.version.cuda}')
" > "$OUT/versions.txt"
nvidia-smi >> "$OUT/versions.txt" 2>/dev/null || true

# Copy training scripts too (lightweight)
echo "Copying scripts..."
mkdir -p "$OUT/scripts"
cp *.py *.sh "$OUT/scripts/" 2>/dev/null || true

tar -czf "$HOME/pathogen_env_snapshot.tar.gz" -C "$HOME" pathogen_env_snapshot/
SIZE=$(du -sh "$HOME/pathogen_env_snapshot.tar.gz" | cut -f1)

echo ""
echo "Done! Archive: ~/pathogen_env_snapshot.tar.gz ($SIZE)"
echo "Download: scp user@cloud:~/pathogen_env_snapshot.tar.gz ."
echo ""
echo "To restore on a new machine:"
echo "  tar -xzf pathogen_env_snapshot.tar.gz"
echo "  conda create -n pathogen python=3.10 -y"
echo "  conda activate pathogen"
echo "  pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128"
echo "  pip install -r pathogen_env_snapshot/pip_requirements.txt"
