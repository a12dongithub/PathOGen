@echo off
echo [1/3] Creating Conda environment 'pathogen_infer'...
call conda create -n pathogen_infer python=3.10 -y
call conda activate pathogen_infer

echo [2/3] Installing dependencies...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install diffusers==0.30.2 transformers==4.44.2 accelerate scipy pandas pyarrow torchmetrics torch-fidelity numba opencv-python-headless datasets scikit-learn joblib tqdm "numpy<2"

echo [3/3] Running generation script...
python generate_inspection_images.py
echo Done!
