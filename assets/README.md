# PathOGen large-asset workspace

This directory is the default Colab workspace for files that are required at runtime but must not be committed to Git.

```text
assets/
├── downloads/                              # Temporary Drive ZIP downloads
├── data/512_final_dataset/                 # Extracted aligned dataset
├── checkpoints/pathogen/checkpoint-30000/  # FID58 diffusion checkpoint
├── checkpoints/cellvit/                    # CellViT++ .pth checkpoint
├── external/CellViT-plus-plus/repository/  # Cloned CellViT++ source
└── outputs/                                # Experiment CSV, JSON, PNG and GeoJSON outputs
```

Run `python experiments/colab/setup_colab.py` to populate and validate this layout. Only these documentation files are tracked; large files remain ignored by Git.
