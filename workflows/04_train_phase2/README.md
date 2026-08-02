# Workflow 04: train Phase 2

Phase 2 starts from a four-channel Phase-1 UNet, expands its input convolution
to eight channels, and trains two structured controls:

- a CNN converts five 512 × 512 nucleus-class heatmaps to four 64 × 64 latent
  channels for direct concatenation; and
- 22 FiLM modules condition UNet ResNet blocks on the standardized 16-value
  morphology/stain vector.

The new input weights are initialized to zero, so adding the spatial path does
not disturb the Phase-1 prediction at initialization. The VAE and text encoder
remain frozen. Tiles, maps, and morphology rows are joined strictly by stem.

## Install and verify

```bash
pip install -e ".[training]"
python workflows/04_train_phase2/run.py --validate-only
python workflows/04_train_phase2/run.py \
  --forward-check --max-samples 1 --train-batch-size 1 \
  --dataloader-workers 0 --no-gradient-checkpointing
```

The forward check builds the train-time architecture from the Phase-1 model and
runs one real conditioned diffusion pass without updating or saving weights.
The repository's six tiles are integration fixtures, not an adequate training
cohort.

## Train

```bash
accelerate launch workflows/04_train_phase2/run.py \
  --config configs/training/phase2.yaml
```

The tracked configuration retains the historical 50,000-step schedule and
CUDA-oriented 8-bit Adam setting. Install `bitsandbytes` separately or add
`--no-use-8bit-adam`. Resume with `--resume-from-checkpoint latest` or an
explicit checkpoint path. CLI arguments override YAML values.

## Outputs

```text
artifacts/runs/phase2_concat_film/
├── checkpoints/checkpoint-<step>/  # resumable Accelerate state
│   └── training_manifest.json
└── models/final/
    ├── unet/                       # trained eight-channel UNet
    ├── spatial_encoder.pt
    ├── film_mlps.pt
    ├── vae/
    ├── text_encoder/
    ├── tokenizer/
    ├── scheduler/
    └── training_manifest.json
```

`models/final` satisfies the same checkpoint contract consumed by Workflow 05.
