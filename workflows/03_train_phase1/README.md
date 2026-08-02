# Workflow 03: train Phase 1

Phase 1 adapts a four-channel Stable Diffusion 2.1 UNet to 512 × 512 H&E tiles.
The VAE and CLIP text encoder remain frozen; the UNet learns the diffusion
noise-prediction objective. The default local initialization continues from the
historical Phase-1 checkpoint while reusing the compatible frozen components
bundled with the Phase-2 model.

The included `data/processed/conditions/metadata.jsonl` has only six fixture
tiles. Replace it with leakage-controlled training metadata before a real run.

## Install and verify

```bash
pip install -e ".[training]"
python workflows/03_train_phase1/run.py --validate-only
python workflows/03_train_phase1/run.py \
  --forward-check --max-samples 1 --train-batch-size 1 \
  --dataloader-workers 0 --no-gradient-checkpointing --no-use-ema
```

`--validate-only` checks every metadata path without loading a diffusion model.
`--forward-check` loads the actual local model, computes one no-gradient loss,
and writes no run artifacts. It is the appropriate bounded architecture check
on a Mac; it is not evidence that a long optimization run will fit in memory.

## Train

The tracked configuration preserves the historical 100,000-step target:

```bash
accelerate launch workflows/03_train_phase1/run.py \
  --config configs/training/phase1.yaml
```

That CUDA-oriented config enables 8-bit Adam. Install `bitsandbytes` separately
on a supported host, or add `--no-use-8bit-adam`. CLI options override YAML.
To start from the original Stable Diffusion model instead of continuing the
local Phase-1 UNet, pass a complete model ID/path and an empty initializer:

```bash
accelerate launch workflows/03_train_phase1/run.py \
  --base-model Manojb/stable-diffusion-2-1-base \
  --initial-unet "" --no-local-files-only
```

Resume with `--resume-from-checkpoint latest` or an explicit checkpoint path.

## Outputs

```text
artifacts/runs/phase1_domain_adapt/
├── checkpoints/checkpoint-<step>/  # Accelerate model/optimizer/RNG state
│   ├── unet_ema/                   # when EMA is enabled
│   └── training_manifest.json
└── models/final/
    ├── unet/                       # four-channel Phase-1 UNet
    ├── vae/
    ├── text_encoder/
    ├── tokenizer/
    ├── scheduler/
    └── training_manifest.json
```

The final directory is self-contained and may be supplied to Workflow 04 as
both `--base-model` and `--phase1-model`.
