# Model and data contract

## Generator

The supported checkpoint is a Stable Diffusion 2.1-derived 512×512 H&E
generator. The prompt is the constant `he`; this is domain-adapted diffusion,
not general text-to-image generation.

It has two controls:

| Control | Input | Model path |
|---|---|---|
| Spatial organization | Five 512×512 nucleus-density maps | CNN encodes `5 → 4` channels at 64×64, concatenated with the four noisy latent channels |
| Morphology/stain | 16 standardized per-tile values | FiLM scale/shift applied to UNet ResNet blocks |

The UNet therefore receives eight 64×64 channels: four noisy VAE latent
channels plus four spatial-control channels. FiLM scale and shift are clamped
to `[-0.5, 0.5]`.

## Spatial-map contract

Each `<stem>.npz` contains key `map` with shape `(512, 512, 5)`. Inputs may be
`uint8` in `[0,255]` or float in `[0,1]`; the loader normalizes to float
`(5,512,512)` in `[0,1]`.

Channel order is fixed:

0. Neoplastic
1. Inflammatory
2. Connective
3. Dead
4. Epithelial

These are blurred, peak-normalized nucleus-centroid density maps. They are not
instance masks and do not preserve calibrated absolute counts.

## Morphology/stain contract

`morphology_stats.parquet` must be indexed by tile stem and contain these exact
training-standardized columns, in order:

`area_mean`, `area_var`, `eccentricity_mean`, `eccentricity_var`,
`solidity_mean`, `solidity_var`, `perimeter_mean`, `perimeter_var`,
`grad_mean`, `grad_var`, `r_mean`, `r_var`, `g_mean`, `g_var`, `b_mean`,
`b_var`.

Never substitute raw values or a newly fitted scaler at inference. Preserve the
training scaler, feature order, clipping policy, split identifier, and hashes
beside every generated dataset.

## Required local layout

```text
data/images/<stem>.png                     # optional image provenance
data/spatial_maps/<stem>.npz               # required
data/morphology_stats.parquet              # required
models/pathogen_phase2/checkpoint_30000/   # required for generation
```

The checkpoint must include `unet/`, `vae/`, `tokenizer/`, `text_encoder/`,
`scheduler/`, `spatial_encoder.pt`, and `film_mlps.pt`.
