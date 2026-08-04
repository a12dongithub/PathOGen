# FID58 checkpoint

The resolved checkpoint directory must contain:

- `unet/config.json`
- `unet/diffusion_pytorch_model.safetensors`
- `vae/`
- `film_mlps.pt`
- `spatial_encoder.pt`

The model Drive ZIP may extract into a nested `checkpoint-30000` folder. The setup script detects the actual directory automatically.
