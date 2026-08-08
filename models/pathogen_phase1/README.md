# PathOGen Phase-1 model

The active Phase-1 initializer is:

```text
models/pathogen_phase1/checkpoint_30000/
└── unet/
    ├── config.json
    └── diffusion_pytorch_model.safetensors
```

Only the four-channel UNet needed by Workflows 03 and 04 was promoted from the
historical `checkpoint-30000.zip`. The duplicate EMA weights, optimizer state,
scheduler state, and random-number state remain archived and are not required
to initialize a new run. Frozen VAE, CLIP, tokenizer, and diffusion-scheduler
components are reused from the compatible local Phase-2 bundle.

| File | SHA-256 |
|---|---|
| `unet/config.json` | `ade13bfb5fdd06ba17a1524ce36ffcc60279c9c913f62546c938d6b8eb07a908` |
| `unet/diffusion_pytorch_model.safetensors` | `45c250dbaa7a9d8dba85566dfc596865584483aeaba80783379689e0badf7a5b` |

Large weights are ignored by ordinary Git. Use an approved model registry or
Git LFS to distribute them.
