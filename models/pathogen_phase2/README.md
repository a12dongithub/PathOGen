# PathOGen Phase-2 model

Canonical local checkpoint:

```text
models/pathogen_phase2/checkpoint_30000/
├── unet/                 # trained eight-channel diffusion UNet
├── vae/                  # frozen VAE bundled by the accelerator checkpoint
├── spatial_encoder.pt    # trained five-channel spatial encoder
├── film_mlps.pt          # trained 16-value morphology/stain FiLM modules
├── tokenizer/            # frozen SD 2.1 tokenizer
├── text_encoder/         # frozen SD 2.1 CLIP text encoder
└── scheduler/            # frozen SD 2.1 noise-scheduler configuration
```

The checkpoint directory was moved from
`/Users/varangrai/Documents/cellpathogen/checkpoint-30000`. The tokenizer, text
encoder, and scheduler were copied from the resolved
`Manojb/stable-diffusion-2-1-base` revision
`0094d483a120f3f33dafbd187ea4aa60d10de75c`, allowing workflow 05 to run with
`--local-files-only` and without relying on the Hugging Face cache.

Accelerator training-resume files (`optimizer.bin`, `scheduler.bin`, and
`random_states_0.pkl`) are not part of the inference model and were moved to the
macOS Trash after the model checkpoint was verified. They are not read by
workflow 05.

Large weights are intentionally ignored by ordinary Git. Use an approved model
registry or Git LFS if this model must be distributed to another machine.
