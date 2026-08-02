# Active artifacts

Only models required by the five active workflows remain active:

```text
artifacts/
├── models/
│   ├── cellvit_plus_plus/cellvit_sam_h_x40_amp_001/model.pth
│   ├── pathogen_phase1/checkpoint_30000/unet/
│   └── pathogen_phase2/checkpoint_30000/
└── runs/                              # new training and generation runs
```

Historical checkpoints, adapters, downstream models, ZIP files, metrics, and
generated results are under `archive/artifacts/`. Do not infer scientific
provenance from an archived filename.
