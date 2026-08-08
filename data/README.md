# Active data

The active data tree contains the aligned inputs and all workflow outputs:

```text
data/
├── images/                                  # original H&E tiles
├── geojsons/                                # source CellViT++ annotations
├── spatial_maps/                            # five-channel NPZ controls
├── morphology_stats.parquet                 # 16-feature control table
├── morphology/                              # scaler, raw table, and manifest
└── evaluations/<run-name>/                   # all generated/evaluation outputs
```

All large data is intentionally ignored by Git. Preserve the source dataset's
split/provenance manifest alongside any evaluation run.
