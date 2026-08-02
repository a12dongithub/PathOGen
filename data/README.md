# Active data

The active data tree contains only the aligned inputs used by Workflows 01, 02,
and 05:

```text
data/
├── interim/
│   ├── tiles/tcga_brca/                    # six prepared 512 px tiles
│   └── annotations/tcga_brca/geojson/      # matching nucleus annotations
└── processed/conditions/
    ├── spatial_maps/                        # six five-channel NPZ files
    ├── morphology/                          # model-compatible table/manifest
    └── metadata.jsonl
```

The six examples are an integration fixture, not a scientific cohort. Full
TCGA-BRCA data must be supplied externally with its split/provenance manifest.
BACH, PanNuke, cached tensors, classification features, NuHTC demo data, and
other inactive datasets were moved under `archive/data/`.
