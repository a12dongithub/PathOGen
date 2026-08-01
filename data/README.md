# Data layout

The data tree separates immutable source data, reproducible intermediate products, task-ready processed inputs, and provenance-poor legacy caches. Large files are excluded from Git; missing-data directories contain a `README.md` describing what must be supplied.

```text
data/
├── raw/
│   ├── bach/
│   │   ├── images/{Normal,Benign,InSitu,Invasive}/
│   │   └── labels.csv
│   ├── pannuke/
│   └── tcga_brca/
│       ├── clinical/               # missing in this workspace
│       └── molecular_subtypes.csv
├── interim/
│   ├── tiles/
│   │   ├── bach/                   # 4,800 existing 512 px tiles
│   │   └── tcga_brca/              # local smoke-test tiles; full cohort absent
│   └── annotations/tcga_brca/geojson/  # matching local smoke-test GeoJSON
├── processed/
│   ├── conditions/
│   │   ├── spatial_maps/           # generated five-channel `.npz` files
│   │   ├── morphology/             # raw/standardized tables + scaler
│   │   └── metadata.jsonl          # Phase-1 ImageFolder metadata
│   └── classification/
│       ├── bach/{manifests,embeddings}/
│       └── tcga_subtypes/{manifests,embeddings}/
├── manifests/                       # dataset registry, checksums, licenses
└── misc/
    ├── tcga_10k_cached_tensors/     # legacy `.pt` caches, not canonical inputs
    ├── nuhtc_demo/                  # separated third-party demo data/outputs
    └── os_metadata/                 # retained `.DS_Store`/Thumbs.db files
```

Present material includes PanNuke folds, BACH source images and tiles, two
historical embedding tables, TCGA subtype/sample manifests, and 8,481 cached
`.pt` tensors. Three matched TCGA-BRCA tiles and GeoJSON files are used locally
to smoke-test condition building. The complete TCGA-BRCA tile/annotation cohort
is not present and must be supplied separately.

Model weights (`.pth`, `.safetensors`, training `.pt`) belong under `artifacts/`, not `data/misc/`. `data/misc/` is reserved for provenance-poor data caches that are not part of a reproducible pipeline contract.
