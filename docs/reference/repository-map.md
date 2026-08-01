# Repository map

This is the canonical post-refactor layout as of 2026-07-27. See [MIGRATION_MANIFEST.md](../../MIGRATION_MANIFEST.md) for old-to-new mappings.

```text
refactored/
├── src/cpathogen/                 # reusable implementation
│   ├── annotation/                # CellViT++ adapter + GeoJSON contract
│   ├── preprocessing/             # maps, morphology, metadata
│   ├── generation/                # phase 1, phase 2, inference
│   │   └── conditioning/          # spatial encoder and FiLM destinations
│   ├── encoders/                  # UNI2, CTransPath, ResNet adapters
│   ├── classification/            # downstream heads and aggregation
│   ├── counterfactuals/           # interventions and matched pairs
│   ├── evaluation/                # realism, fidelity, classifier/representation metrics
│   │   └── misc/legacy_fid/       # retained historical FID utilities
│   └── utils/                     # paths, I/O, configuration, reproducibility
├── workflows/                     # chronological runnable entry points 01-08
├── experiments/                   # historical research scripts by question
│   └── misc/inspection/           # uncategorized inspection utilities
├── tools/                         # human-facing curator and dataset conversion tools
├── configs/                       # portable reference data/model/training settings
├── data/                          # raw, interim, processed, manifests, misc caches
├── artifacts/                     # runs, shared models, and archived model material
├── tests/                         # unit, integration, fixtures
├── docs/                          # onboarding, architecture, guides, references
├── reports/                       # posters, manuscripts, figures, tables
├── third_party/nuhtc/             # vendored legacy/alternative segmentation project
└── archive/                       # cold storage and non-canonical material
```

## Chronological workflow

| Step | Entry point | Canonical implementation | Status |
|---:|---|---|---|
| 1 | `workflows/01_annotate_nuclei/run.py` | `annotation/cellvit_adapter.py`, `annotation/geojson.py` | TODO stubs; CellViT++ is absent |
| 2 | `workflows/02_build_conditions/run.py` | `preprocessing/spatial_maps.py`, `morphology_features.py`, `metadata.py` | implemented and validated on a three-tile smoke-test set |
| 3 | `workflows/03_train_phase1/run.py` | `generation/phase1.py` | wrapper present |
| 4 | `workflows/04_train_phase2/run.py` | `generation/phase2.py` | wrapper and cloud script present |
| 5 | `workflows/05_generate_counterfactuals/run.py` | `generation/inference.py`, `counterfactuals/` | orchestration TODO; legacy scripts in `misc/` |
| 6 | `workflows/06_extract_embeddings/run.py` | `encoders/` | TODO |
| 7 | `workflows/07_train_classifiers/run.py` | `classification/` | TODO |
| 8 | `workflows/08_evaluate_counterfactuals/run.py` | `evaluation/` | TODO |

TODO files are intentional executable placeholders, not completed functionality.

## Canonical source names

| File | Purpose |
|---|---|
| `preprocessing/spatial_maps.py` | Convert nucleus GeoJSON into five-channel `.npz` maps |
| `preprocessing/morphology_features.py` | Write raw and standardized 16-feature tables, scaler, and manifest |
| `preprocessing/metadata.py` | Write phase-1 `metadata.jsonl` |
| `generation/phase1.py` | H&E diffusion-domain adaptation |
| `generation/phase2.py` | Concatenated spatial conditioning plus morphology FiLM |
| `generation/inference.py` | Condition pairing, sampling, and validation |

## Data contract

```text
data/
├── raw/
│   ├── bach/{images,labels.csv}
│   ├── pannuke/
│   └── tcga_brca/{clinical,molecular_subtypes.csv}
├── interim/
│   ├── tiles/{bach,tcga_brca}
│   └── annotations/tcga_brca/geojson
├── processed/
│   ├── conditions/{spatial_maps,morphology,metadata.jsonl}
│   └── classification/{bach,tcga_subtypes}/{manifests,embeddings}
├── manifests/{datasets.yaml,checksums.csv,licenses.md}
└── misc/{tcga_10k_cached_tensors,os_metadata}
```

The BACH source images and 4,800 BACH tiles are present. A three-tile TCGA-BRCA
smoke-test subset and its derived conditions are available locally but ignored by
Git; the complete TCGA-BRCA cohort remains absent. Existing `.pt` caches were
isolated in `data/misc/` because their provenance is incomplete and they are not
canonical condition inputs.

## Artifact contract

New training/evaluation work should use:

```text
artifacts/runs/<run-id>/
├── checkpoints/
├── models/
├── metrics/
├── figures/
└── run manifest/config files
```

The extracted legacy phase-2 checkpoint is under `artifacts/runs/legacy_phase2_fid58/checkpoints/checkpoint-30000/`. Checkpoint ZIP archives live in `artifacts/misc/checkpoint_archives/`. Shared downstream and adapter weights remain under `artifacts/models/`.

## Historical and third-party code

`experiments/` preserves numbered filenames because their order communicates research chronology. Project paths were normalized, but these scripts are still historical records and have not been scientifically rerun. `src/cpathogen/evaluation/misc/legacy_fid/` and `workflows/05_generate_counterfactuals/misc/` serve the same preservation role for old generator utilities.

`third_party/nuhtc/` is vendored upstream-derived code. Large NuHTC demo data is isolated under `data/misc/nuhtc_demo/`, and its PanNuke checkpoint is under `artifacts/models/third_party/nuhtc/`. Other upstream machine-specific examples are deliberately not treated as CPathoGen path configuration.
