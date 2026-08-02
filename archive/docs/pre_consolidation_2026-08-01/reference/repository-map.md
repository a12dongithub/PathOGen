# Repository map

This is the canonical post-refactor layout as of 2026-08-01.

```text
refactored/
├── src/cpathogen/                 # reusable implementation
│   ├── annotation/                # CellViT++ adapter + GeoJSON contract
│   ├── preprocessing/             # maps, morphology, metadata
│   ├── generation/                # phase 1/2, checkpoint loading, matched inference
│   │   └── conditioning/          # spatial encoder and FiLM destinations
│   ├── encoders/                  # UNI2, CTransPath, ResNet adapters
│   ├── classification/            # downstream heads and aggregation
│   ├── counterfactuals/           # interventions and matched pairs
│   ├── evaluation/                # realism, fidelity, classifier/representation metrics
│   │   └── misc/legacy_fid/       # retained historical FID utilities
│   └── utils/                     # paths, I/O, configuration, reproducibility
├── workflows/                     # chronological runnable entry points 01-08
├── experiments/                   # Python-only control intervention plugins
├── tools/                         # human-facing curator and dataset conversion tools
├── configs/                       # portable reference data/model/training settings
├── data/                          # raw, interim, processed, manifests, misc caches
├── artifacts/                     # runs, shared models, and archived model material
├── tests/                         # unit, integration, fixtures
├── docs/                          # onboarding, architecture, guides, references
│   └── project_reports/           # posters, manuscripts, and figures
├── third_party/
│   ├── cellvit_plus_plus/         # pinned source used by Workflow 01
│   └── nuhtc/                     # vendored legacy/alternative segmentation project
└── archive/
    └── experiments/               # historical experiment scripts awaiting review
```

## Chronological workflow

| Step | Entry point | Canonical implementation | Status |
|---:|---|---|---|
| 1 | `workflows/01_annotate_nuclei/run.py` | `annotation/cellvit_adapter.py`, `annotation/geojson.py` | implemented for source tiles and Workflow 05 matched pairs; validated on Apple MPS |
| 2 | `workflows/02_build_conditions/run.py` | `preprocessing/spatial_maps.py`, `morphology_features.py`, `metadata.py` | implemented and validated on a three-tile smoke-test set |
| 3 | `workflows/03_train_phase1/run.py` | `generation/phase1.py` | wrapper present |
| 4 | `workflows/04_train_phase2/run.py` | `generation/phase2.py` | wrapper and cloud script present |
| 5 | `workflows/05_generate_counterfactuals/run.py` | `generation/checkpoints.py`, `generation/counterfactuals.py`, `counterfactuals/` | implemented; validated on Apple MPS |
| 6 | `workflows/06_extract_embeddings/run.py` | `encoders/` | TODO |
| 7 | `workflows/07_train_classifiers/run.py` | `classification/` | TODO |
| 8 | `workflows/08_evaluate_counterfactuals/run.py` | `evaluation/` | TODO |

Remaining TODO files are intentional executable placeholders, not completed functionality.

## Canonical source names

| File | Purpose |
|---|---|
| `preprocessing/spatial_maps.py` | Convert nucleus GeoJSON into five-channel `.npz` maps |
| `annotation/cellvit_adapter.py` | Load the pinned CellViT-SAM-H model, annotate source/generated tiles, and write run manifests |
| `annotation/geojson.py` | Canonical per-nucleus GeoJSON conversion and validation |
| `preprocessing/morphology_features.py` | Write raw and standardized 16-feature tables, scaler, and manifest |
| `preprocessing/metadata.py` | Write phase-1 `metadata.jsonl` |
| `generation/phase1.py` | H&E diffusion-domain adaptation |
| `generation/phase2.py` | Concatenated spatial conditioning plus morphology FiLM |
| `generation/checkpoints.py` | Validate/load Phase-2 UNet, VAE, spatial encoder, FiLM, text, and scheduler components |
| `generation/counterfactuals.py` | Matched-noise DDIM sampling |
| `counterfactuals/conditions.py` | Aligned lazy condition store and tensor contracts |
| `counterfactuals/interventions.py` | Identity-by-default experiment transformation interface |

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

The BACH source images and 4,800 BACH tiles are present. A six-tile TCGA-BRCA
smoke-test subset, matching GeoJSON files, spatial maps, and the model-compatible
full-cohort standardized morphology table are available locally but ignored by
ordinary Git; the complete TCGA-BRCA image cohort remains external. Existing `.pt` caches were
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

The canonical Phase-2 inference model is under
`artifacts/models/pathogen_phase2/checkpoint_30000/`. It includes the trained
UNet, spatial encoder, and FiLM modules, plus the frozen VAE, tokenizer, text
encoder, and scheduler needed for offline generation. Checkpoint ZIP archives
live in `artifacts/misc/checkpoint_archives/`. Shared downstream and adapter
weights remain under `artifacts/models/`.

The canonical nucleus annotator is under
`artifacts/models/cellvit_plus_plus/cellvit_sam_h_x40_amp_001/model.pth`.
Its tracked README records architecture, scale, and SHA-256; the 2.6 GB weight
file itself is ignored by ordinary Git. Its tracked reference configuration is
`configs/models/cellvit_sam_h_x40_amp_001.yaml`.

## Historical and third-party code

`archive/experiments/` preserves numbered filenames because their order
communicates research chronology. These scripts have not been scientifically
rerun. Promote them one at a time into `experiments/` using its review policy.
`src/cpathogen/evaluation/misc/legacy_fid/` and
`workflows/05_generate_counterfactuals/misc/` serve the same preservation role
for old generator utilities.

`third_party/cellvit_plus_plus/` is a pinned upstream source snapshot used by
Workflow 01; `UPSTREAM.md` records its revision, license/citation terms, and the
single portability patch. `third_party/nuhtc/` is vendored upstream-derived
legacy/alternative code. Large NuHTC demo data is isolated under
`data/misc/nuhtc_demo/`, and its PanNuke checkpoint is under
`artifacts/models/third_party/nuhtc/`. Other upstream machine-specific examples
are deliberately not treated as CPathoGen path configuration.
