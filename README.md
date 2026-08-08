# CPathoGen

CPathoGen is a research prototype for generating controllable 512 × 512 H&E
histopathology tiles and using matched counterfactuals to study pathology vision
models. It is not a clinical system.

## Active scope

Five workflows are active:

| Workflow | Purpose | Status |
|---|---|---|
| `01_annotate_nuclei` | Run CellViT++ on source tiles or generated matched pairs and write nucleus GeoJSON | Implemented and tested on Apple MPS |
| `02_build_conditions` | Build five spatial maps, 16 morphology/stain values, scaler, and metadata | Implemented and tested on the six-tile fixture |
| `05_generate_counterfactuals` | Apply Python interventions in memory and generate matched baseline/counterfactual images | Implemented and tested on Apple MPS |
| `06_evaluate_phase2_fid_kid` | Generate held-out Phase-2 tiles and calculate FID/KID against real sources | Implemented |
| `07_rank_control_consistency` | Re-annotate generated/baseline tiles and rank control agreement | Implemented |

Training is intentionally out of scope for this repository revision. The
historical scripts remain in the main branch/archive as provenance; active
workflows consume frozen checkpoints only.

```text
prepared H&E tiles
    → 01 CellViT++ nucleus GeoJSON
    → 02 spatial + morphology/stain conditions
    → 05 matched counterfactual generation
    → 01 repeat CellViT++ annotation on generated images
```

The repeated annotation produces measurements and pair provenance; it does not
by itself establish counterfactual fidelity or biological causality.

## Documentation

The project documentation is deliberately independent of repository mechanics:

| Document | Contents |
|---|---|
| [Project](docs/PROJECT.md) | Scientific question, pathology background, terminology, models, datasets, metrics, and preliminary evidence |
| [Method](docs/METHOD.md) | Annotation, condition construction, direct-concat-plus-FiLM generator, tensor contracts, inference, and evaluation model |
| [Experiments](docs/EXPERIMENTS.md) | Complete generator/probing plan, baselines, statistical protocol, paper plan, and run template |
| [Limitations](docs/LIMITATIONS.md) | Claim boundaries, unresolved validity/reproducibility issues, licenses, and reporting rules |

Operational commands and file layouts live beside each workflow rather than in
the scientific documentation.

## Active repository layout

```text
refactored/
├── src/cpathogen/
│   ├── annotation/              # Workflow 01 adapter and GeoJSON contract
│   ├── preprocessing/           # Workflow 02 condition builders
│   ├── counterfactuals/         # Workflow 05 condition/intervention contracts
│   ├── generation/              # Workflow 05 checkpoint loading and sampling
│   │   └── conditioning/        # spatial encoder and FiLM compatibility
│   └── utils/paths.py           # shared repository-relative paths
├── workflows/
│   ├── 01_annotate_nuclei/
│   ├── 02_build_conditions/
│   ├── 05_generate_counterfactuals/
│   ├── 06_evaluate_phase2_fid_kid/
│   └── 07_rank_control_consistency/
├── experiments/                 # active Workflow 05 intervention plugins
├── configs/                     # active data/model/experiment contracts
├── data/                        # six aligned TCGA tiles, GeoJSON, and conditions
├── artifacts/
│   ├── models/                  # CellViT++, Phase-1, and Phase-2 checkpoints
│   └── runs/                    # generation/evaluation outputs
├── tests/                       # active unit and checkpoint integration checks
├── third_party/cellvit_plus_plus/
├── docs/                        # four project documents plus report assets
└── archive/                     # historical code, data, models, results, and docs
```

## Installation

Use separate optional dependency groups:

```bash
pip install -e ".[annotation]"      # Workflow 01
pip install -e ".[preprocessing]"   # Workflow 02
pip install -e ".[inference]"       # Workflow 05
```

Run commands from the repository root. Detailed examples are in the workflow
READMEs:

- [Workflow 01](workflows/01_annotate_nuclei/README.md)
- [Workflow 02](workflows/02_build_conditions/README.md)
- [Workflow 05](workflows/05_generate_counterfactuals/README.md)
- [Workflow 06](workflows/06_evaluate_phase2_fid_kid/README.md)
- [Workflow 07](workflows/07_rank_control_consistency/README.md)

## Models and local fixture

The active repository expects:

```text
artifacts/models/cellvit_plus_plus/cellvit_sam_h_x40_amp_001/model.pth
artifacts/models/pathogen_phase1/checkpoint_30000/unet/
artifacts/models/pathogen_phase2/checkpoint_30000/
data/interim/tiles/tcga_brca/
data/interim/annotations/tcga_brca/geojson/
data/processed/conditions/
```

The included six-tile fixture is for integration checks. It is not a
scientific validation cohort. Large
model files are ignored by ordinary Git and require an approved model registry
or Git LFS for distribution.

## Archive policy

Substantive historical material was preserved by category under `archive/`.
Unimplemented placeholder modules and regenerable caches were deleted. Archived
code and results are not supported entry points and may have incomplete
provenance. Large archived data/models remain ignored by Git.

## Responsible use, license, and citation

Synthetic images and model outputs must not be used for diagnosis or treatment.
The project does not yet have a finalized root license or citation file. Every
dataset, checkpoint, and third-party component retains its own terms. In
particular, review the CellViT++ license and mandatory CellViT/CellViT++ citation
requirements before use or redistribution.
