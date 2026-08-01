# CPathoGen

CPathoGen is a research prototype for generating controllable 512 x 512 H&E histopathology tiles and using them as counterfactuals to probe pathology vision models. The intended controls are cellular spatial organization and tile-level nuclear morphology/stain statistics.

> **Research status:** this repository is an experimental workspace, not a packaged library or clinical system. It contains source code, copied third-party projects, datasets, checkpoints, generated outputs, archives, posters, and administrative files. Read [Known issues](docs/reference/known-issues.md) before running or citing results.

## What the project does

1. Tile TCGA-BRCA whole-slide images into H&E patches.
2. Obtain weak nucleus locations and cell types from a segmentation/classification system.
3. Convert those annotations into five-channel cellular spatial maps and a 16-value morphology/stain vector.
4. Adapt a Stable Diffusion 2.1 backbone to H&E in a first training phase.
5. Add spatial and morphology conditioning in a second phase.
6. Generate controlled counterfactual tiles.
7. Measure how foundation encoders and downstream breast-cancer classifiers react to the controlled changes.

The current training code implements spatial conditioning with a small CNN whose output is concatenated with the noisy diffusion latents, plus FiLM modulation for the 16-value vector. Some posters and older artifacts call the spatial component “ControlNet”; that naming does not match the current `phase2.py` implementation. See [Architecture: implementation versus research narrative](docs/architecture/system.md#implementation-versus-research-narrative).

## Start here

| Document | Purpose |
|---|---|
| [Documentation index](docs/README.md) | Reading order and document ownership |
| [Background for engineers](docs/onboarding/background.md) | H&E, pathology vocabulary, foundation models, datasets, metrics, and responsible interpretation |
| [Project overview](docs/onboarding/project-overview.md) | Research question, scope, data, outputs, and current status |
| [Architecture](docs/architecture/system.md) | End-to-end pipeline, tensor shapes, model phases, and inference |
| [Repository map](docs/reference/repository-map.md) | What every major folder and script family means |
| [Setup and workflows](docs/guides/setup-and-workflows.md) | Environment, expected data layout, preprocessing, training, and evaluation |
| [Known issues](docs/reference/known-issues.md) | Reproducibility gaps, version conflicts, hard-coded paths, and risks |
| [Paper experiment registry](docs/research/paper-experiments.md) | Planned generator, probing, statistical, and paper experiments |

## Repository at a glance

| Path | Role | Treat as |
|---|---|---|
| `src/cpathogen/` | Generator preprocessing/training/evaluation and encoder modules | Canonical source |
| `workflows/` | Numbered operational entry points | Intended workflow layer; incomplete |
| `experiments/` | Foundation-model benchmarks, counterfactual probes, and representation audits | Historical experiment records; paths now use the common layout but remain lightly validated |
| `data/` | PanNuke, BACH, TCGA intermediate tensors, features, and small manifests | Local data; main TCGA generator cohort remains incomplete |
| `artifacts/` | Checkpoints, fitted models, generated images, metrics, and historical results | Outputs with incomplete provenance |
| `third_party/nuhtc/` | Bundled NuHTC/MMDetection nuclei-segmentation project | Third-party/legacy segmentation code |
| `reports/` | Posters, figures, and manuscripts | Narrative/output material |
| `archive/` | Duplicate source, administrative records, caches, and ZIP snapshots | Non-canonical cold storage |

The workspace was approximately 191 GB when documented on 2026-07-26. About 92 GB is now isolated under `archive/workspace_snapshots/`, with other large datasets and model artifacts separated from source.

## Current evidence and claims

- The CVPR 2026 poster describes training on roughly one million TCGA-BRCA tiles from 1,114 slides and reports an approximate FID of 56 for the adapted generator.
- The older abstract records preliminary FID values of about 62 for the phase-1 LDM and about 102 for early conditional training at 15k steps.
- A stored legacy ControlNet evaluation reports FID 480.8966 and should not be confused with the later concat-conditioned checkpoint or the poster result.
- Downstream artifacts compare UNI2-h, CTransPath, and ResNet50 representations/classifiers and contain color, morphology, noise, spatial, and layerwise sensitivity experiments.

These are preliminary research artifacts, not independently reproduced benchmarks. The recommended validation plan is in the [paper experiment registry](docs/research/paper-experiments.md).

## Reproducibility warning

There is not yet a clean one-command local setup. The generator cloud script pins its main packages, but the full workspace has no root dependency lockfile. The refactored phase-2 entry point accepts explicit tile, spatial-map, and morphology paths and preprocessing now writes a scaler, but the complete million-tile training dataset and CellViT++ implementation are still absent. Follow [Setup and workflows](docs/guides/setup-and-workflows.md) and resolve the remaining blockers in [Known issues](docs/reference/known-issues.md) before launching a costly run.

## Responsible use

CPathoGen is for research into generative pathology, robustness, and model explanation. Generated images are synthetic and must not be used for diagnosis, treatment decisions, or clinical validation without an appropriately designed study and expert review. Preserve TCGA and other dataset licenses/terms, avoid publishing patient-linked metadata, and report failed generations and filtering criteria alongside successful examples.

## License and citation

The workspace does not currently contain a project-level license or a formal CPathoGen citation. Individual components and datasets have their own terms, including the bundled NuHTC/MMDetection licenses and the PanNuke dataset card. Add a root license, third-party notices, data-use statement, model-card information, and a citation file before public release.
