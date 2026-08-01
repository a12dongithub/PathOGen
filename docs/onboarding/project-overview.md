# Project overview

## Research question

Pathology foundation models can be accurate while still relying on stain, texture, scanner artifacts, or other shortcuts. CPathoGen asks whether controlled, realistic counterfactual H&E images can reveal which cellular and morphological signals a model actually uses.

The central research claim proposed in the [paper experiment registry](../research/paper-experiments.md) is:

> CPathoGen generates realistic, controllable H&E counterfactuals that preserve requested cellular spatial structure and morphology, enabling causal probing of pathology vision models.

That claim has three separable requirements:

1. **Realism** - generated images should resemble held-out H&E images.
2. **Intervention fidelity** - requested spatial and morphology changes should appear accurately and selectively.
3. **Probing utility** - the counterfactuals should expose meaningful differences in representation or prediction sensitivity.

The current repository contains preliminary evidence and extensive experiment code, but it does not yet contain the complete validation needed to establish all three requirements.

## Intended end-to-end use

Given an H&E tile or a desired cell layout, CPathoGen should make a synthetic tile while allowing the researcher to change one controlled factor, for example:

- increase or decrease a cell type in a region;
- change tumor-immune mixing or separation;
- alter nuclear area, perimeter, eccentricity, or solidity;
- change gradient/texture or RGB-derived stain statistics;
- keep conditions fixed while changing the diffusion seed.

The generated images are then passed through pathology encoders and downstream classifiers. Comparing embeddings, class probabilities, prediction flips, and layer activations across matched counterfactuals provides a controlled stress test.

## Major project components

### 1. Weak cellular annotation

The abstract and posters describe CellViT++ processing of TCGA-BRCA slides to obtain nucleus positions and cell types. The full CellViT++ source and the complete annotation-generation pipeline are not present in this workspace. A separate bundled NuHTC project and PanNuke folds appear to be a legacy or alternative segmentation strand.

Expected annotation output is per-tile GeoJSON containing nucleus polygons and a `properties.classification.name` cell-type label.

### 2. Conditional H&E generator

The active generator source lives in `src/cpathogen/generation/`, with preprocessing under `src/cpathogen/preprocessing/`. It is based on `Manojb/stable-diffusion-2-1-base` and uses two stages:

- phase 1 adapts the UNet to H&E texture using the constant prompt `"he"`;
- phase 2 conditions the model on a five-channel spatial map and a 16-value standardized morphology/stain vector.

Current phase-2 code uses direct latent concatenation for spatial conditioning and FiLM for the 16-value vector. See [Architecture](../architecture/system.md).

### 3. Foundation-model and downstream probing

`experiments/` contains the grouped counterfactual experiment records. The main model families represented are:

- UNI2-h;
- CTransPath;
- ResNet50 as a conventional baseline;
- logistic regression, random forest, XGBoost, and MLP downstream heads;
- experimental activation predictors, spatial adapters, and LoRA variants.

Current label/task strands include four-class BACH classification and TCGA-BRCA molecular subtype classification (Basal-like, HER2-enriched, and Luminal in the poster-era benchmark).

### 4. Evaluation and paper planning

The stored scripts and result folders cover FID, visual checkpoint comparison, classifier benchmarks, color/morphology/noise invariance, cellular spatial shifts, forward-propagation audits, layer sweeps, LoRA ablations, and embedding manifolds.

The [paper experiment registry](../research/paper-experiments.md) is the forward-looking plan. It expands the evaluation to pathology-feature distances, re-segmentation-based fidelity, intervention crosstalk, seed robustness, sample supervision, patient-level statistics, XAI faithfulness, and possible cross-domain validation.

## Data in the workspace

| Dataset/artifact | Location | Observed role |
|---|---|---|
| TCGA-BRCA tiles/annotations | Complete training copy not present | Main CPathoGen training and molecular-subtype experiments; poster states 1,114 slides and about 1.038 million tiles |
| PanNuke Parquet folds | `data/raw/pannuke/` | Five-class nucleus instance-segmentation data for the segmentation strand |
| BACH source images | `data/raw/bach/images/` | Four-class breast histology classification: Normal, Benign, InSitu, Invasive |
| BACH 512 tiles | `data/interim/tiles/bach/` | Derived inputs for feature extraction/classification |
| TCGA cached tensors | `data/misc/tcga_10k_cached_tensors/` | Provenance-poor legacy features/activations; not canonical generator inputs |
| Generator checkpoints | `artifacts/runs/<run>/checkpoints/` | Phase-1/phase-2 training state and evaluation inputs |

Do not infer dataset splits from directory names alone. Several scripts create splits internally, and some validation code falls back to a training subset when a validation directory is unavailable.

## Preliminary artifacts

The most recent visual narrative appears under `reports/posters/`, including `poster.jpeg` and `CVPR Workshop ONLY Poster.pdf`. They describe:

- controllable cellular spatial structure and tumor morphology;
- approximate generator FID of 56;
- molecular-subtype benchmarks across UNI2-h, CTransPath, and ResNet50;
- high counterfactual sensitivity even for the strongest predictive encoder.

The older `reports/manuscripts/abstract.docx` records phase-1 FID around 62 and early conditional FID around 102 at 15k steps. A separate stored ControlNet evaluation reports FID 480.8966. These numbers belong to different checkpoints/architectures or stages and must not be merged into a single performance claim.

## What is in scope

- controllable histopathology image generation;
- generator realism and conditioning fidelity;
- counterfactual probing of pathology representations and downstream predictions;
- evaluation methodology, failure analysis, and research reproducibility.

## What is not established

- clinical safety or diagnostic validity;
- causal biological conclusions without verified intervention fidelity and confounder control;
- a general-purpose text-to-image model;
- a fully reproducible software release;
- ownership/licensing of the complete combined workspace under one project license.

## Names and versions

Use **CPathoGen** for the current project/paper. Use **PathOGen** when referring to the generator repository, code symbols, or historical checkpoints. When reporting a result, always record the conditioning architecture explicitly: phase-1 LDM, legacy ControlNet, or direct-concat spatial encoder plus FiLM.
