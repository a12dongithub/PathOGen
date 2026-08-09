# Project and scope

## Goal

CPathoGen investigates whether controlled edits to H&E tiles change pathology
model representations or predictions. The experiment creates matched baseline
and counterfactual images from the same source-tile controls and the same
initial diffusion noise.

This supports a sensitivity study, not a clinical claim. A changed classifier
prediction is not proof of biological causality.

## Current status

- A frozen Phase-2 direct-concat-plus-FiLM generator and its local checkpoint
  already exist.
- A selected set of 1,000 high-quality generated images already exists outside
  Git.
- The next deliverable is a counterfactual dataset for a small, prespecified
  set of biologically motivated controls and levels.
- After dataset generation, the same cases and manifests will be evaluated by
  multiple breast-pathology classifiers.

## This branch

Included: condition loading, in-memory interventions, matched-noise generation,
and manifests.

Excluded: diffusion training, CellViT++ inference, construction of maps and
features, FID/KID, sample re-ranking, notebooks, historical code, and
classifier implementations. Add only the experiment definitions and downstream
evaluation code that are needed for the new study.

## Claim boundaries

- The five cellular labels are model-derived broad nucleus categories, not
  expert ground truth or precise cell lineages.
- RGB statistics and Sobel-gradient features are stain/texture proxies as well
  as visual signals; they are not purely biological morphology.
- Shared noise reduces sampling variation but cannot guarantee that an
  intervention changed only one visual factor.
- Generated images must not be used for diagnosis, treatment, or clinical
  validation.
