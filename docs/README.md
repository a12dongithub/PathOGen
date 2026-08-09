# CPathoGen experiment documentation

This branch is the counterfactual-dataset stage of CPathoGen: it turns existing
per-tile controls into matched synthetic images for later evaluation by breast
pathology classifiers. It does not train, annotate, select, or rank images.

| Document | Use it for |
|---|---|
| [Project and scope](project.md) | Goal, current assets, claim boundaries, and branch boundaries |
| [Model and data contract](model-and-data.md) | Exact inputs, architecture, checkpoint, and reproducibility requirements |
| [Experiment protocol](experiment-protocol.md) | Control-knob design, counterfactual datasets, and downstream evaluation plan |
| [Generation workflow](generation-workflow.md) | Running Workflow 05 and understanding its output manifests |

The repository is intentionally small. Images, condition data, checkpoints,
and generated results remain local or in approved object storage; they are not
versioned in Git.
