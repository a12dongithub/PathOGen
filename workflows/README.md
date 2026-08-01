# Workflows

The numbered directories represent the intended end-to-end order:

1. tile slides;
2. annotate nuclei;
3. build spatial and morphology/stain conditions;
4. train phase 1 H&E adaptation;
5. train phase 2 structured conditioning;
6. generate counterfactuals;
7. extract image embeddings;
8. train downstream classifiers; and
9. evaluate matched counterfactuals.

Every numbered directory contains `run.py`. Implemented stages delegate to canonical modules; missing orchestration is represented by an explicit TODO/`NotImplementedError`. Checkpoint-specific historical utilities are retained under the relevant workflow's `misc/` directory.
