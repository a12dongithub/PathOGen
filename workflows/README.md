# Workflows

The numbered directories represent the intended end-to-end order:

1. annotate nuclei in prepared H&E tiles;
2. build spatial and morphology/stain conditions;
3. train phase 1 H&E adaptation;
4. train phase 2 structured conditioning;
5. generate counterfactuals;
6. extract image embeddings;
7. train downstream classifiers; and
8. evaluate matched counterfactuals.

Every numbered directory contains `run.py`. Implemented stages delegate to canonical modules; missing orchestration is represented by an explicit TODO/`NotImplementedError`. Checkpoint-specific historical utilities are retained under the relevant workflow's `misc/` directory.
