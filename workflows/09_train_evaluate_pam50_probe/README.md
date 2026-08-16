# Workflow 09: CTransPath PAM50 counterfactual probe

This workflow trains five patient-disjoint CTransPath logistic heads for
TCGA-BRCA PAM50 Basal versus Luminal A. Tile embeddings are mean-pooled and
L2-normalized within patient before head training and performance evaluation.

For a counterfactual whose source patient belongs to the binary cohort, the
workflow uses only that patient's held-out-fold head. A source patient outside
the binary cohort was absent from all five training sets, so its score is the
mean of the five heads. The prediction CSV records this choice for every row.

Run on a CUDA VM:

```bash
uv sync --extra probe
uv run python workflows/09_train_evaluate_pam50_probe/cloud_run.py --device cuda
```

Outputs include patient-level out-of-fold metrics and predictions, all five
heads, tile and patient embeddings, and tile-level counterfactual probabilities
with baseline deltas.
