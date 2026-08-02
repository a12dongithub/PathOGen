# Workflows

The active repository contains only the five stages needed for annotation,
condition construction, training, and counterfactual generation:

1. `01_annotate_nuclei` — implemented;
2. `02_build_conditions` — implemented;
3. `03_train_phase1` — implemented Phase-1 H&E domain adaptation;
4. `04_train_phase2` — implemented direct-concat-plus-FiLM training; and
5. `05_generate_counterfactuals` — implemented.

Each entry point delegates reusable work to `src/cpathogen/`.
Workflow 05 loads intervention plugins from `experiments/` and transforms
conditions in memory. Workflow 01 can be run again on Workflow 05's
`pairs.jsonl` to annotate generated baselines and counterfactuals.

Historical training/evaluation scripts are archived and are not supported
interfaces. The new training workflows retain the established architecture and
hyperparameter references without carrying forward obsolete ControlNet/cloud
options. Operational details belong in each workflow's README.
