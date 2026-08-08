# Workflows

The active repository contains the five stages needed for annotation,
condition construction, frozen-checkpoint generation, and evaluation:

1. `01_annotate_nuclei` — implemented;
2. `02_build_conditions` — implemented;
3. `05_generate_counterfactuals` — matched counterfactual generation;
4. `06_evaluate_phase2_fid_kid` — held-out generation, grids, FID, and KID;
   and
5. `07_rank_control_consistency` — CellViT++ control-consistency ranking.

Each entry point delegates reusable work to `src/cpathogen/`.
Workflow 05 loads intervention plugins from `experiments/` and transforms
conditions in memory. Workflow 01 can be run again on Workflow 05's
`pairs.jsonl` to annotate generated baselines and counterfactuals.

Historical training/evaluation scripts are not supported interfaces. The
frozen direct-concat-plus-FiLM checkpoint is the only supported inference
architecture. Operational details belong in each workflow's README.
