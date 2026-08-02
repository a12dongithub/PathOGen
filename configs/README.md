# Active configuration

Only portable contracts used by active workflows remain here:

- `data/default.yaml` — tile, annotation, and condition paths plus channel/feature order;
- `models/cellvit_sam_h_x40_amp_001.yaml` — Workflow 01 model identity;
- `models/phase2_concat_film.yaml` — Workflow 05 architecture contract; and
- `training/phase1.yaml` — historical-scale Phase-1 optimizer/schedule defaults;
- `training/phase2.yaml` — historical-scale Phase-2 optimizer/schedule defaults; and
- `experiments/` — conventions for Workflow 05 intervention plugins.

The training YAML files point at the self-contained local fixtures but retain
long-run hyperparameters. Replace the six-tile metadata/input directories with
a leakage-controlled training cohort before optimization. The pre-refactor
references remain under `archive/configs/training/` for provenance.
