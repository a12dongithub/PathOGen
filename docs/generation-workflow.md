# Generation workflow

Workflow 05 is the only supported executable in this branch. It loads a frozen
checkpoint and Python-defined interventions, then writes matched baseline and
counterfactual images.

## Intervention interface

An experiment module exposes:

```python
def build_interventions() -> list[ConditionIntervention]:
    ...
```

Each intervention subclasses `ConditionIntervention` and overrides
`modify_spatial(spatial, context)`, `modify_morphology(morphology, context)`,
or both. It must return finite tensors that retain shapes `(5,H,W)` and `(16,)`.
The framework supplies cloned tensors, deterministic intervention randomness,
and read-only access to aligned donor conditions.

## Commands

```bash
pip install -e ".[inference]"
python workflows/05_generate_counterfactuals/run.py \
  --experiment experiments.<module> --list-interventions
```

Validate a full run without loading the diffusion model:

```bash
python workflows/05_generate_counterfactuals/run.py \
  --experiment experiments.<module> --all-tiles --dry-run \
  --output-dir data/evaluations/<run-name>-dry-run
```

Generate a complete matched dataset:

```bash
python workflows/05_generate_counterfactuals/run.py \
  --experiment experiments.<module> --all-tiles --seed 42 \
  --steps 30 --spatial-strength 2 --batch-size 1 \
  --device cuda --dtype float16 --local-files-only \
  --output-dir data/evaluations/<run-name>
```

## Output

```text
<run-name>/
├── run_manifest.json       # checkpoint, environment, arguments, interventions
├── pairs.jsonl             # one baseline/counterfactual record per pair
└── images/<stem>/seed_<seed>/
    ├── baseline.png
    └── <intervention-slug>.png
```

Baselines and counterfactuals in one pair share the initial latent. Normal
evaluation across different tiles may use independent seeds. Record all seed,
checkpoint, preprocessing, and code revisions in the run manifest.
