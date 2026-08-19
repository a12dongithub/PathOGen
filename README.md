# CPathoGen experiment pipeline

Lean codebase for generating matched H&E counterfactual datasets from the
frozen direct-concat-plus-FiLM Phase-2 checkpoint.

Start with [project documentation](docs/README.md).

Included:

- loading aligned five-channel spatial maps and standardized 16-value vectors;
- defining in-memory experimental interventions;
- matched-noise baseline/counterfactual sampling; and
- run and pair manifests for downstream classifier experiments;
- patient-disjoint PAM50 and censor-aware survival probes for UNI2-h,
  CTransPath, Virchow2, and ResNet50; and
- transfer evaluation adapters for PathLUPI, OTSurv, and CPMP, including
  counterfactual JSONL outputs.

Excluded: foundation-model pretraining, CellViT++ annotation, condition
building, FID/KID, candidate ranking, notebooks, historical experiments, and
vendored third-party repositories.

See
[`workflows/10_train_evaluate_endpoint_models/README.md`](workflows/10_train_evaluate_endpoint_models/README.md)
for the endpoint and XAI evaluation protocol.

## Required local assets

Assets are deliberately untracked. Supply their paths explicitly, or use the
default layout:

```text
data/images/<stem>.png                 # optional but recommended for provenance
data/spatial_maps/<stem>.npz           # required key: map, shape: (512, 512, 5)
data/morphology_stats.parquet          # required 16 standardized features
models/pathogen_phase2/checkpoint_30000/
```

## Run

```bash
pip install -e ".[inference]"
python workflows/05_generate_counterfactuals/run.py \
  --experiment path.to.experiment \
  --intervention <slug> \
  --all-tiles --seed 42 --device cuda --dtype float16 --local-files-only \
  --output-dir data/evaluations/<run-name>
```

An experiment is a Python module (or `.py` file) exposing
`build_interventions()`. New biological control knobs belong in a future
`experiments/` package and must only transform the supplied spatial map and/or
morphology vector.
