# Artifacts

This directory contains migrated checkpoints, model weights, generated images, metrics, and historical results. Most legacy artifacts do not have complete sidecar provenance. Do not infer architecture, dataset, split, seed, or metric protocol from a filename alone.

- New work belongs under `runs/<run-id>/{checkpoints,models,metrics,figures}` with a run manifest.
- Shared pretrained/downstream weights may remain under `models/`.
- Third-party weights removed from vendor source live under `models/third_party/<project>/`.
- Extracted legacy checkpoints are under `runs/legacy_*/checkpoints/`.
- Checkpoint ZIP archives are cold material under `misc/checkpoint_archives/`.
- Historical generated results and metrics were placed under descriptive `runs/legacy_*` or experiment-named run directories without asserting that their provenance is complete.
