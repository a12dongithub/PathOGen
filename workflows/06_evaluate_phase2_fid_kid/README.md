# Workflow 06: Phase-2 generation and FID/KID

This workflow generates held-out tiles from the frozen Phase-2 checkpoint and
copies the matching real source tiles into the same run directory. It writes a
stem-level `manifest.json`, `generated/`, `real/`, and the first 200 legacy
three-panel spatial-map/real/generated grids by default.

Use `--all-tiles` for a complete aligned set or `--num-tiles 100 --sample-seed 42`
for a reproducible pilot. FID/KID computation should use the manifest and the
same image preprocessing for both directories; it is intentionally separate
from generation so Workflow 07 can rank control consistency before computing a
secondary subset score. It uses the same direct-concat-plus-FiLM DDIM sampler
as the historical Phase-2 validation routine: each normal evaluation tile has
its own device-seeded latent noise; only Workflow 05 deliberately shares
noise across a baseline/counterfactual pair.
