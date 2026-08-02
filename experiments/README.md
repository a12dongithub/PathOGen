# Counterfactual experiments

Active experiments contain only Python definitions of how the two Phase-2
controls are transformed. Reusable model and data code lives in `src/cpathogen/`;
checkpoint loading, matched-noise generation, and artifact writing live in
`workflows/05_generate_counterfactuals/`.

Each module exposes:

```python
def build_interventions() -> list[ConditionIntervention]:
    ...
```

An intervention subclasses `ConditionIntervention` and overrides
`modify_spatial(...)`, `modify_morphology(...)`, or both. The inherited method is
an identity transform. The workflow supplies cloned tensors and read-only access
to the aligned condition store, allowing donor swaps and other changes without
creating a new GeoJSON, spatial-map, or morphology file for every experiment.

## Promoted historical control experiments

| Active module | Historical behavior represented |
|---|---|
| `spatial.full_plane_categories` | Entire-plane neoplastic/inflammatory/connective maps (04) |
| `spatial.relabel_all_cells` | Relabel the original cell envelope (05/07) |
| `spatial.rotate_maps` | Spatial rotations by 90/180/270 degrees (08) |
| `spatial.donor_maps` | Twenty random spatial donor swaps (10) |
| `spatial.tissue_to_inflammatory` | Progressive neoplastic/epithelial-to-inflammatory shifts (20/21) |
| `morphology.full_feature_sweep` | All 16 features set to −2…+2 standard deviations (22/23) |
| `morphology.shape_variance_sweep` | Area/perimeter variance sweep (30) |
| `morphology.stain_variance_sweep` | RGB variance sweep (27) |
| `morphology.donor_vectors` | Twenty random morphology donor swaps (11/17/19) |
| `joint.legacy_morphology_rotation` | Donor swaps plus rotations from experiment 08 |
| `joint.donor_grid` | Four morphology donors by five spatial donors (16) |

Classifier training, plotting, feature extraction, and representation audits stay
in `archive/experiments/`. The historical “noise invariance” probe repeats the
same controls with different diffusion noise, so it is a sampling audit rather
than a control intervention.
