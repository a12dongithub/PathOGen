# Experiment protocol

## Study design

Each control knob should represent one interpretable tissue property and have a
prespecified transform of the spatial map, the morphology/stain vector, or
both. Level 0 is the original condition. Levels 1–3 (or 1–4) apply fixed,
declared increments, such as 10%, 20%, and 30%; the chosen scale must be valid
for that feature and recorded in the manifest.

Use the same selected 1,000 source cases, fixed seed bank, level definitions,
and acceptance policy for every knob and every downstream model.

## Proposed knob families

Finalize only 5–10 after checking that the generator produces recoverable,
selective changes. Good candidates are:

| Family | Biological interpretation | Spatial transform | Vector transform |
|---|---|---|---|
| Neoplastic density | Tumor nuclear burden | Scale neoplastic map | No change unless separately justified |
| Inflammatory density | Immune infiltration | Scale inflammatory map | No change |
| Tumor–immune balance | Microenvironment composition | Reciprocal neoplastic/inflammatory changes | No change |
| Stromal density | Connective-tissue content | Scale connective map | No change |
| Spatial dispersion/mixing | Tissue organization | Blur, relocate, or mix map mass | No change |
| Nuclear size | Nuclear enlargement | No change | Adjust `area_mean` and, when justified, `area_var` |
| Nuclear shape irregularity | Pleomorphism proxy | No change | Adjust eccentricity, solidity, and/or perimeter with a declared rule |
| Nuclear heterogeneity | Within-tile variation | No change | Adjust variance features only |
| Stain intensity | H&E / scanner-style proxy | No change | Adjust RGB means; report as stain, not morphology |
| Stain heterogeneity | Color variation | No change | Adjust RGB variances |

Do not imply that every knob is independent. For example, nuclear area and
perimeter are correlated, and density changes can affect recovered morphology.
Measure all recovered controls for every intervention and report crosstalk.

## Dataset generation

For every `(tile, knob, level, seed)`:

1. Load the original spatial map and standardized vector.
2. Apply only the declared transformation in memory.
3. Generate baseline and counterfactual from cloned, identical initial noise.
4. Write images plus a `pairs.jsonl` row containing source stem, seed,
   intervention parameters, and output paths.
5. Keep failures and acceptance/rejection reasons; never filter on a desired
   classifier response.

Run `--dry-run` before GPU generation. Use a new output directory for each
immutable run. A realistic initial estimate is `1,000 × 5–10 knobs × 3–4
non-baseline levels × seed-count` counterfactual images, plus matching
baselines; decide whether baselines are shared across knobs before generating.

## Downstream evaluation

Select models before looking at results. Prefer breast-cancer task models that
accept native or resized tiles at 512 pixels or below, and record architecture,
weights, dataset, preprocessing, label definition, and output semantics.

For each model, report paired baseline-versus-level changes, confidence
intervals clustered by patient/slide where available, dose response, failure
cases, and multiplicity correction across knobs/models. Keep an identity/null
control, a seed-variation control, and a shuffled or donor-condition control.
