# PathOGen Paper Experiment Registry

## 1. Central paper claim

PathOGen generates realistic, controllable H&E counterfactuals that preserve requested cellular spatial structure and morphology, enabling causal probing of pathology vision models.

The paper must establish three distinct properties:

1. **Image realism:** generated patches resemble real H&E images.
2. **Intervention fidelity:** requested spatial and morphological edits occur accurately and selectively.
3. **XAI utility:** controlled counterfactuals reveal meaningful differences in model sensitivity and downstream predictions.

The core paper should stay focused enough for the WACV eight-page limit. Secondary diagnostics, extensive seed studies, and expanded model results belong in supplementary material.

---

## 2. Experimental principles

- Split data at the patient or slide level before fitting preprocessing artifacts.
- Fit morphology and stain transformations on the training split only.
- Save the scaler, feature order, clipping thresholds, preprocessing version, and split hash.
- Use fixed held-out cases and seed lists across generator/model comparisons.
- Report bootstrap 95% confidence intervals at the patient or slide level.
- Report both unfiltered generations and supervisor-filtered generations.
- Never select seeds using the desired downstream prediction change.
- Include null, shuffled-condition, and matched-random intervention controls.
- Distinguish patch encoders from task-trained diagnostic, MIL, and survival systems.

---

## 3. Generator evaluations

### G1 — Distributional realism and reconstruction ceiling — Main

**Questions**

- How close are generated images to held-out real H&E patches?
- How much quality is already lost by the VAE before diffusion sampling?

**Metrics**

- FID on at least 10,000 generated and held-out real images.
- KID with confidence intervals.
- Precision and recall for generative models.
- LPIPS and SSIM for VAE reconstruction analysis where paired comparisons apply.
- Pathology-feature Fréchet/KID distances using UNI2-h, CONCH, CTransPath, and Virchow2 features.

**Comparisons**

- Real held-out images versus real held-out images as a metric floor.
- VAE reconstructions versus real images.
- Unconditional or weakly conditioned baseline, if available.
- Current checkpoint versus corrected-preprocessing/H200 checkpoint.
- Raw weights versus EMA weights.

**Engineering diagnostic folded into G1**

Check blur, artifacts, blank tissue, stain range, saturation, compression, duplicate tiles, and scanner/site imbalance. This is not a separate headline experiment.

### G2 — No separate headline experiment

Former low-level image-quality checks are incorporated into G1 and the data-quality audit.

### G3 — Spatial fidelity — Main

Reapply CellViT++ and, where feasible, a second independent nucleus detector/classifier to generated images.

**Metrics**

- Cell-count correlation and mean absolute error per cell type.
- Density correlation and calibration per class.
- Detection precision, recall, and F1.
- Cell-type confusion matrix and macro-F1.
- Centroid matching distance using optimal or nearest-neighbor matching.
- Spatial Earth Mover's Distance or Wasserstein distance.
- Pair-correlation or Ripley's K agreement.
- Tumor–immune nearest-neighbor distance error.
- Tumor–immune density and mixing-index error.
- Spatial autocorrelation agreement, such as Moran's I.

**Controls**

- Correct spatial map.
- Shuffled spatial map.
- Zero spatial map.
- Spatial map from another patient/slide.
- Morphology held fixed while spatial structure changes.

### G4 — Morphological and stain fidelity — Main

Re-segment generated nuclei and recompute the features used for conditioning.

**Morphology metrics**

- Pearson and Spearman correlation between requested and recovered features.
- MAE and normalized MAE.
- Area, perimeter, eccentricity, solidity, major/minor axis, circularity, and nuclear-size distribution.
- Feature-distribution Wasserstein distance.
- Cell-type-specific morphology where labels support it.
- Monotonic dose-response under controlled feature sweeps.

**Color/stain metrics**

- RGB mean and variance correlations as diagnostics.
- H&E optical-density/stain-vector agreement as the preferred biological stain analysis.
- Hematoxylin/eosin concentration statistics.
- Delta E or another perceptual color difference metric.

**Intervention-crosstalk matrix**

For each edited condition, measure all recovered spatial, morphology, stain, and identity features. The diagonal should be strong and off-diagonal changes should be small. This is a key result for showing that controls are selective rather than entangled.

### G5 — Layerwise conditioning analysis — Supplementary

- Disable or scale FiLM at selected UNet blocks.
- Measure where morphology/stain changes emerge.
- Compare early, middle, and late block sensitivity.
- Report realism and fidelity changes, not only qualitative examples.

### G6 — Seed robustness — Supplementary

Seed changes are expected to alter texture and fine appearance. The relevant claim is that spatial and morphology fidelity remain stable.

- Use a fixed seed bank per condition.
- Report variance of G1, G3, and G4 metrics across seeds.
- Report within-condition versus between-condition variance.
- Show a small qualitative seed grid.

### G7 — Sample-quality supervisor and rejection system — Main engineering component

The supervisor removes weak generations using realism and condition-adherence criteria. It must not use the downstream model prediction or whether the desired prediction flip occurred.

**Stage 1: deterministic gates**

- Blank/background fraction.
- Blur and high-frequency artifact score.
- Color/stain out-of-range checks.
- Cell-count and spatial-adherence thresholds.
- Morphology-adherence thresholds.
- Optional identity/content-preservation threshold.

**Stage 2: learned supervisor, only if deterministic gates are insufficient**

Possible inputs:

- UNI2-h, CONCH, or Virchow2 image features.
- Requested-versus-recovered spatial differences.
- Requested-versus-recovered morphology differences.
- Stain/quality features.

Possible heads:

- Realism probability.
- Spatial-adherence score.
- Morphology-adherence score.
- Stain-quality score.
- Identity-preservation score.

**Evaluation**

- No filtering versus deterministic gates versus learned supervisor versus combined system.
- Acceptance rate and compute per accepted sample.
- Coverage–quality curves.
- Calibration and AUROC/AUPRC against expert or carefully curated quality labels.
- G1/G3/G4 before and after filtering.
- Fixed generation budget per input.
- Report all unfiltered results alongside filtered results.

---

## 4. RGB/scaler investigation before H200 retraining

The current model sometimes produces incorrect coloration, while manually adjusting RGB controls can recover plausible H&E color. This suggests a conditioning/preprocessing problem rather than only insufficient training.

### Required checks

- Recover and save the exact original StandardScaler if possible.
- Confirm that UI/inference feature order exactly matches training feature order.
- Confirm all inference values are transformed rather than passed raw.
- Fit all new scalers on training slides only.
- Plot raw and standardized distributions by feature, slide, scanner, and site.
- Inspect extreme RGB mean/variance rows and their source images.
- Measure the percentage of values with `|z| > 3`, `4`, `5`, and `8`.
- Log FiLM scale/shift distributions and clamp/saturation rates.
- Test whether RGB-variance outliers trigger global color failures.
- Verify that the same preprocessing artifact is used for train, validation, test, UI, and generation.

### Diagnostic experiments with the existing checkpoint

- Original conditions.
- RGB z-score clipping to `[-3, 3]` and `[-4, 4]`.
- RGB features replaced by the training median.
- RGB mean edited independently from RGB variance.
- Morphology-only controls with RGB features neutralized.
- Stain-reference controls, if available.

These are diagnostic interventions; changing the feature representation or scaler generally requires retraining.

### Next-model design

Separate conditions into:

1. **Geometry/morphology:** area, eccentricity, solidity, axes, perimeter, counts, and density.
2. **Cell-type-specific morphology:** tumor, immune, stromal, epithelial, and necrotic where sufficiently represented.
3. **Stain/style:** H&E optical-density vectors, stain concentrations, or a reference-style embedding.

Use log transforms or robust scaling for heavy-tailed positive features. Do not present raw RGB variance as biological morphology.

---

## 5. H200 retraining plan

Do not spend the H200 run only extending the existing potentially broken preprocessing setup.

### Recommended staged approach

1. Run a short throughput and overfit test on a small subset.
2. Validate corrected scaler serialization and round-trip inference.
3. Train a short corrected-preprocessing ablation.
4. Compare G1/G3/G4 and color-failure rate against the current checkpoint.
5. Launch the full H200 run only after the corrected run improves or preserves control fidelity.

### Training improvements to test

- BF16 mixed precision.
- Phase-2 EMA.
- Min-SNR loss weighting, beginning with gamma 5.
- Condition dropout independently for spatial and morphology/stain controls.
- Multi-scale spatial injection or ControlNet/adapter residual conditioning.
- A histology-fine-tuned VAE if the reconstruction ceiling is poor.
- Paired geometric augmentation for H&E and spatial maps.
- Controlled stain augmentation that does not alter morphology targets.
- Balanced sampling for rare cell classes and rare morphology regimes.

### H200 experiment matrix

Keep the matrix small enough to afford full evaluations:

- Current configuration reproduced on corrected split.
- Corrected scaler/features.
- Corrected features + EMA + Min-SNR.
- Best configuration + improved spatial conditioning, if compute permits.

---

## 6. Model panel

### 6.1 Foundation and representation encoders

Use these to measure feature sensitivity and invariance; do not imply they are task predictors out of the box.

- UNI2-h.
- CONCH.
- CTransPath.
- Virchow2.
- ResNet50 as a conventional baseline.

### 6.2 Task-trained breast-cancer models

Train or obtain models with explicit downstream heads on the same patient-level splits.

- ResNet/ConvNeXt diagnostic classifier.
- CLAM or attention-MIL.
- TransMIL.
- A foundation-encoder MIL model using frozen and fine-tuned variants.
- Relevant breast subtype, grade, receptor, metastasis, recurrence, or treatment-response models where labels are available.

### 6.3 Survival models

Survival models are patient/WSI-level systems and must be evaluated separately from patch encoders.

- Pathology-only Cox-MIL baseline.
- DeepSurv-style baseline over aggregated pathology features.
- SurvMamba.
- MCAT, when genomic/clinical modalities are available.
- PORPOISE, when multimodal data are available.
- SurvPath, when pathway-level molecular features are available.

**Survival metrics**

- Concordance index.
- Time-dependent AUC.
- Integrated Brier score.
- Hazard-ratio or risk-score changes under interventions.
- Patient-level bootstrap confidence intervals.

Patch interventions must be propagated through the same WSI sampling and aggregation pipeline used by the survival model.

---

## 7. Probing experiments

### P1 — Color/stain invariance — Main

- Hold spatial map and morphology geometry fixed.
- Sweep stain/style conditions across realistic quantiles.
- Measure embedding cosine distance, centered kernel alignment where useful, and downstream prediction change.
- Compare raw RGB edits with H&E optical-density interventions.
- Include conventional stain augmentation/normalization baselines.

### P2 — Morphology sensitivity — Main

- Hold spatial structure and stain fixed.
- Sweep nuclear area, eccentricity, solidity, axes, and related features one at a time.
- Use low/median/high quantiles plus a continuous dose-response curve.
- Measure encoder and task-model sensitivity.
- Confirm recovered morphology with segmentation.

### P3 — Spatial tumor–immune sensitivity — Main

- Hold morphology and stain fixed.
- Vary tumor–immune distance, immune density, tumor density, mixing, clustering, and exclusion patterns.
- Evaluate embedding shifts and downstream predictions.
- Confirm intervention fidelity using G3 metrics.

### P4 — Seed invariance — Maybe / supplementary

Seed is not a biological signal and output appearance varies substantially. Use it mainly to establish confidence intervals and robustness of P1–P3, not as a central biological experiment.

### P5 — Layerwise probing — Supplementary

This overlaps with G5. Use hooks or blockwise representations to locate where stain, morphology, and spatial information appear in each encoder.

### P6 — Cross-model sensitivity benchmark — Main

This is the primary use of foundation encoders. For identical subtle interventions, compare which models capture the requested variation while remaining invariant to nuisance changes.

**Metrics**

- Standardized embedding change per intervention unit.
- Signal-to-seed-variance ratio.
- Linear-probe recoverability of intervention magnitude.
- Rank consistency across patients/slides.
- Biological-sensitivity versus nuisance-sensitivity ratio.
- Downstream-head prediction change where a trained head exists.

### P7 — Interaction and nonlinearity analysis — Optional

- Tumor–immune distance × immune density.
- Nuclear morphology × stain.
- Morphology × spatial organization.
- Test whether combined changes are additive or synergistic.

### P8 — Qualitative behavior panel — Main or supplementary

Show carefully selected but transparently defined examples of:

- Strong color correction through RGB/stain controls.
- Morphology-only edits.
- Spatial tumor–immune edits.
- Multiple seeds preserving conditions.
- Supervisor acceptance and rejection examples.
- At least one clear failure case.

Selection must come from a fixed evaluated pool, with selection criteria stated.

### P9 — Counterfactual evaluation of XAI methods — Optional main, otherwise supplementary

This does not compare saliency maps visually. It tests whether regions or concepts ranked as important by an explanation method cause larger model changes when edited on-manifold.

**Protocol**

1. Obtain an explanation map or concept ranking using Grad-CAM, Integrated Gradients, attention, occlusion, or another baseline.
2. Rank regions/cellular concepts by attributed importance.
3. Use PathOGen to apply a controlled, biologically plausible edit to the top-ranked region/concept.
4. Apply the same-sized and same-magnitude edit to a random region and a bottom-ranked region.
5. Verify that all outputs pass G1/G3/G4 or the fixed supervisor threshold.
6. Measure the target-model score drop/change as increasing fractions of ranked evidence are edited.

**Metrics**

- Area over the perturbation curve (AOPC).
- Prediction change for top versus random versus bottom interventions.
- Rank correlation between attribution and counterfactual effect.
- Flip rate under a fixed intervention budget.
- Fidelity-conditioned effect, excluding or separately reporting failed generations.

PathOGen functions as an on-manifold perturbation engine, replacing black rectangles, blur, or unrealistic pixel noise. This supports faithfulness evaluation; it does not prove that a saliency method reveals biological causality.

### P10 — Cross-domain framework validation — Stretch

Apply the complete framework to a genuinely different domain, preferably colorectal histopathology, with a domain-specific generator/data pipeline and downstream task.

The result should demonstrate transfer of the framework—not merely reuse the breast model on another stain distribution.

Possible settings:

- Colorectal tumor–immune organization and MSI prediction.
- Prostate gland morphology and grading.
- Renal tumor morphology/subtyping.

Report domain-specific fidelity and probing results using the same high-level protocol.

---

## 8. Data-quality audit of 1,000 H&E images

Use a deterministic random sample of 1,000 training tiles, stratified by slide/site if possible.

### Automated triage

- Tissue/background fraction.
- Blur/focus score.
- Over/underexposure.
- Saturation and hue outliers.
- Compression/block artifacts.
- Pen/marker and fold/artifact heuristics.
- Duplicate and near-duplicate detection.
- RGB/morphology conditioning outlier score.

### Human review

- Review random contact sheets and all automatically flagged tiles.
- Label acceptable, borderline, reject, and uncertain.
- Record artifact type.
- Do not delete tiles solely from automated heuristics.

### Analyses

- Compare bad-image rate across slide, scanner, source, and site.
- Compare RGB-conditioning outliers against visual-quality labels.
- Determine whether color failures originate from source H&E quality, feature calculation, scaling, or inference mismatch.
- Freeze and report any exclusion rule before final retraining.

---

## 9. Baselines and ablations

### Generator baselines

- No morphology conditioning.
- No spatial conditioning.
- Shuffled morphology.
- Shuffled spatial map.
- Direct spatial concatenation versus multi-scale adapter/ControlNet.
- Existing scaler versus corrected train-only scaler.
- Raw RGB features versus separated stain/style condition.
- StandardScaler versus robust/log-scaled features for a newly trained model.
- Raw checkpoint versus EMA.
- Standard MSE versus Min-SNR weighting.
- No condition dropout versus independent condition dropout.

### Counterfactual/XAI baselines

- Pixel-space color augmentation.
- Stain normalization/augmentation.
- Blur or occlusion.
- Inpainting where appropriate.
- Nearest-real-example retrieval.
- Conventional latent interpolation if a suitable baseline exists.

Every baseline should be checked for realism and intervention magnitude so that stronger prediction changes are not simply caused by more severe artifacts.

---

## 10. Statistical protocol

- Define the primary endpoint for each experiment before running the final test set.
- Use patient/slide-level bootstrap confidence intervals.
- Use paired tests because interventions share the same source image.
- Correct for multiple comparisons within each experiment family.
- Report effect sizes, not only p-values.
- Keep the test split sealed during threshold and supervisor calibration.
- Report sample count at tile, slide, and patient levels.
- Use the same fixed cases, intervention magnitudes, and seed bank across model comparisons.
- Report failures and supervisor rejection rates by condition, slide, and subgroup.

---

## 11. Eight-page WACV paper allocation

Suggested main-paper budget:

| Section | Approximate pages | Content |
|---|---:|---|
| Introduction | 0.75 | Motivation, gap, contributions |
| Related work | 0.5 | Histology generation, counterfactual XAI, pathology foundation models |
| Method | 1.5 | Generator, spatial/morphology controls, supervisor |
| Experimental setup | 0.75 | Data, splits, models, metrics |
| Generator fidelity | 1.25 | G1, G3, G4 and key ablations |
| Model probing | 1.5 | P1, P2, P3, P6 |
| Qualitative/XAI result | 0.75 | P8 and possibly compact P9 |
| Discussion/limitations/conclusion | 0.75 | Failure modes, causality limits, clinical scope |

### Main paper priorities

- G1, G3, G4.
- Compact G7 supervisor description and impact.
- P1, P2, P3, P6.
- One strong P8 figure.
- P9 only if results are compelling and space permits.

### Supplementary priorities

- G5 and G6.
- P4 and P5.
- Full model tables and per-feature curves.
- Detailed RGB/scaler diagnostics.
- Full supervisor calibration and rejection galleries.
- Extended P9 comparisons.
- P10, unless it becomes a major second-domain result.

---

## 12. Recommended main figures and tables

### Figures

1. **Framework overview:** conditions → generator → counterfactuals → fidelity checks → model probing.
2. **Controllability grid:** morphology, stain, and tumor–immune spatial interventions.
3. **Fidelity results:** G3/G4 dose-response and crosstalk matrix.
4. **Cross-model probing:** model × intervention sensitivity heatmap for P6.
5. **Qualitative behavior/failures:** compact P8 panel, possibly including supervisor decisions.

### Tables

1. G1 realism and pathology-distribution metrics.
2. G3/G4 fidelity and generator ablations.
3. P1/P2/P3/P6 model comparison.
4. Optional P9 XAI-faithfulness table.

---

## 13. Minimum viable WACV submission

Required:

- Correct patient/slide-held-out evaluation.
- Corrected, serialized preprocessing.
- G1, G3, and G4 with confidence intervals.
- Strong evidence that conditions are used and selective.
- P1, P2, P3, and P6 across at least four meaningfully different model families.
- At least one task-trained breast-cancer model in addition to embedding encoders.
- Transparent supervisor/filtering evaluation if filtering is used.
- Qualitative successes and failures.
- Reproducible experiment manifests.

High-value additions:

- Pathologist review.
- Independent nucleus-analysis model.
- Survival-model intervention study.
- P9 on-manifold XAI faithfulness.
- External-site validation.

Stretch:

- P10 full second-domain demonstration.

---

## 14. Experiment tracking template

Copy this block for every run:

```yaml
experiment_id:
date:
owner:
hypothesis:
code_commit:
checkpoint_path:
checkpoint_sha256:
dataset_version:
split_manifest:
patient_count:
slide_count:
tile_count:
preprocessing_version:
scaler_path:
scaler_sha256:
feature_order:
generator_configuration:
inference_scheduler:
inference_steps:
seed_list:
condition_definition:
intervention_magnitude:
supervisor_checkpoint:
supervisor_thresholds:
acceptance_rate:
primary_metric:
secondary_metrics:
confidence_interval_method:
result_artifacts:
notes:
```

---

## 15. Immediate action list

- [ ] Inspect and label the 1,000-image H&E quality sample.
- [ ] Recover/verify the original morphology scaler and feature ordering.
- [ ] Audit RGB variance outliers and FiLM saturation.
- [ ] Create a patient/slide-level train/validation/test manifest.
- [ ] Measure the VAE reconstruction ceiling.
- [ ] Implement a canonical G1/G3/G4 evaluator.
- [ ] Define deterministic G7 quality gates.
- [ ] Run short corrected-preprocessing training on H200.
- [ ] Select the best checkpoint using validation data only.
- [ ] Freeze the test protocol and intervention grid.
- [ ] Train or obtain breast-cancer task/MIL models.
- [ ] Run P1, P2, P3, and P6.
- [ ] Decide whether P9 has sufficient results for the main paper.
- [ ] Keep P10 as a separate cross-domain extension unless completed early.
