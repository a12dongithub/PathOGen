# CPathoGen limitations and evidence requirements

CPathoGen is an experimental research system. Generated images may contain
plausible-looking false structures, and neither generation nor model sensitivity
is cleared for diagnosis, treatment, or clinical validation. The following
issues must be resolved before strong scientific or reproducibility claims.

## Claim boundaries

- A prediction change on a generated image is not biological causality. Realism,
  intervention fidelity, selectivity, identity preservation, and confounders
  must be assessed independently.
- A shared diffusion seed improves matching but does not guarantee that only the
  requested feature changed.
- CellViT++ outputs are pseudo-labels. Reusing the same detector before and after
  generation makes measurements comparable but can reproduce detector bias or
  react to synthetic artifacts.
- Foundation encoders are representation models, not clinical diagnostic models
  without a validated task-specific system.
- Filtering generations by whether they cause a desired model response would
  bias the experiment. Report raw results, filtering rules, rejection rates, and
  failure cases.

## Architecture identity

The project record uses three different labels: Phase-1 H&E latent diffusion,
legacy ControlNet conditioning, and direct spatial concatenation plus FiLM. The
supported Phase-2 checkpoint is the third architecture; it is not ControlNet.
Every result must record architecture, checkpoint hash, preprocessing bundle,
sample manifest, seed list, inference settings, and metric implementation.
Legacy ControlNet and direct-concat results must never be combined.

## Data and split validity

Posters describe roughly 1.038 million TCGA-BRCA tiles from 1,114 slides, but
the complete cohort, historical patient/slide split, case list, and
annotation-quality report are not available in the active project. A small
aligned fixture proves software execution, not cohort-scale performance.

Tiles from one patient or slide are correlated. All train/validation/test splits
must be sealed at patient or slide level before tiling-derived examples or
fitted transformations are produced. Runtime tile splits and hand-picked cases
cannot support patient-level generalization claims. Report tile, slide, and
patient counts separately.

The morphology scaler is serialized, but the preprocessing stage cannot prove
that its inputs came only from a sealed training split. Each condition bundle
must include the split hash, scaler hash, 16-feature order, fitting population,
clipping/outlier policy, image scale, and preprocessing version. The same bundle
must be used for training, validation, generation, and any interface.

## Annotation and condition limits

The five classes—Neoplastic, Inflammatory, Connective, Dead, and Epithelial—are
broad computational categories, not exhaustive cell phenotypes. “Dead” is not a
direct viability assay, a nucleus label is only a proxy for a complete cell, and
terminology must remain fixed across maps, models, plots, and interventions.

Spatial conditions are blurred, per-channel peak-normalized nucleus-centroid
density maps. They are not instance masks, do not retain nuclear shape, and do
not preserve absolute calibrated counts. If absolute density, exact geometry,
or cell interactions are claimed, the representation and recovery metric must
be redesigned or supplemented.

The 16-value vector mixes nuclear geometry with Sobel-gradient and RGB
stain/scanner proxies. It should not be described entirely as biological
morphology. Heavy-tailed features, RGB variance outliers, scaler mismatch, and
FiLM clamp saturation can cause global color failures. Optical-density or
stain-vector measurements are preferred for biological stain analysis.

Cached spatial maps must be checked for expected key, shape, dtype, class order,
preprocessing version, and source hash before reuse; existence alone is not
sufficient validation.

## Generator evaluation limits

Historical validation has two serious problems: it could fall back to training
images when no explicit validation set was supplied, and distributed FID was
computed independently per process while only the main process's shard was
logged. Such results are diagnostics, not held-out full-set estimates.

Preliminary FID references—approximately 56 in a poster, approximately 62 for
Phase 1 and 102 for early conditional training in an abstract, 480.8966 for a
legacy ControlNet evaluation, and “FID58” in archive names—have incompatible or
incomplete provenance. Recompute headline metrics from a sealed held-out
manifest and a hashed checkpoint/protocol.

Generic Inception FID, especially on about 2,000 images, cannot establish
histopathology realism or condition adherence. The minimum evaluation should
combine a larger fixed sample with KID, generative precision/recall,
pathology-encoder distribution distances, VAE reconstruction analysis,
re-segmentation-based spatial and morphology recovery, intervention crosstalk,
seed confidence intervals, and expert review where feasible. See
[Experiments](EXPERIMENTS.md).

## Counterfactual and downstream evaluation

Interventions require null, shuffled-condition, donor-condition, and
matched-random controls. Requested and recovered changes must be compared for
all controlled features, not only the targeted feature. A strong diagonal and
small off-diagonal values in a crosstalk matrix are necessary evidence of
selectivity.

For downstream models, use identical cases, magnitudes, and seed banks across
encoders. Report effect sizes, paired tests, multiple-comparison correction,
patient/slide bootstrap confidence intervals, subgroup failure rates, and
within-condition versus between-condition variance. Molecular-subtype
predictions from H&E remain research estimates, not molecular assays.

## Reproducibility and provenance

The complete software environment is not locked. Publication runs need exact
Python, package, accelerator, CUDA/driver or MPS, model revision, checkpoint
hash, and random-seed records. Third-party model cards, licenses, access terms,
and local modifications must be reviewed separately.

Historical models, cached tensors, metrics, generated images, and experiment
scripts often lack sidecars specifying code revision, data split, label order,
image transform, or tensor schema. Their filenames are not provenance. Treat
them as research records until validated and registered.

The project has no finalized root license, data-use policy, model card, citation
file, or combined third-party notice. CellViT++ has its own Commons Clause and
mandatory citation conditions; archived PanNuke, NuHTC/MMDetection, BACH, TCGA,
foundation models, and other weights each retain separate terms. Institutional
approval is required before selecting a compatible release license.

Administrative records contain personal information and must remain outside any
public release. Large data, checkpoint archives, and historical results should
move to checksummed, access-controlled object storage rather than source
distribution.

## Naming and reporting rules

- Use **CPathoGen** for the project and paper; reserve **PathOGen** for historical
  generator identifiers.
- Call the supported spatial method **direct concat + FiLM**, not ControlNet.
- Say nucleus when the segmented object is a nucleus.
- State MPP, magnification, and resize history for all physical-size claims.
- Do not call all 16 values morphology or all pathology networks foundation
  models.
- Do not treat FID, a detector score, or a visually selected example as complete
  validation.
- Preserve immutable experiment IDs and manifests; never silently overwrite or
  mix runs.

The full experimental safeguards, baselines, statistics, supervisor policy, and
minimum publication package are specified in [Experiments](EXPERIMENTS.md).
