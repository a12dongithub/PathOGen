# Known issues and decisions needed

This register records confirmed problems or ambiguities in the workspace, updated for the 2026-07-27 directory migration. “Blocking” means the issue should be resolved before a new expensive training run or a public reproducibility claim.

## Blocking: architecture name does not match current code

The abstract and poster describe ControlNet spatial conditioning. The current Git-backed `phase2.py`, `inference.py`, and cloud command explicitly implement **direct spatial concatenation without ControlNet**. An older ControlNet checkpoint and evaluator also remain in the workspace.

Decision required:

- either document the paper method as direct concat + FiLM and ensure all reported results use it;
- or restore/version a verified ControlNet implementation and rerun the claimed experiments.

Every result should include an explicit architecture label.

## Blocking: split-safe morphology fitting is not enforced

The refactored `morphology_features.py` writes raw and standardized Parquet tables, `scaler.joblib`, and a feature manifest. It still fits the scaler over every input tile supplied by the caller; the program cannot prove those inputs belong only to the training split.

Resolution: create patient/slide split manifests first, invoke preprocessing on training inputs only when fitting the scaler, reuse the serialized bundle for validation/inference, and add the split hash and clipping/outlier policy to the manifest.

## Blocking: complete main dataset and annotation-quality record are absent

The posters describe approximately 1.038 million TCGA-BRCA tiles from 1,114
slides and CellViT++ annotations. Workflow 01 now pins the CellViT++ source
revision and local CellViT-SAM-H x40 checkpoint, provides source-tile and
counterfactual-pair commands, and records hashes/settings in manifests. The full
tile set, historical split/case list, and annotation-quality report are not
present in this workspace.

Resolution: recover the TCGA case/slide identifiers and split rules, run the
pinned workflow over the intended cohort, perform expert/sample QC plus detector
agreement checks, and publish non-restricted manifests without embedding
restricted images in source control.

## Blocking: validation can fall back to training data

`inference._collect_validation_pairs` can fall back to training inputs when explicit validation paths are unavailable. A metric produced through this fallback is not a held-out estimate.

Resolution: require an explicit validation manifest for publication runs and make fallback opt-in with a visible “training-subset diagnostic” label.

## Blocking: distributed FID is calculated on one shard

Both phase-1 and phase-2 validation split the selected evaluation cases across Accelerate processes. Each process computes FID from its local images, there is no gather or distributed metric reduction, and only the main process logs its local value. On an eight-GPU run, the displayed score therefore represents roughly one eighth of the selected set rather than the stated full set.

Resolution: gather generated/real features or images across processes, compute one metric over the complete fixed manifest, and record the effective sample count. Recompute any headline score that came from the current multi-process validation path.

## High: spatial preprocessing resume validation is shallow

The extension mismatch was repaired and preprocessing now skips existing `.npz` outputs. It still does not validate the stored `map` key, shape, dtype, or class-order metadata before skipping.

Resolution: validate the full map contract and a preprocessing-version sidecar before accepting cached output.

## High: source trees and result versions are duplicated (partially organized)

There are two generator directories, five large checkpoint ZIPs plus an extracted checkpoint, two 45-46 GB top-level workspace ZIPs, and multiple result copies. Several same-named scripts are byte-identical, while others differ.

The 2026-07-27 migration separated canonical source, legacy source, checkpoints, and result directories without deleting any copy. Remaining resolution:

- keep `src/cpathogen/` canonical;
- store checkpoint hashes and manifests in source control;
- move raw checkpoints/results/archives to versioned object storage;
- replace duplicated files with references in a data/model registry.

Do not delete any current copy until checksums and recovery requirements are recorded.

## High: project-level migration is not committed

The former nested generator `.git/` was moved to the refactored project root, preserving generator history. The reorganized documentation, downstream experiments, manifests, and path moves have not yet been reviewed and committed as a coherent project-level revision.

Resolution: review the migration diff, decide which newly organized project files belong in history, and commit source, documentation, configs, tests, and small manifests only. Keep large data/checkpoints excluded and add external retrieval instructions.

## High: migrated generator working tree is dirty

Before migration, the generator Git repository reported ten modified tracked files and one untracked test script. Most apparent changes were CRLF line-ending conversions. Ignoring end-of-line whitespace, the confirmed substantive tracked change was in `inference.py`, where VAE decode no longer forces the latent tensor to float32. `generate_test_30k_fid58.py` was untracked. These files have moved but remain uncommitted.

Resolution: normalize line endings with a real `.gitattributes`, review the dtype change, commit intentional code with an experiment reference, and keep local result helpers intentionally tracked or intentionally excluded.

## High: path normalization is implemented but not runtime-validated

Project scripts now use the shared repository layout, explicit CLI paths, or `cpathogen.utils.paths`. Historical scripts were mechanically redirected to canonical data/artifact locations, but most have not been rerun. Upstream examples inside `third_party/nuhtc/` retain their own machine-specific paths by design.

Resolution: smoke-test each retained workflow, migrate remaining experiment constants into configuration/CLI arguments, and add startup assertions that print resolved inputs and outputs.

## High: environment is not fully locked

`run_cloud.sh` pins Diffusers and Transformers but not most packages or model revisions. Historical snapshot/restore scripts use a different environment name and a nightly CUDA 12.8 PyTorch source, while the training script uses stable CUDA 11.8 wheels. The foundational experiments have no consolidated requirements file.

Resolution: capture exact Python/package/CUDA/driver/model revisions from a known run, add separate locked environments for generator and probing code, and test them from an empty environment.

## High: preliminary metrics have incompatible provenance

The workspace contains at least these FID narratives:

- poster: approximately 56;
- abstract: approximately 62 for phase 1 and approximately 102 for early conditional training at 15k steps;
- legacy `phase2_fid_results/fid_result.txt`: 480.8966 at conditioning scale 0.5;
- checkpoint/archive names containing `FID58`.

These differ in architecture, checkpoint, data, sample selection, or metric setup. A filename is not metric provenance.

Resolution: create a result registry with checkpoint hash, architecture, held-out manifest, sample count, seed, preprocessing bundle, generation settings, and metric library version. Recompute headline metrics using the sealed protocol.

## Medium: current FID implementation is insufficient for pathology claims

The code uses standard Inception FID and some evaluations use only 2,000 images. This measures distributional similarity imperfectly and does not prove spatial/morphology adherence.

Resolution: retain FID as one metric, increase fixed held-out sample size, add KID/precision/recall, pathology-encoder distances, VAE reconstruction analysis, re-segmentation fidelity, crosstalk, and seed confidence intervals as specified in the [paper experiment registry](../research/paper-experiments.md).

## Medium: five-class terminology varies

Spatial preprocessing names channel 4 `Epithelial`, while at least one foundational comment calls the corresponding class “Non-Neoplastic.” This can mislabel plots or interventions even when tensor order is unchanged.

Resolution: store a single class-order JSON with maps/checkpoints and import it in every visualizer/evaluator.

## Medium: map construction is centroid-density, not segmentation mask

The five-channel spatial maps place impulses at approximate polygon centroids and blur them. They do not preserve nucleus shape or an exact instance mask. Per-channel peak normalization also removes absolute density scale within a tile.

Resolution: describe these as blurred centroid-density maps. If absolute counts/density or nucleus geometry should be controllable, design and evaluate a representation that retains those quantities.

## Medium: experiment scripts overwrite or share output locations

Many historical scripts now write under `artifacts/runs/`, but their run names remain fixed. Reruns may overwrite results or silently mix outputs from different code/checkpoints.

Resolution: create immutable experiment IDs and write all outputs beneath `runs/<experiment_id>/` with a manifest.

## Medium: patient/slide split discipline is inconsistent

Some scripts correctly split by original BACH image or TCGA patient, while others select tiles, construct splits at runtime, or use hand-picked examples. There is no central split registry.

Resolution: create and version patient/slide-level train/validation/test manifests. Use the same sealed test set across generator, encoder, and downstream comparisons.

## Medium: fitted artifacts lack metadata

Joblib models, `.pth` adapters, Parquet features, and cached `.pt` tensors generally lack sidecar metadata describing code revision, training split, label order, encoder revision, transform, or tensor schema.

Resolution: add sidecar JSON/YAML manifests and schema/version checks at load time.

## Medium: third-party provenance is incomplete (directory separation completed)

NuHTC and its vendored MMDetection tree are now isolated under `third_party/nuhtc/`, while generator scripts remain project source. The workspace still lacks a combined third-party notices file and a record of local modifications.

Resolution: preserve upstream licenses and copyright notices, identify local modifications, and create `THIRD_PARTY_NOTICES.md` before release.

## Medium: project-level license and citation are missing

PanNuke's dataset card declares CC BY-NC-SA 4.0, and NuHTC/MMDetection include their own licenses. The project root has no license, data-use policy, model card, or citation file.

Resolution: obtain author/institution approval, select a compatible code license, document data/model terms separately, and add `CITATION.cff` after the project title/authors/version are finalized.

## Medium: administrative files contain personal information (isolated)

The signed funding application and email document are now isolated under `archive/administrative/`. They remain unrelated to executable research provenance and must not be included in a public release.

Resolution: move administrative records to access-controlled storage and run a privacy review before sharing the repository or archives.

## Medium: storage layout remains large (directory separation completed)

The workspace is about 191 GB. Source, data, artifacts, documentation assets, and archives are now separated, but the two full snapshots and extracted copies still occupy the same filesystem. This continues to affect backup and retention.

Resolution: separate source, restricted/raw data, model registry, experiment runs, publications, and administrative records. Use checksummed object storage and retention policies for large artifacts.

## Low: naming and formatting drift

- Project name varies between CPathoGen and PathOGen.
- `archive/legacy_code/foundational_gitattributes` lacked the leading dot required for Git behavior and is retained only as a historical file.
- Mixed LF/CRLF line endings make unchanged files appear modified.
- `__pycache__`, `.DS_Store`, `Thumbs.db`, and an Office lock file are present.
- The foundational numbering contains gaps and two `31_` scripts.

Resolution: adopt the naming rules in [Project overview](../onboarding/project-overview.md#names-and-versions), add repository hygiene rules, and preserve experiment numbers in a manifest rather than renaming historical scripts without provenance.

## Research and clinical limitations

- Generated images are synthetic and can contain plausible-looking false structures.
- Model sensitivity to a generated edit is not biological causality unless realism, intervention fidelity, identity preservation, and confounders are verified.
- Filtering generations based on a desired prediction change would bias the experiment.
- Foundation encoders are not clinical diagnostic models without a validated downstream task.
- No output in this repository is cleared for clinical use.

The experiment registry's matched controls, re-segmentation, raw-versus-filtered reporting, patient-level confidence intervals, and failure-case reporting should be treated as minimum publication safeguards.
