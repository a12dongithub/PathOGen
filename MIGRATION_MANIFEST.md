# Refactoring migration manifest

The workspace was reorganized on 2026-07-27 using same-filesystem moves. Existing files were not copied.

| Previous location | New location | Classification |
|---|---|---|
| `README.md`, `docs/`, `PAPER_EXPERIMENTS.md` | `README.md`, `docs/` hierarchy | Documentation |
| `PathOGen-Training/PathOGen-Training/*.py` | `src/cpathogen/`, `workflows/`, `tests/` | Canonical generator source |
| `PathOGen-Training/PathOGen-Training/.git` | `.git` | Preserved generator Git history at the new project root |
| `Foundational/Foundational/01-54*.py` | `experiments/` groups | Historical experiments |
| `Foundational/NuHTC-main/` | `third_party/nuhtc/` | Third-party/legacy source |
| `PathOGen-Training/PathOGen/*.py` | `archive/legacy_code/pathogen_legacy/` | Duplicate/legacy source |
| `Foundational/Data/` | `data/raw/pannuke/` | Raw dataset copy |
| `Foundational/Foundational/Data/` | `data/raw/bach/images/` | Raw dataset copy |
| `Foundational/Foundational/Data_512/` | `data/interim/tiles/bach/` | Derived tile dataset |
| `Foundational/Foundational/Data_10k_Intermediate/*.pt` | `data/misc/tcga_10k_cached_tensors/` | Legacy cached tensors |
| Generator and downstream weights/results | `artifacts/` | Generated/fitted artifacts |
| Posters and abstracts | `reports/` | Research communication |
| Administrative documents | `archive/administrative/` | Restricted/non-project material |
| Top-level ZIP snapshots | `archive/workspace_snapshots/` | Cold archive |

## Canonicalization pass

A second pass on 2026-07-27 aligned names and locations with the documented workflow contract:

| Previous refactored location/name | Canonical location/name |
|---|---|
| `data/raw/bach/{class}/` | `data/raw/bach/images/{class}/` |
| `data/raw/bach/microscopy_ground_truth.csv` | `data/raw/bach/labels.csv` |
| `data/processed/bach_512/` | `data/interim/tiles/bach/` |
| `data/processed/features/*.parquet` | `data/processed/classification/<task>/embeddings/<encoder>/` |
| `data/interim/tcga_10k_intermediate/*.pt` | `data/misc/tcga_10k_cached_tensors/` |
| `generate_spatial_maps.py` | `src/cpathogen/preprocessing/spatial_maps.py` |
| `generate_morphology_features.py` | `src/cpathogen/preprocessing/morphology_features.py` |
| `generate_metadata.py` | `src/cpathogen/preprocessing/metadata.py` |
| `train_text_to_image_base.py` | `src/cpathogen/generation/phase1.py` |
| `train_pathogen.py` | `src/cpathogen/generation/phase2.py` |
| `validation_utils.py` | `src/cpathogen/generation/inference.py` |
| top-level historical generation/FID utilities | workflow or evaluation `misc/` directories |
| extracted checkpoints | `artifacts/runs/<run>/checkpoints/` |
| archived checkpoint ZIP files | `artifacts/misc/checkpoint_archives/` |
| legacy generated/result folders | `artifacts/runs/<descriptive-legacy-run>/` |
| historical loose metrics | matching `artifacts/runs/<run>/metrics/` directories |

Missing canonical stages were added as explicit Python TODO stubs. Missing datasets have directory-level README placeholders. Hard-coded project data/output paths were updated, but historical experiment behavior and artifact provenance have not been scientifically revalidated.
