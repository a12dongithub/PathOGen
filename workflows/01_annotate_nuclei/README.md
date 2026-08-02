# Workflow 01: annotate nuclei with CellViT++

This workflow runs the pinned CellViT++ `CellViT-SAM-H-x40-AMP-001` model on
prepared image tiles. It has two uses in CPathoGen:

1. annotate real H&E tiles before Workflow 02 builds spatial and morphology
   conditions; and
2. re-annotate Workflow 05 baseline/counterfactual images so a later evaluation
   workflow can measure whether the requested cell changes appeared.

The adapter uses the upstream model and `DetectionCellPostProcessor`, but is
tile-oriented and supports CUDA, Apple MPS, and CPU. The upstream whole-slide
runner remains appropriate for WSI processing on its supported CUDA stack.
Prepared tiles retain their native pixel scale; a non-divisible edge is padded
only to the model's 16-pixel patch multiple and detections are clipped back to
the original image.

## Install and model

From the repository root:

```bash
pip install -e ".[annotation]"
```

The default checkpoint is:

```text
artifacts/models/cellvit_plus_plus/
└── cellvit_sam_h_x40_amp_001/
    └── model.pth
```

The checkpoint is trained for x40 images (approximately 0.25 micrometres per
pixel). Do not label x20 or resampled material as x40 without recording the
actual scale. The SHA-256 and model metadata are documented in the model
directory README and repeated in every run manifest.

## Annotate real source tiles

The defaults read all supported images in
`data/interim/tiles/tcga_brca/` and write matching GeoJSON files to
`data/interim/annotations/tcga_brca/geojson/`:

```bash
python workflows/01_annotate_nuclei/run.py
```

Use `--overwrite` to replace existing annotations. Without it, existing files
are validated instead of recomputed. To run one image into a separate directory:

```bash
python workflows/01_annotate_nuclei/run.py \
  --image data/interim/tiles/tcga_brca/<stem>.png \
  --output-dir artifacts/runs/cellvit_source_check/annotations \
  --device auto
```

## Annotate generated matched pairs

Point the workflow at the `pairs.jsonl` written by Workflow 05:

```bash
python workflows/01_annotate_nuclei/run.py \
  --pairs-manifest artifacts/runs/<generation-run>/pairs.jsonl \
  --device auto
```

With no explicit `--output-dir`, annotations are written beside the generation
manifest:

```text
artifacts/runs/<generation-run>/
├── pairs.jsonl
├── images/
└── cellvit_plus_plus_annotations/
    ├── annotation_manifest_<timestamp>.json
    └── <source-tile-stem>/
        └── seed_<10-digit-seed>/
            ├── baseline.geojson
            ├── <intervention-a>.geojson
            └── <intervention-b>.geojson
```

The baseline and counterfactual metadata share a `pair_group_id`, source tile
stem, and generation seed. Counterfactuals additionally store their unique
`pair_id` and full intervention record. This is the join contract for comparing
nucleus counts, class composition, geometry, and spatial organization later;
Workflow 01 produces the measurements but does not claim intervention success.

## Output contract

Each output is a GeoJSON `FeatureCollection` with one polygon feature per
nucleus. It records the PanNuke-compatible class name (Neoplastic,
Inflammatory, Connective, Dead, or Epithelial), CellViT++ type probability,
centroid, bounding box, source image, image dimensions, checkpoint hash,
upstream revision, magnification, and source/pair provenance.

Generated contours are clipped to image bounds. `--validate-only` checks
existing outputs without loading the model; older repository annotations are
accepted if a contour crosses the tile edge, and the number of such points is
reported in the manifest. Use `--allow-empty` only when a legitimate image may
contain no detected nuclei.

Run `python workflows/01_annotate_nuclei/run.py --help` for sampling,
confidence-threshold, recursive-directory, dtype, and device options.

## Upstream provenance and terms

The exact source snapshot and the small Apple/CPU portability patch are recorded
in `third_party/cellvit_plus_plus/UPSTREAM.md`. Review its upstream license and
mandatory CellViT/CellViT++ citation terms before distributing results.
