# CellViT++ upstream snapshot

- Repository: <https://github.com/TIO-IKIM/CellViT-plus-plus>
- Branch: `main`
- Commit: `463c5c44bfdebfbe3943597eaa84daf3f5e26a5f`
- Snapshot date: 2026-08-01
- License: see `LICENSE`; original inference code is Apache 2.0 with a Commons
  Clause and mandatory CellViT/CellViT++ citation requirements.

The `cellvit/` package is an unbundled source snapshot used by the thin adapter
in `src/cpathogen/annotation/`. One portability change is marked directly in
`cellvit/utils/tools.py`: CuPy imports are optional so the upstream CPU
postprocessor can run on Apple Silicon and CPU-only machines. No model or
postprocessing equations were changed.

The supported CUDA whole-slide CLI remains the upstream
`cellvit/detect_cells.py`. CPathoGen's workflow 01 uses the same upstream model
classes and `DetectionCellPostProcessor` through a tile-oriented adapter because
the upstream in-memory WSI runner requires CUDA, CuPy, and Ray.

Workflow 01 must not be used to imply that CPathoGen owns the vendored code or
that CellViT++ predictions are pathologist ground truth. Preserve this notice,
the upstream license, checkpoint provenance, and the required paper citations
when redistributing the adapter or publishing results.
