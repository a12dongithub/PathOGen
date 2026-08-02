# Archive

The archive preserves substantive material removed from the active workflow
surface. Nothing here is a supported import, workflow, dataset, model, or result
unless it is deliberately promoted and revalidated.

```text
archive/
├── code/           # historical training, downstream, and generation utilities
├── configs/        # historical training references
├── data/           # BACH, PanNuke, caches, classification data, NuHTC demo data
├── artifacts/      # historical models, checkpoints, results, and ZIP archives
├── docs/           # pre-consolidation documentation retained for traceability
├── experiments/    # historical classifiers, probes, and representation audits
├── notebooks/      # inactive notebooks
├── tools/          # inactive curator/conversion utilities
├── third_party/    # inactive upstream projects such as NuHTC
├── smoke/          # bounded execution records
├── legacy_code/    # earlier generator snapshots
└── administrative/ # private non-research records; never publish
```

Large archived data/models remain ignored by Git and should ultimately move to
checksummed access-controlled object storage. Unimplemented stubs and
regenerable caches were deleted rather than archived.
