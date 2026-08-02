# Model conditions

This directory contains derived conditioning inputs. Six sample source tiles and
their nucleus GeoJSON remain under `data/interim/`.

Workflow 05 reads the repository-local files directly:

```text
spatial_maps/<stem>.npz
morphology/standardized.parquet
```

The standardized table is the model-compatible 38,978-row table copied from the
original prepared dataset. Workflow 05 intersects its index with the six local
spatial maps and does not create modified condition files.

Running `python workflows/02_build_conditions/run.py` recomputes five-channel
spatial maps and a new morphology/scaler bundle from the selected input tiles.
Do not overwrite the bundled standardized table unless retraining or intentionally
changing the checkpoint's preprocessing contract.
