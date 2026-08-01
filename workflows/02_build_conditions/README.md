# Build conditions

This workflow converts matched H&E tiles and nucleus GeoJSON files into:

- five-channel spatial-map NPZ files;
- raw and standardized 16-value morphology/stain tables;
- a serialized `StandardScaler` and feature manifest; and
- Phase-1 ImageFolder metadata.

Run `python workflows/02_build_conditions/run.py --help` from the repository
root after installing the `preprocessing` optional dependencies. The scaler is
only scientifically valid when the supplied tiles are the sealed training split.
