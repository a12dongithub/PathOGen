# Model conditions

This directory contains derived conditioning inputs. Source tiles and nucleus
GeoJSON remain under `data/interim/`.

Run `python workflows/02_build_conditions/run.py` to create five-channel spatial
maps, raw and standardized morphology/stain tables, the fitted scaler and feature
manifest, and `metadata.jsonl`.
