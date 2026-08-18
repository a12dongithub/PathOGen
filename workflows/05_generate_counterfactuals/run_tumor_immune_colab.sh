#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 DATA_ROOT CHECKPOINT OUTPUT_ROOT [BATCH_SIZE]" >&2
  exit 2
fi

DATA_ROOT=$1
CHECKPOINT=$2
OUTPUT_ROOT=$3
BATCH_SIZE=${4:-8}
PYTHON_BIN=${PYTHON_BIN:-python}

REPOSITORY_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
COHORT="$REPOSITORY_ROOT/configs/counterfactuals/tumor_immune_cohort_1000.csv"
RUNNER="$REPOSITORY_ROOT/workflows/05_generate_counterfactuals/run.py"

if [[ ! -f "$DATA_ROOT/morphology_stats.parquet" && ! -f "$DATA_ROOT/morphology/standardized.parquet" ]]; then
  echo "Morphology table not found under DATA_ROOT=$DATA_ROOT" >&2
  exit 1
fi
if [[ ! -d "$DATA_ROOT/spatial_maps" || ! -d "$DATA_ROOT/geojsons" ]]; then
  echo "DATA_ROOT must contain spatial_maps/ and geojsons/: $DATA_ROOT" >&2
  exit 1
fi
if [[ ! -f "$CHECKPOINT/unet/config.json" ]]; then
  echo "Checkpoint must directly contain unet/config.json: $CHECKPOINT" >&2
  exit 1
fi
if [[ ! "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "BATCH_SIZE must be a positive integer" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"
export PYTHONUNBUFFERED=1

COMMON_ARGS=(
  --data-root "$DATA_ROOT"
  --candidate-manifest "$COHORT"
  --checkpoint "$CHECKPOINT"
  --steps 30
  --spatial-strength 2
  --batch-size "$BATCH_SIZE"
  --device cuda
  --dtype float16
  --local-files-only
  --omit-baseline
  --tile-folder-layout
  --resume
)

echo "[1/2] Peritumoral ring (40 px tumor diameter): 1,000 x 3 = 3,000 PNGs"
"$PYTHON_BIN" "$RUNNER" \
  --experiment experiments.spatial.peritumoral_immune_ring \
  "${COMMON_ARGS[@]}" \
  --output-dir "$OUTPUT_ROOT/peritumoral_immune_ring_diameter40px"

echo "[2/2] Fixed-count separation (40 px tumor diameter): 1,000 x 5 = 5,000 PNGs"
"$PYTHON_BIN" "$RUNNER" \
  --experiment experiments.spatial.tumor_immune_mixing \
  "${COMMON_ARGS[@]}" \
  --output-dir "$OUTPUT_ROOT/tumor_immune_separation_diameter40px"

echo "Completed both experiments under: $OUTPUT_ROOT"
