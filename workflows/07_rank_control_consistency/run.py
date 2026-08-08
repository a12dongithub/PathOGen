#!/usr/bin/env python3
"""Re-annotate a Workflow 06 run and rank generated tiles by control agreement."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from cpathogen.preprocessing.morphology_features import build_morphology_features
from cpathogen.preprocessing.spatial_maps import build_spatial_maps


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--annotation-model", type=Path, default=REPO / "models/cellvit_plus_plus/cellvit_sam_h_x40_amp_001/model.pth")
    p.add_argument("--cellvit-root", type=Path, default=REPO / "third_party/cellvit_plus_plus")
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--top-k", action="append", type=int, default=[50, 250])
    p.add_argument("--skip-annotation", action="store_true")
    return p.parse_args()


def _run_annotation(image_dir: Path, output_dir: Path, args: argparse.Namespace) -> None:
    command = [
        sys.executable, str(REPO / "workflows/01_annotate_nuclei/run.py"),
        "--input-dir", str(image_dir), "--output-dir", str(output_dir),
        "--model", str(args.annotation_model), "--cellvit-root", str(args.cellvit_root),
        "--device", args.device, "--dtype", args.dtype, "--overwrite",
    ]
    subprocess.run(command, cwd=REPO, check=True)


def _map(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        value = data["map"]
    value = value.astype(np.float32)
    return value / 255.0 if value.max(initial=0) > 1 else value


def _build_conditions(image_dir: Path, geojson_dir: Path, output_dir: Path, n_jobs: int) -> pd.DataFrame:
    build_spatial_maps(geojson_dir, output_dir / "spatial_maps", n_jobs=n_jobs, overwrite=True)
    return build_morphology_features(
        image_dir, geojson_dir,
        output_dir / "morphology/raw.parquet",
        output_dir / "morphology/standardized.parquet",
        output_dir / "morphology/scaler.joblib",
        output_dir / "morphology/feature_manifest.json",
        n_jobs=n_jobs,
    )


def main() -> None:
    args = _args()
    run = args.run_dir.expanduser().resolve()
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    records = manifest["records"]
    generated = run / "generated"
    real = run / "real"
    annotations = run / "control_consistency" / "annotations"
    source_geojson = annotations / "baseline"
    generated_geojson = annotations / "generated"
    if not args.skip_annotation:
        _run_annotation(real, source_geojson, args)
        _run_annotation(generated, generated_geojson, args)
    conditions = run / "control_consistency" / "conditions"
    source_features = _build_conditions(real, source_geojson, conditions / "baseline", args.n_jobs)
    generated_features = _build_conditions(generated, generated_geojson, conditions / "generated", args.n_jobs)
    scaler = StandardScaler().fit(source_features.to_numpy())
    dump(scaler, conditions / "baseline_scaler.joblib")
    source_std = pd.DataFrame(scaler.transform(source_features), index=source_features.index, columns=source_features.columns)
    generated_std = pd.DataFrame(scaler.transform(generated_features), index=generated_features.index, columns=generated_features.columns)

    rows = []
    for record in records:
        stem = Path(record["generated_path"]).stem
        source_map = _map(conditions / "baseline/spatial_maps" / f"{stem}.npz")
        generated_map = _map(conditions / "generated/spatial_maps" / f"{stem}.npz")
        morph_delta = generated_std.loc[stem].to_numpy() - source_std.loc[stem].to_numpy()
        spatial_rmse = float(np.sqrt(np.mean((generated_map - source_map) ** 2)))
        morphology_mae = float(np.mean(np.abs(morph_delta)))
        rows.append({
            "index": record["index"], "stem": record["stem"],
            "generated_path": record["generated_path"], "real_path": record["real_path"],
            "spatial_rmse": spatial_rmse, "morphology_mae_standardized": morphology_mae,
        })
    ranking = pd.DataFrame(rows)
    ranking["control_score"] = (
        ranking["spatial_rmse"].rank(pct=True) + ranking["morphology_mae_standardized"].rank(pct=True)
    ) / 2.0
    ranking = ranking.sort_values(["control_score", "index"]).reset_index(drop=True)
    ranking.to_csv(run / "control_consistency/ranking.csv", index=False)
    for k in sorted(set(args.top_k)):
        selected = ranking.head(min(k, len(ranking)))
        (run / "control_consistency" / f"top_{k}_manifest.json").write_text(
            json.dumps({"count": len(selected), "selection": "lowest_control_score", "records": selected.to_dict("records")}, indent=2) + "\n",
            encoding="utf-8",
        )
    print(f"Ranked {len(ranking)} tiles; best control score={ranking.control_score.iloc[0]:.6f}")


if __name__ == "__main__":
    main()
