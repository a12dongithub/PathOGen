#!/usr/bin/env python3
"""Generate Phase-2 evaluation tiles and calculate FID/KID."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from cpathogen.counterfactuals import ConditionStore
from cpathogen.generation.checkpoints import load_phase2_generation_models
from cpathogen.generation.counterfactuals import generate_matched_conditions
from cpathogen.generation.visualization import comparison_grid


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=REPO / "data")
    p.add_argument("--images-dir", type=Path, default=REPO / "data/images")
    p.add_argument("--spatial-maps-dir", type=Path)
    p.add_argument("--morphology-table", type=Path)
    p.add_argument("--checkpoint", type=Path, default=REPO / "models/pathogen_phase2/checkpoint_30000")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--stem", action="append", dest="stems")
    p.add_argument("--num-tiles", type=int, default=100)
    p.add_argument("--all-tiles", action="store_true")
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--candidates-per-tile", type=int, default=1,
        help="Independent deterministic noise samples generated for each source tile.",
    )
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--spatial-strength", type=float, default=2.0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--prompt", default="he")
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    p.add_argument("--base-model")
    p.add_argument("--revision")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--skip-metrics", action="store_true")
    p.add_argument("--num-grids", type=int, default=200)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if args.steps < 1:
        p.error("--steps must be at least 1")
    if args.spatial_strength < 0:
        p.error("--spatial-strength must be non-negative")
    if args.candidates_per_tile < 1:
        p.error("--candidates-per-tile must be at least 1")
    return args


def _select(args: argparse.Namespace, stems: tuple[str, ...]) -> list[str]:
    if args.stems:
        missing = sorted(set(args.stems) - set(stems))
        if missing:
            raise ValueError(f"Requested stems are not aligned: {missing[:5]}")
        return list(dict.fromkeys(args.stems))
    if args.all_tiles:
        return list(stems)
    if args.num_tiles < 1:
        raise ValueError("--num-tiles must be positive")
    return random.Random(args.sample_seed).sample(list(stems), min(args.num_tiles, len(stems)))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate torchmetrics FID/KID without coupling model loading to metrics."""
    from PIL import Image
    from torchvision.transforms import Compose, Resize, ToTensor
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = Compose([Resize((299, 299)), ToTensor()])
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    kid = KernelInceptionDistance(
        feature=2048, normalize=True, subsets=10,
        subset_size=min(100, len(records)),
    ).to(device)
    for start in range(0, len(records), 32):
        batch = records[start : start + 32]
        real = torch.stack([transform(Image.open(item["real_path"]).convert("RGB")) for item in batch])
        generated = torch.stack([transform(Image.open(item["generated_path"]).convert("RGB")) for item in batch])
        # TorchMetrics' ``normalize=True`` expects float RGB tensors in [0, 1]
        # and performs its own conversion to uint8.  This is also what the
        # historical evaluator supplied to FID.
        fid.update(real.to(device), real=True)
        fid.update(generated.to(device), real=False)
        kid.update(real.to(device), real=True)
        kid.update(generated.to(device), real=False)
    kid_mean, kid_std = kid.compute()
    result = {
        "count": len(records),
        "device": str(device),
        "fid": float(fid.compute().item()),
        "kid_mean": float(kid_mean.item()),
        "kid_std": float(kid_std.item()),
    }
    return result


def _candidate_seed(base_seed: int, stem: str, candidate_index: int) -> int:
    payload = f"{base_seed}|{stem}|candidate|{candidate_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFFFFFF


def main() -> None:
    args = _args()
    # CLI paths are repository-relative by convention, while ConditionStore
    # interprets supplied relative paths relative to its data root.
    if args.images_dir is not None and not args.images_dir.is_absolute():
        args.images_dir = (REPO / args.images_dir).resolve()
    if args.spatial_maps_dir is not None and not args.spatial_maps_dir.is_absolute():
        args.spatial_maps_dir = (REPO / args.spatial_maps_dir).resolve()
    if args.morphology_table is not None and not args.morphology_table.is_absolute():
        args.morphology_table = (REPO / args.morphology_table).resolve()
    out = args.output_dir.expanduser().resolve()
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    generated = out / "generated"
    real = out / "real"
    grids = out / "grids"
    generated.mkdir(exist_ok=True)
    real.mkdir(exist_ok=True)
    if args.num_grids < 0:
        raise ValueError("--num-grids must be non-negative")
    if args.num_grids:
        grids.mkdir(exist_ok=True)

    store = ConditionStore(
        args.data_root,
        images_dir=args.images_dir,
        spatial_maps_dir=args.spatial_maps_dir,
        morphology_table=args.morphology_table,
    )
    selected = _select(args, store.stems)
    models = load_phase2_generation_models(
        args.checkpoint,
        base_model=args.base_model,
        revision=args.revision,
        device=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
    )
    jobs = [
        (source_index, stem, candidate_index, _candidate_seed(args.seed, stem, candidate_index))
        for source_index, stem in enumerate(selected)
        for candidate_index in range(args.candidates_per_tile)
    ]
    records: list[dict[str, Any]] = []
    for start in range(0, len(jobs), args.batch_size):
        batch_jobs = jobs[start : start + args.batch_size]
        conditions = [store.load(stem) for _, stem, _, _ in batch_jobs]
        images = generate_matched_conditions(
            models,
            conditions,
            seed=args.seed,
            prompt=args.prompt,
            num_inference_steps=args.steps,
            spatial_strength=args.spatial_strength,
            matched_noise=False,
            per_condition_seeds=[seed for _, _, _, seed in batch_jobs],
        )
        for (source_index, stem, candidate_index, seed), image, condition in zip(
            batch_jobs, images, conditions, strict=True
        ):
            generated_id = f"{source_index:06d}_{stem}__candidate_{candidate_index:02d}"
            generated_path = generated / f"{generated_id}.png"
            real_path = real / f"{stem}.png"
            image.save(generated_path)
            source = store.images_dir / f"{stem}.png"
            if not source.is_file():
                raise FileNotFoundError(f"Real source tile not found: {source}")
            if not real_path.exists():
                shutil.copy2(source, real_path)
            if source_index < args.num_grids and candidate_index == 0:
                spatial_map = condition.spatial.permute(1, 2, 0).cpu().numpy()
                with Image.open(source) as source_image:
                    comparison_grid(
                        spatial_map, source_image, image, args.checkpoint.name
                    ).save(grids / f"{source_index:06d}_{stem}.png")
            records.append({
                "index": len(records),
                "stem": stem,
                "source_stem": stem,
                "source_index": source_index,
                "candidate_index": candidate_index,
                "generation_seed": seed,
                "generated_id": generated_id,
                "generated_path": str(generated_path),
                "real_path": str(real_path),
                "source_image": str(source),
                "spatial_map": str(store.spatial_maps_dir / f"{stem}.npz"),
                "morphology_row": str(store.morphology_table),
            })
        print(f"Generated {min(start + args.batch_size, len(jobs))}/{len(jobs)} candidates", flush=True)

    _write_json(out / "manifest.json", {
        "schema_version": 1,
        "workflow": "06_evaluate_phase2_fid_kid",
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "seed": args.seed,
        "sample_seed": args.sample_seed,
        "steps": args.steps,
        "spatial_strength": args.spatial_strength,
        "candidates_per_tile": args.candidates_per_tile,
        "matched_initial_noise": False,
        "num_grids": args.num_grids,
        "records": records,
    })
    if not args.skip_metrics:
        if args.candidates_per_tile == 1:
            result = _metrics(records)
            _write_json(out / "fid_kid.json", result)
            print(json.dumps(result, indent=2), flush=True)
        else:
            by_candidate = {}
            for candidate_index in range(args.candidates_per_tile):
                result = _metrics([item for item in records if item["candidate_index"] == candidate_index])
                by_candidate[str(candidate_index)] = result
            summary = {
                "candidates_per_tile": args.candidates_per_tile,
                "per_candidate": by_candidate,
                "fid_mean": sum(item["fid"] for item in by_candidate.values()) / len(by_candidate),
                "kid_mean_mean": sum(item["kid_mean"] for item in by_candidate.values()) / len(by_candidate),
            }
            _write_json(out / "fid_kid_by_candidate.json", summary)
            print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote {len(records)} generated candidates for {len(selected)} source tiles to {out}")


if __name__ == "__main__":
    main()
