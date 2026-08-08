#!/usr/bin/env python3
"""Generate Phase-2 evaluation tiles and calculate FID/KID."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from cpathogen.counterfactuals import ConditionStore
from cpathogen.generation.checkpoints import load_phase2_generation_models
from cpathogen.generation.counterfactuals import generate_matched_conditions


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=REPO / "data/processed/conditions")
    p.add_argument("--images-dir", type=Path, default=REPO / "data/interim/tiles/tcga_brca")
    p.add_argument("--spatial-maps-dir", type=Path)
    p.add_argument("--morphology-table", type=Path)
    p.add_argument("--checkpoint", type=Path, default=REPO / "artifacts/models/pathogen_phase2/checkpoint_30000")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--stem", action="append", dest="stems")
    p.add_argument("--num-tiles", type=int, default=100)
    p.add_argument("--all-tiles", action="store_true")
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--prompt", default="he")
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    p.add_argument("--base-model")
    p.add_argument("--revision")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--skip-metrics", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


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


def _metrics(records: list[dict[str, Any]], output: Path) -> None:
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
        fid.update((real * 255).to(torch.uint8).to(device), real=True)
        fid.update((generated * 255).to(torch.uint8).to(device), real=False)
        kid.update((real * 255).to(torch.uint8).to(device), real=True)
        kid.update((generated * 255).to(torch.uint8).to(device), real=False)
    kid_mean, kid_std = kid.compute()
    result = {
        "count": len(records),
        "device": str(device),
        "fid": float(fid.compute().item()),
        "kid_mean": float(kid_mean.item()),
        "kid_std": float(kid_std.item()),
    }
    _write_json(output / "fid_kid.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main() -> None:
    args = _args()
    out = args.output_dir.expanduser().resolve()
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    generated = out / "generated"
    real = out / "real"
    generated.mkdir(exist_ok=True)
    real.mkdir(exist_ok=True)

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
    records: list[dict[str, Any]] = []
    for start in range(0, len(selected), args.batch_size):
        batch_stems = selected[start : start + args.batch_size]
        conditions = [store.load(stem) for stem in batch_stems]
        images = generate_matched_conditions(
            models,
            conditions,
            seed=args.seed + start,
            prompt=args.prompt,
            num_inference_steps=args.steps,
        )
        for offset, (stem, image) in enumerate(zip(batch_stems, images, strict=True)):
            index = start + offset
            generated_path = generated / f"{index:06d}_{stem}.png"
            real_path = real / f"{index:06d}_{stem}.png"
            image.save(generated_path)
            source = store.images_dir / f"{stem}.png"
            if not source.is_file():
                raise FileNotFoundError(f"Real source tile not found: {source}")
            shutil.copy2(source, real_path)
            records.append({
                "index": index,
                "stem": stem,
                "generated_path": str(generated_path),
                "real_path": str(real_path),
                "source_image": str(source),
                "spatial_map": str(store.spatial_maps_dir / f"{stem}.npz"),
                "morphology_row": str(store.morphology_table),
            })
        print(f"Generated {min(start + args.batch_size, len(selected))}/{len(selected)}", flush=True)

    _write_json(out / "manifest.json", {
        "schema_version": 1,
        "workflow": "06_evaluate_phase2_fid_kid",
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "seed": args.seed,
        "sample_seed": args.sample_seed,
        "steps": args.steps,
        "records": records,
    })
    if not args.skip_metrics:
        _metrics(records, out)
    print(f"Wrote {len(records)} generated/real pairs to {out}")


if __name__ == "__main__":
    main()
