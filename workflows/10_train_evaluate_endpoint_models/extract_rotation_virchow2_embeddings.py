#!/usr/bin/env python3
"""Extract rotation embeddings and fill missing Virchow2 endpoint caches."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from cpathogen.endpoints.encoders import (
    build_encoder,
    choose_device,
    extract_embeddings_sharded,
    release_encoder,
)
from cpathogen.endpoints.variants import normalize_variant_manifests

GATED_MODELS = frozenset(("uni2h", "virchow2", "conch"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-root", type=Path, required=True)
    parser.add_argument("--counterfactual-root", type=Path, required=True)
    parser.add_argument("--real-images-dir", type=Path, required=True)
    parser.add_argument("--rotation-manifest", type=Path, required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        default=("resnet50", "ctranspath", "uni2h", "virchow2", "conch"),
    )
    parser.add_argument(
        "--full-models",
        nargs="+",
        default=("virchow2",),
        help="Models whose real and counterfactual caches must also be extracted.",
    )
    parser.add_argument("--ctranspath-checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--virchow-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--shard-size", type=int, default=1024)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    return parser.parse_args()


def save_cache(path: Path, id_key: str, ids: np.ndarray, values: np.ndarray) -> None:
    temporary = path.with_suffix(".npz.part")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **{id_key: ids, "embeddings": values})
    temporary.replace(path)


def cache_matches(path: Path, id_key: str, ids: np.ndarray) -> bool:
    if not path.is_file():
        return False
    with np.load(path, allow_pickle=False) as payload:
        return (
            id_key in payload
            and np.array_equal(payload[id_key].astype(str), ids.astype(str))
            and payload["embeddings"].shape[0] == len(ids)
            and np.isfinite(payload["embeddings"]).all()
        )


def extract(
    *,
    model_name: str,
    bundle,
    paths: list[Path],
    ids: np.ndarray,
    codes: list[int] | None,
    cache: Path,
    id_key: str,
    cache_root: Path,
    device,
    args: argparse.Namespace,
    label: str,
) -> None:
    if cache_matches(cache, id_key, ids):
        print(f"[{model_name}] reusing {cache.name}", flush=True)
        return
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{label}: {len(missing)} images missing; first={missing[0]}")
    batch_size = (
        args.virchow_batch_size if model_name == "virchow2" else args.batch_size
    )
    values = extract_embeddings_sharded(
        bundle,
        paths,
        ids,
        shard_root=cache_root / "shards",
        device=device,
        batch_size=batch_size,
        num_workers=args.num_workers,
        description=f"{model_name} {label}",
        augmentation_codes=codes,
        shard_size=args.shard_size,
    )
    save_cache(cache, id_key, ids, values)


def main() -> None:
    args = parse_args()
    endpoint_root = args.endpoint_root.expanduser().resolve()
    counterfactual_root = args.counterfactual_root.expanduser().resolve()
    real_images = args.real_images_dir.expanduser().resolve()
    cache_root = endpoint_root / "embedding_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    tile_manifest = pd.read_csv(endpoint_root / "tile_manifest.csv")
    variants = pd.read_csv(
        endpoint_root / "counterfactual_variant_manifest.csv"
    ).reset_index(drop=True)
    rotations = normalize_variant_manifests(
        [args.rotation_manifest.expanduser().resolve()]
    ).reset_index(drop=True)
    rotations.to_csv(endpoint_root / "rotation_variant_manifest.csv", index=False)
    device = choose_device(args.device)

    tile_ids = tile_manifest["tile_id"].astype(str).to_numpy()
    tile_paths = [real_images / f"{tile_id}.png" for tile_id in tile_ids]
    variant_ids = variants["variant_id"].astype(str).to_numpy()
    variant_paths = [
        counterfactual_root
        / str(row.experiment)
        / str(row.source_tile_id)
        / f"{row.condition}.png"
        for row in variants.itertuples(index=False)
    ]
    rotation_ids = rotations["variant_id"].to_numpy(dtype=str)
    rotation_paths = [Path(value) for value in rotations["image_path"]]
    rotation_codes = rotations["augmentation_code"].fillna(0).astype(int).tolist()

    for model_name in args.models:
        if model_name in GATED_MODELS and not os.environ.get("HF_TOKEN"):
            raise RuntimeError(
                f"{model_name} is gated. Set HF_TOKEN through a secret/environment "
                "variable after accepting its model license."
            )
        print(f"\n=== {model_name} embeddings on {device} ===", flush=True)
        ctranspath_checkpoint = args.ctranspath_checkpoint
        if model_name == "ctranspath" and ctranspath_checkpoint is None:
            from huggingface_hub import hf_hub_download

            ctranspath_checkpoint = Path(
                hf_hub_download(
                    repo_id="jamesdolezal/CTransPath",
                    filename="ctranspath.pth",
                )
            )
        bundle = build_encoder(
            model_name,
            device=device,
            ctranspath_checkpoint=ctranspath_checkpoint,
        )
        try:
            extract(
                model_name=model_name,
                bundle=bundle,
                paths=rotation_paths,
                ids=rotation_ids,
                codes=rotation_codes,
                cache=cache_root / f"{model_name}_rotation.npz",
                id_key="variant_ids",
                cache_root=cache_root,
                device=device,
                args=args,
                label="rotation",
            )
            if model_name in args.full_models:
                extract(
                    model_name=model_name,
                    bundle=bundle,
                    paths=tile_paths,
                    ids=tile_ids,
                    codes=None,
                    cache=cache_root / f"{model_name}_tiles.npz",
                    id_key="tile_ids",
                    cache_root=cache_root,
                    device=device,
                    args=args,
                    label="real tiles",
                )
                extract(
                    model_name=model_name,
                    bundle=bundle,
                    paths=variant_paths,
                    ids=variant_ids,
                    codes=None,
                    cache=cache_root / f"{model_name}_counterfactuals.npz",
                    id_key="variant_ids",
                    cache_root=cache_root,
                    device=device,
                    args=args,
                    label="counterfactuals",
                )
        finally:
            release_encoder(bundle)

    print(f"Wrote extension caches under {cache_root}", flush=True)


if __name__ == "__main__":
    main()
