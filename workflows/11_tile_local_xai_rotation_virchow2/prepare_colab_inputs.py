#!/usr/bin/env python3
"""Resolve Drive archives and stage only inputs required by the A100 extension."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

REQUIRED_ENDPOINT_FILES = (
    "clinical_normalized.csv",
    "tile_manifest.csv",
    "counterfactual_variant_manifest.csv",
)
REQUIRED_CACHES = (
    "resnet50_tiles.npz",
    "resnet50_counterfactuals.npz",
    "ctranspath_tiles.npz",
    "ctranspath_counterfactuals.npz",
    "uni2h_tiles.npz",
    "uni2h_counterfactuals.npz",
    "conch_pretrained_bags.npz",
    "conch_counterfactuals_pretrained.npz",
)
REQUIRED_FOLD_FILES = (
    "pam50_patient_oof_predictions.csv",
    "survival_patient_oof_predictions.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mydrive-root", type=Path, required=True)
    parser.add_argument("--cvpr-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--endpoint-source", type=Path)
    parser.add_argument(
        "--counterfactual-source",
        type=Path,
        help="Already-extracted counterfactual directory or a ZIP archive.",
    )
    parser.add_argument(
        "--skip-dataset",
        action="store_true",
        help="Do not locate or extract 512_final_dataset.zip.",
    )
    return parser.parse_args()


def unique_paths(paths: list[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve()).lower()
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def extract_archives(archives: list[Path], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for index, archive in enumerate(archives, start=1):
        print(f"[extract {index}/{len(archives)}] {archive}", flush=True)
        with zipfile.ZipFile(archive) as payload:
            payload.extractall(destination)


def locate_counterfactual_root(search_root: Path) -> Path | None:
    if not search_root.is_dir():
        return None
    if valid_counterfactual_root(search_root):
        return search_root
    manifests = sorted(search_root.rglob("organized_bucket_images.csv"))
    for manifest in manifests:
        parent = manifest.parent
        if (parent / "nuclear_enlargement").is_dir() and (
            parent / "stain_brightness"
        ).is_dir():
            return parent
    return None


def valid_counterfactual_root(path: Path) -> bool:
    return (
        (path / "organized_bucket_images.csv").is_file()
        and (path / "nuclear_enlargement").is_dir()
        and (path / "stain_brightness").is_dir()
    )


def resolve_counterfactual_root(
    explicit: Path | None,
    search_roots: list[Path],
    unpack_root: Path,
) -> Path:
    if explicit is not None:
        explicit = explicit.expanduser().resolve()
        root = locate_counterfactual_root(explicit)
        if root is not None:
            print(f"[reuse] extracted counterfactual root: {root}", flush=True)
            return root
        if explicit.is_file() and explicit.suffix.lower() == ".zip":
            extract_archives([explicit], unpack_root)
            root = locate_counterfactual_root(unpack_root)
            if root is not None:
                return root
        raise FileNotFoundError(f"Invalid --counterfactual-source: {explicit}")

    for search_root in search_roots:
        direct = search_root / "CPathOGen_Counterfactuals"
        root = locate_counterfactual_root(direct)
        if root is not None:
            print(f"[reuse] extracted counterfactual root: {root}", flush=True)
            return root

    extraction_complete = unpack_root / ".extraction_complete"
    if extraction_complete.is_file():
        root = locate_counterfactual_root(unpack_root)
        if root is not None:
            print(f"[reuse] staged counterfactual root: {root}", flush=True)
            return root
    if unpack_root.exists():
        shutil.rmtree(unpack_root)

    archives: list[Path] = []
    for search_root in search_roots:
        archives.extend(search_root.rglob("CPathOGen_Counterfactuals*.zip"))
    archives = unique_paths(sorted(archives))
    if not archives:
        raise FileNotFoundError(
            "Could not find an extracted CPathOGen_Counterfactuals directory "
            "or CPathOGen_Counterfactuals*.zip archive."
        )
    extract_archives(archives, unpack_root)
    extraction_complete.write_text("complete\n", encoding="utf-8")
    root = locate_counterfactual_root(unpack_root)
    if root is None:
        raise FileNotFoundError("Could not locate the consolidated counterfactual folder")
    return root


def locate_dataset_root(search_root: Path) -> Path | None:
    for metadata in sorted(search_root.rglob("morphology_stats.parquet")):
        parent = metadata.parent
        if (parent / "images").is_dir():
            return parent
    return None


def valid_endpoint_root(path: Path) -> bool:
    return (
        all((path / name).is_file() for name in REQUIRED_ENDPOINT_FILES)
        and all(
            (path / "embedding_cache" / name).is_file() for name in REQUIRED_CACHES
        )
        and all(
            (path / "models" / "resnet50" / name).is_file()
            for name in REQUIRED_FOLD_FILES
        )
    )


def endpoint_candidates(search_roots: list[Path]) -> list[Path]:
    candidates: list[Path] = []
    for root in search_roots:
        if not root.is_dir():
            continue
        candidates.extend(
            (
                root / "PathOGenResults" / "endpoint_models",
                root / "endpoint_models",
            )
        )
        for manifest in root.rglob("tile_manifest.csv"):
            if manifest.parent.name == "endpoint_models":
                candidates.append(manifest.parent)
    return unique_paths(candidates)


def locate_endpoint_source(
    explicit: Path | None,
    search_roots: list[Path],
    unpack_root: Path,
) -> Path:
    if explicit is not None:
        explicit = explicit.expanduser().resolve()
        if explicit.is_dir() and valid_endpoint_root(explicit):
            return explicit
        if explicit.is_file() and explicit.suffix.lower() == ".zip":
            extract_archives([explicit], unpack_root)
        else:
            raise FileNotFoundError(f"Invalid --endpoint-source: {explicit}")
    for candidate in endpoint_candidates([unpack_root, *search_roots]):
        if valid_endpoint_root(candidate):
            return candidate
    archives: list[Path] = []
    for root in search_roots:
        if root.is_dir():
            archives.extend(root.rglob("PathOGenResults*.zip"))
    archives = unique_paths(sorted(archives))
    if not archives:
        raise FileNotFoundError(
            "Could not find an extracted endpoint_models directory or a "
            "PathOGenResults*.zip archive under the supplied Drive roots."
        )
    extract_archives(archives, unpack_root)
    for candidate in endpoint_candidates([unpack_root]):
        if valid_endpoint_root(candidate):
            return candidate
    raise FileNotFoundError("The PathOGenResults archive lacks the required endpoint caches")


def copy_endpoint_once(source: Path, destination: Path) -> None:
    if valid_endpoint_root(destination):
        print(f"[reuse] staged endpoint root: {destination}", flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"[copy] endpoint artifacts: {source} -> {destination}", flush=True)
    destination.mkdir(parents=True, exist_ok=True)
    cache_destination = destination / "embedding_cache"
    fold_destination = destination / "models" / "resnet50"
    cache_destination.mkdir(parents=True, exist_ok=True)
    fold_destination.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_ENDPOINT_FILES:
        shutil.copy2(source / name, destination / name)
    for name in REQUIRED_CACHES:
        shutil.copy2(source / "embedding_cache" / name, cache_destination / name)
    for name in REQUIRED_FOLD_FILES:
        shutil.copy2(
            source / "models" / "resnet50" / name,
            fold_destination / name,
        )
    if not valid_endpoint_root(destination):
        raise RuntimeError("Copied endpoint root failed validation")


def main() -> None:
    args = parse_args()
    mydrive = args.mydrive_root.expanduser().resolve()
    cvpr = args.cvpr_root.expanduser().resolve()
    work = args.work_root.expanduser().resolve()
    output = args.output_root.expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    search_roots = unique_paths([cvpr, mydrive])

    dataset_root: Path | None = None
    if not args.skip_dataset:
        dataset_unpack = work / "dataset_unpack"
        dataset_complete = dataset_unpack / ".extraction_complete"
        dataset_root = (
            locate_dataset_root(dataset_unpack) if dataset_complete.is_file() else None
        )
        if dataset_root is None:
            if dataset_unpack.exists():
                shutil.rmtree(dataset_unpack)
            dataset_archives = []
            exact = cvpr / "512_final_dataset.zip"
            if exact.is_file():
                dataset_archives = [exact]
            else:
                for root in search_roots:
                    dataset_archives.extend(root.rglob("512_final_dataset.zip"))
            dataset_archives = unique_paths(dataset_archives)
            if not dataset_archives:
                raise FileNotFoundError("Could not find 512_final_dataset.zip")
            extract_archives([dataset_archives[0]], dataset_unpack)
            dataset_complete.write_text("complete\n", encoding="utf-8")
            dataset_root = locate_dataset_root(dataset_unpack)
        if dataset_root is None:
            raise FileNotFoundError("Could not locate the extracted 512_final_dataset")

    counterfactual_root = resolve_counterfactual_root(
        args.counterfactual_source,
        search_roots,
        work / "counterfactual_unpack",
    )

    endpoint_source = locate_endpoint_source(
        args.endpoint_source,
        search_roots,
        work / "endpoint_unpack",
    )
    endpoint_root = output / "endpoint_models"
    copy_endpoint_once(endpoint_source, endpoint_root)

    paths = {
        "dataset_root": str(dataset_root) if dataset_root is not None else None,
        "real_images_dir": str(dataset_root / "images") if dataset_root else None,
        "counterfactual_root": str(counterfactual_root),
        "endpoint_root": str(endpoint_root),
        "results_root": str(output / "results"),
    }
    (output / "resolved_paths.json").write_text(
        json.dumps(paths, indent=2), encoding="utf-8"
    )
    print(json.dumps(paths, indent=2), flush=True)


if __name__ == "__main__":
    main()
