import argparse
import json
import os
from collections.abc import Iterable
from pathlib import Path

from cpathogen.utils.paths import CONDITIONS_METADATA, TCGA_TILES


def build_metadata(
    tiles_dir: str | Path,
    output: str | Path,
    stems: Iterable[str] | None = None,
    prompt: str = "he",
) -> list[Path]:
    """Write ImageFolder metadata for a set of prepared H&E tiles."""
    tiles_dir = Path(tiles_dir)
    metadata_file = Path(output)

    if not tiles_dir.is_dir():
        raise FileNotFoundError(f"Tile directory does not exist: {tiles_dir}")

    requested_stems = set(stems) if stems is not None else None
    image_by_stem: dict[str, Path] = {}
    for extension in ("*.png", "*.jpg", "*.jpeg"):
        for image_path in sorted(tiles_dir.glob(extension)):
            if requested_stems is not None and image_path.stem not in requested_stems:
                continue
            if image_path.stem in image_by_stem:
                raise ValueError(
                    f"Multiple tile files found for stem: {image_path.stem}"
                )
            image_by_stem[image_path.stem] = image_path

    if requested_stems is not None:
        missing = sorted(requested_stems - image_by_stem.keys())
        if missing:
            raise ValueError(f"Missing tile files for stems: {missing[:5]}")

    images = [image_by_stem[stem] for stem in sorted(image_by_stem)]
    if not images:
        raise ValueError(f"No PNG or JPEG tiles found in {tiles_dir}")

    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    with metadata_file.open("w", encoding="utf-8") as handle:
        for image_path in images:
            entry = {
                "file_name": os.path.relpath(image_path, metadata_file.parent),
                "text": prompt,
            }
            handle.write(json.dumps(entry) + "\n")

    print(f"Wrote metadata for {len(images)} tiles to {metadata_file}")
    return images


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles-dir", default=str(TCGA_TILES))
    parser.add_argument(
        "--output",
        default=str(CONDITIONS_METADATA),
    )
    parser.add_argument("--prompt", default="he")
    args = parser.parse_args(argv)
    build_metadata(args.tiles_dir, args.output, prompt=args.prompt)


if __name__ == "__main__":
    main()
