from collections.abc import Generator
from pathlib import Path
from typing import Any

import datasets
import numpy as np
from datasets import Dataset
from datasets.splits import NamedSplit
from numpy.typing import NDArray
from PIL import Image
from tqdm import tqdm


tissue_map = {
    "Bile-duct": "Bile Duct",
    "HeadNeck": "Head & Neck",
    "Adrenal_gland": "Adrenal Gland",
}

features = datasets.Features(
    {
        "image": datasets.Image(mode="RGB"),
        "instances": datasets.Sequence(datasets.Image(mode="1")),
        "categories": datasets.Sequence(
            datasets.ClassLabel(
                num_classes=5,
                names=[
                    "Neoplastic",
                    "Inflammatory",
                    "Connective",
                    "Dead",
                    "Epithelial",
                ],
            )
        ),
        "tissue": datasets.ClassLabel(
            num_classes=19,
            names=[
                "Adrenal Gland",
                "Bile Duct",
                "Bladder",
                "Breast",
                "Cervix",
                "Colon",
                "Esophagus",
                "Head & Neck",
                "Kidney",
                "Liver",
                "Lung",
                "Ovarian",
                "Pancreatic",
                "Prostate",
                "Skin",
                "Stomach",
                "Testis",
                "Thyroid",
                "Uterus",
            ],
        ),
    }
)


def one_hot_mask(
    mask: NDArray[np.float64],
) -> tuple[NDArray[np.bool], NDArray[np.uint8]]:
    """Converts a mask to one-hot encoding.

    Returns:
        A dictionary with the following keys:
            - masks: A 3D array with shape (num_masks, height, width) containing the
                one-hot encoded masks.
            - labels: A 1D array with shape (num_masks,) containing the class labels.
    """
    masks: list[NDArray[np.bool]] = []
    labels: list[NDArray[np.uint8]] = []

    for c in range(mask.shape[-1] - 1):
        masks.append(mask[..., c] == np.unique(mask[..., c])[1:, None, None])
        labels.append(np.full(masks[-1].shape[0], c, dtype=np.uint8))

    return np.concatenate(masks), np.concatenate(labels)


def process(path: str, subfolder: str) -> Generator[dict[str, Any], None, None]:
    images = np.load(Path(path, "images", subfolder, "images.npy"), mmap_mode="r")
    masks = np.load(Path(path, "masks", subfolder, "masks.npy"), mmap_mode="r")
    types = np.load(Path(path, "images", subfolder, "types.npy"))

    for image, mask, tissue in tqdm(
        zip(images, masks, types, strict=True), total=len(images)
    ):
        mask, labels = one_hot_mask(mask)

        yield {
            "image": Image.fromarray(image.astype(np.uint8)),
            "instances": [Image.fromarray(m) for m in mask],
            "categories": labels,
            "tissue": tissue_map.get(tissue, tissue),
        }


if __name__ == "__main__":
    fold1 = Dataset.from_generator(
        process,
        gen_kwargs={"path": "PanNuke/Fold 1", "subfolder": "fold1"},
        features=features,
        split=NamedSplit("fold1"),
        keep_in_memory=True,
    )
    fold1.push_to_hub("RationAI/PanNuke")
    fold2 = Dataset.from_generator(
        process,
        gen_kwargs={"path": "PanNuke/Fold 2", "subfolder": "fold2"},
        features=features,
        split=NamedSplit("fold2"),
        keep_in_memory=True,
    )
    fold2.push_to_hub("RationAI/PanNuke")
    fold3 = Dataset.from_generator(
        process,
        gen_kwargs={"path": "PanNuke/Fold 3", "subfolder": "fold3"},
        features=features,
        split=NamedSplit("fold3"),
        keep_in_memory=True,
    )
    fold3.push_to_hub("RationAI/PanNuke")
