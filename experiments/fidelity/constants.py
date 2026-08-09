"""Shared feature and cell-type definitions from PathOGen training."""

from __future__ import annotations

MORPH_FEATURES = [
    "area_mean",
    "area_var",
    "eccentricity_mean",
    "eccentricity_var",
    "solidity_mean",
    "solidity_var",
    "perimeter_mean",
    "perimeter_var",
    "grad_mean",
    "grad_var",
    "r_mean",
    "r_var",
    "g_mean",
    "g_var",
    "b_mean",
    "b_var",
]

MORPH_MEAN_FEATURES = [name for name in MORPH_FEATURES if name.endswith("_mean")]

CELL_TYPES = {
    0: "Neoplastic",
    1: "Inflammatory",
    2: "Connective",
    3: "Dead",
    4: "Epithelial",
}

CELL_TYPE_ALIASES = {
    "neoplastic": "Neoplastic",
    "tumor": "Neoplastic",
    "tumour": "Neoplastic",
    "inflammatory": "Inflammatory",
    "immune": "Inflammatory",
    "connective": "Connective",
    "stromal": "Connective",
    "stroma": "Connective",
    "dead": "Dead",
    "necrotic": "Dead",
    "epithelial": "Epithelial",
    "non-neoplastic epithelium": "Epithelial",
    "non-neoplastic epithelial": "Epithelial",
    # Segmentation-only evaluators such as StarDist intentionally do not
    # invent a PanNuke class.  Keeping these instances lets total-count,
    # centroid, and morphology metrics use their contours while typed metrics
    # remain explicitly unavailable.
    "unclassified": "Unclassified",
}

CELL_NAME_TO_CHANNEL = {name: channel for channel, name in CELL_TYPES.items()}
CELL_NAMES_WITH_TOTAL = ["Total", *CELL_TYPES.values()]

CELL_COLORS = {
    "Neoplastic": (255, 0, 0),
    "Inflammatory": (34, 221, 77),
    "Connective": (35, 92, 236),
    "Dead": (254, 255, 0),
    "Epithelial": (255, 159, 68),
}
