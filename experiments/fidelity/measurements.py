"""Recompute the 16 training morphology statistics from CellViT contours."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .constants import MORPH_FEATURES
from .data import CellObservation


def morphology_measurements(
    image: Image.Image, cells: list[CellObservation]
) -> dict[str, float]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(sobel_x**2 + sobel_y**2)

    values: dict[str, list[float]] = {
        "area": [],
        "eccentricity": [],
        "solidity": [],
        "perimeter": [],
        "grad": [],
        "r": [],
        "g": [],
        "b": [],
    }
    for cell in cells:
        contour = np.rint(cell.contour).astype(np.int32)
        contour[:, 0] = np.clip(contour[:, 0], 0, width - 1)
        contour[:, 1] = np.clip(contour[:, 1], 0, height - 1)
        if len(contour) < 3:
            continue
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        solidity = area / hull_area if hull_area > 0 else 0.0
        eccentricity = 0.0
        if len(contour) >= 5:
            _, axes, _ = cv2.fitEllipse(contour)
            minor, major = sorted((float(axes[0]), float(axes[1])))
            if major > 0:
                eccentricity = float(np.sqrt(max(0.0, 1.0 - (minor / major) ** 2)))

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [contour], 1)
        if not np.any(mask):
            continue
        grad_mean = float(cv2.mean(gradient, mask=mask)[0])
        rgb_mean = cv2.mean(rgb, mask=mask)
        values["area"].append(area)
        values["eccentricity"].append(eccentricity)
        values["solidity"].append(solidity)
        values["perimeter"].append(perimeter)
        values["grad"].append(grad_mean)
        values["r"].append(float(rgb_mean[0]))
        values["g"].append(float(rgb_mean[1]))
        values["b"].append(float(rgb_mean[2]))

    output: dict[str, float] = {}
    for base in ("area", "eccentricity", "solidity", "perimeter", "grad", "r", "g", "b"):
        array = np.asarray(values[base], dtype=np.float64)
        output[f"{base}_mean"] = float(np.mean(array)) if array.size else float("nan")
        output[f"{base}_var"] = float(np.var(array)) if array.size else float("nan")
    if list(output) != MORPH_FEATURES:
        raise AssertionError("Morphology measurement order diverged from training feature order")
    output["detected_nuclei"] = float(len(values["area"]))
    return output
