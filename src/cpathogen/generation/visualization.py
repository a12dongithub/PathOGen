"""Legacy-compatible visual grids for Phase-2 evaluation."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont


CELL_TYPES = (
    ("Neoplastic", (255, 255, 255)),
    ("Inflammatory", (0, 255, 255)),
    ("Connective", (0, 255, 0)),
    ("Dead", (255, 255, 0)),
    ("Non-Neoplastic Epi.", (255, 128, 0)),
)


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def spatial_map_to_rgb_with_legend(spatial_map: np.ndarray) -> Image.Image:
    """Render the five map channels exactly as the historical evaluator did."""
    values = spatial_map.astype(np.float32, copy=False)
    if values.max(initial=0.0) <= 1.0:
        values = values * 255.0
    height, width, channels = values.shape
    rgb = np.zeros((height, width, 3), dtype=np.float32)
    for channel, (_, color) in enumerate(CELL_TYPES[:channels]):
        rgb += (values[:, :, channel] / 255.0)[:, :, None] * np.asarray(color)
    image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    items = [
        item for channel, item in enumerate(CELL_TYPES[:channels])
        if values[:, :, channel].max(initial=0.0) > 1.0
    ]
    if not items:
        return image
    y_start = height - 20 * len(items) - 10
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle([2, y_start - 4, 170, height - 2], fill=(0, 0, 0, 180))
    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = _font(14)
    for index, (name, color) in enumerate(items):
        y = y_start + index * 20
        draw.ellipse([8, y + 3, 18, y + 13], fill=color)
        draw.text((24, y), name, fill="white", font=font)
    return image


def _add_label(image: Image.Image, text: str) -> Image.Image:
    draw = ImageDraw.Draw(image)
    font = _font(16)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    width, height = right - left, bottom - top
    x, y = (image.width - width) // 2, 4
    draw.rectangle([x - 4, y - 2, x + width + 4, y + height + 2], fill="black")
    draw.text((x, y), text, fill="white", font=font)
    return image


def comparison_grid(
    spatial_map: np.ndarray, real: Image.Image, generated: Image.Image, checkpoint_name: str
) -> Image.Image:
    """Return the historical three-panel evaluation grid."""
    spatial = _add_label(
        spatial_map_to_rgb_with_legend(spatial_map).resize((512, 512), Image.Resampling.NEAREST),
        "Spatial Map",
    )
    reference = _add_label(real.convert("RGB").resize((512, 512)), "Real H&E")
    synthetic = _add_label(generated.convert("RGB").resize((512, 512)), checkpoint_name)
    grid = Image.new("RGB", (1536, 512))
    grid.paste(spatial, (0, 0))
    grid.paste(reference, (512, 0))
    grid.paste(synthetic, (1024, 0))
    return grid
