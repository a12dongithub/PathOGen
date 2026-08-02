"""Spatial and feature-wise conditioning components."""

from cpathogen.generation.conditioning.film import (
    FiLM_MLP,
    FiLMMLP,
    film_condition,
    inject_film_into_unet,
    set_film_condition,
)
from cpathogen.generation.conditioning.spatial_encoder import SpatialCondEncoder

__all__ = [
    "FiLMMLP",
    "FiLM_MLP",
    "SpatialCondEncoder",
    "film_condition",
    "inject_film_into_unet",
    "set_film_condition",
]
