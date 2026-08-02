"""Spatial conditioning compatible with the supported Phase-2 checkpoint.

The layer layout is a checkpoint contract; changing it requires retraining or
an explicit weight migration.
"""

from __future__ import annotations

from torch import Tensor, nn


class SpatialCondEncoder(nn.Module):
    """Downsample five 512-pixel control maps to four latent-space channels."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(5, 32, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 4, 1),
        )
        self.norm = nn.GroupNorm(1, 4)

    def forward(self, spatial_maps: Tensor) -> Tensor:
        """Return ``(batch, 4, 64, 64)`` features for 512-pixel inputs."""
        return self.norm(self.net(spatial_maps))
