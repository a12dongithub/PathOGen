"""Minimal inference-only copies of the trained PathOGen conditioning modules."""

from __future__ import annotations

import torch
from torch import nn


class FiLMMLP(nn.Module):
    def __init__(self, in_dim: int = 16, out_dim: int = 320):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim * 2),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gamma_beta = self.net(x)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma.clamp(-0.5, 0.5)
        beta = beta.clamp(-0.5, 0.5)
        return gamma.unsqueeze(-1).unsqueeze(-1), beta.unsqueeze(-1).unsqueeze(-1)


def inject_film_into_unet(unet, film_dim: int = 16) -> nn.ModuleList:
    film_mlps = nn.ModuleList()
    for module in unet.modules():
        if module.__class__.__name__ != "ResnetBlock2D":
            continue
        mlp = FiLMMLP(film_dim, module.out_channels).to(unet.device)
        film_mlps.append(mlp)
        module.original_forward = module.forward
        module.film_mlp = mlp

        def new_forward(self, hidden_states, temb=None, **kwargs):
            output = self.original_forward(hidden_states, temb, **kwargs)
            if getattr(self, "current_morph16", None) is not None:
                gamma, beta = self.film_mlp(self.current_morph16)
                output = (1.0 + gamma) * output + beta
            return output

        module.forward = new_forward.__get__(module, module.__class__)
    return film_mlps


class SpatialCondEncoder(nn.Module):
    """Downsample a 512x512x5 map to 64x64x4 latent-space features."""

    def __init__(self):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.net(x))
