"""FiLM morphology/stain conditioning for the Phase-2 PathOGen UNet.

Names and layer ordering match the historical trainer so the separate
``film_mlps.pt`` checkpoint can be loaded directly.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from torch import Tensor, nn


class FiLMMLP(nn.Module):
    """Map the 16-value morphology condition to per-channel scale and bias."""

    def __init__(self, in_dim: int = 16, out_dim: int = 320) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim * 2),
        )

    def forward(self, values: Tensor) -> tuple[Tensor, Tensor]:
        gamma, beta = self.net(values).chunk(2, dim=-1)
        gamma = gamma.clamp(-0.5, 0.5)
        beta = beta.clamp(-0.5, 0.5)
        return (
            gamma.unsqueeze(-1).unsqueeze(-1),
            beta.unsqueeze(-1).unsqueeze(-1),
        )


# Historical code and checkpoint documentation use this spelling.
FiLM_MLP = FiLMMLP


def inject_film_into_unet(unet: nn.Module, film_dim: int = 16) -> nn.ModuleList:
    """Attach a FiLM MLP to every Diffusers ``ResnetBlock2D`` in ``unet``."""
    film_mlps = nn.ModuleList()
    try:
        first_parameter = next(unet.parameters())
        module_kwargs = {
            "device": first_parameter.device,
            "dtype": first_parameter.dtype,
        }
    except StopIteration:
        module_kwargs = {}

    for module in unet.modules():
        if module.__class__.__name__ != "ResnetBlock2D":
            continue
        if hasattr(module, "film_mlp"):
            raise ValueError("UNet already has FiLM modules injected")

        film_mlp = FiLMMLP(film_dim, module.out_channels).to(**module_kwargs)
        film_mlps.append(film_mlp)
        module.original_forward = module.forward
        module.film_mlp = film_mlp

        def forward_with_film(self, hidden_states, temb=None, **kwargs):
            output = self.original_forward(hidden_states, temb, **kwargs)
            morphology = getattr(self, "current_morph16", None)
            if morphology is not None:
                gamma, beta = self.film_mlp(morphology)
                output = (1.0 + gamma) * output + beta
            return output

        module.forward = forward_with_film.__get__(module, module.__class__)

    if not film_mlps:
        raise ValueError("No ResnetBlock2D modules found in the supplied UNet")
    return film_mlps


def set_film_condition(unet: nn.Module, morphology: Tensor | None) -> int:
    """Set the current morphology tensor and return the number of FiLM blocks."""
    count = 0
    for module in unet.modules():
        if hasattr(module, "film_mlp"):
            module.current_morph16 = morphology
            count += 1
    if count == 0:
        raise ValueError("UNet has no injected FiLM blocks")
    return count


@contextmanager
def film_condition(unet: nn.Module, morphology: Tensor) -> Iterator[None]:
    """Apply one morphology condition for a forward pass, then always clear it."""
    set_film_condition(unet, morphology)
    try:
        yield
    finally:
        set_film_condition(unet, None)
