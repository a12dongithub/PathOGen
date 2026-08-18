from types import SimpleNamespace

import pytest
import torch

from cpathogen.generation.counterfactuals import _initial_latents, _to_pil


def test_repeated_per_condition_seed_replays_identical_noise() -> None:
    models = SimpleNamespace(
        device=torch.device("cpu"),
        dtype=torch.float32,
        unet=SimpleNamespace(
            config=SimpleNamespace(sample_size=8, out_channels=4)
        ),
    )
    latents = _initial_latents(
        models,
        batch_size=4,
        seed=0,
        matched_noise=False,
        per_condition_seeds=[11, 11, 29, 29],
    )
    torch.testing.assert_close(latents[0], latents[1], rtol=0.0, atol=0.0)
    torch.testing.assert_close(latents[2], latents[3], rtol=0.0, atol=0.0)
    assert not torch.equal(latents[0], latents[2])


def test_nonfinite_decoder_output_is_rejected() -> None:
    decoded = torch.zeros((1, 3, 8, 8), dtype=torch.float32)
    decoded[0, 0, 0, 0] = torch.nan
    with pytest.raises(RuntimeError, match="non-finite"):
        _to_pil(decoded)
