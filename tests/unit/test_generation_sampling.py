from __future__ import annotations

from types import SimpleNamespace

import torch

from cpathogen.generation.counterfactuals import _initial_latents


def _models() -> SimpleNamespace:
    return SimpleNamespace(
        device=torch.device("cpu"),
        dtype=torch.float32,
        unet=SimpleNamespace(config=SimpleNamespace(sample_size=2, out_channels=4)),
    )


def test_evaluation_latents_match_historical_device_seeded_batch() -> None:
    models = _models()
    actual = _initial_latents(models, batch_size=3, seed=42, matched_noise=False)
    expected = torch.randn(
        (3, 4, 2, 2), generator=torch.Generator(device="cpu").manual_seed(42)
    )
    torch.testing.assert_close(actual, expected)
    assert not torch.equal(actual[0], actual[1])


def test_counterfactual_latents_share_one_seeded_sample() -> None:
    models = _models()
    actual = _initial_latents(models, batch_size=3, seed=42, matched_noise=True)
    expected = torch.randn(
        (1, 4, 2, 2), generator=torch.Generator(device="cpu").manual_seed(42)
    )
    torch.testing.assert_close(actual, expected.expand_as(actual))
