from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

WORKFLOW = (
    Path(__file__).parents[1]
    / "workflows"
    / "11_tile_local_xai_rotation_virchow2"
)
SCRIPT = WORKFLOW / "run_pathlupi_fixedbag.py"


class FakePathLUPI(torch.nn.Module):
    def forward(self, **kwargs):
        del kwargs
        survival = torch.tensor([[0.9, 0.7, 0.4, 0.1]])
        return None, survival


def test_predict_one_uses_bin_containing_five_years(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(WORKFLOW))
    spec = importlib.util.spec_from_file_location("run_pathlupi_fixedbag", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    risk, probability = module.predict_one(FakePathLUPI(), torch.zeros((2, 4)))

    assert risk == pytest.approx(-2.1)
    assert probability == pytest.approx(0.4)
    assert module.SURVIVAL_BIN_INDEX == 2
    assert module.SURVIVAL_INTERVAL_MONTHS == (42.4, 78.9)
