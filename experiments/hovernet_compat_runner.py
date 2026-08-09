#!/usr/bin/env python
"""Run the official HoVer-Net CLI with compatibility shims for modern Colab."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: hovernet_compat_runner.py HOVERNET_ROOT [run_infer args]"
        )
    project_root = Path(sys.argv.pop(1)).expanduser().resolve()
    entrypoint = project_root / "run_infer.py"
    if not entrypoint.is_file():
        raise FileNotFoundError(
            f"Official HoVer-Net run_infer.py missing: {entrypoint}"
        )

    # The official implementation predates NumPy 1.24 and PyTorch 2.6.
    # These aliases preserve its behavior without downgrading Colab's CUDA stack.
    for name, value in (
        ("bool", bool),
        ("complex", complex),
        ("float", float),
        ("int", int),
        ("object", object),
        ("str", str),
    ):
        if name not in np.__dict__:
            setattr(np, name, value)
    if not hasattr(np.lib, "pad"):
        np.lib.pad = np.pad  # type: ignore[attr-defined]

    original_torch_load = torch.load

    def compatible_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = compatible_torch_load  # type: ignore[assignment]
    sys.path.insert(0, str(project_root))
    sys.argv[0] = str(entrypoint)
    runpy.run_path(str(entrypoint), run_name="__main__")


if __name__ == "__main__":
    main()
