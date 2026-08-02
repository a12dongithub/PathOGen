#!/usr/bin/env python3
"""Workflow 01: annotate source or generated tiles with CellViT++."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from cpathogen.annotation.cellvit_adapter import main


if __name__ == "__main__":
    main()
