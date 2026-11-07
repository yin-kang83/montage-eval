# -*- coding: utf-8 -*-
"""Smoke-run the montage harness on a synthetic archive."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MONTAGE_ARCHIVE", str(ROOT / "data" / "archive"))
os.environ.setdefault("MONTAGE_OUT", str(ROOT / "results"))
sys.path.insert(0, str(ROOT))

from montage_eval import core  # noqa: E402


if __name__ == "__main__":
    core.main()
