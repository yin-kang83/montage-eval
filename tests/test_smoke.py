"""Smoke tests for montage-eval."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_run(tmp_path):
    env_archive = ROOT / "data" / "archive"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_synthetic.py")], check=True)
    env = {
        "MONTAGE_ARCHIVE": str(env_archive),
        "MONTAGE_OUT": str(tmp_path),
        "PYTHONPATH": str(ROOT),
    }
    import os
    e = os.environ.copy(); e.update(env)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "run_eval.py")], check=True, env=e)
    data = json.loads((tmp_path / "montage_results.json").read_text(encoding="utf-8"))
    assert data["n"] > 0
    assert "last" in data["results"]
    assert "oracle" in data["results"]
