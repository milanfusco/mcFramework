"""Smoke test for demos/demo_backend_benchmark.py (thin wrapper over the CLI)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")


def _load_demo_module():
    """Load the benchmark demo from file with a headless backend."""
    matplotlib.use("Agg", force=True)
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "demos" / "demo_backend_benchmark.py"
    spec = importlib.util.spec_from_file_location("demo_backend_benchmark", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module.__name__] = module
    spec.loader.exec_module(module)
    return module


def test_demo_exposes_main():
    module = _load_demo_module()
    assert callable(module.main)


def test_main_quick_save(tmp_path):
    out = tmp_path / "bench.png"
    js = tmp_path / "bench.json"
    # One small size keeps the smoke test fast; --no-show avoids opening a window.
    rc = _load_demo_module().main(
        ["--quick", "--sizes", "1000", "--no-show", "--save", str(out), "--json", str(js)]
    )
    assert rc == 0
    assert out.exists() and out.stat().st_size > 0
    assert js.exists() and js.stat().st_size > 0
