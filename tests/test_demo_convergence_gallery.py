"""Smoke tests for demos/demo_convergence_gallery.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")


def _load_demo_module():
    """Load the gallery demo from file with a headless backend."""
    matplotlib.use("Agg", force=True)
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "demos" / "demo_convergence_gallery.py"
    spec = importlib.util.spec_from_file_location("demo_convergence_gallery", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module.__name__] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def demo_module():
    return _load_demo_module()


@pytest.fixture(scope="module")
def quick_gallery(demo_module):
    return demo_module.run_gallery(quick=True, seed=0)


def test_run_gallery_quick_converges(demo_module, quick_gallery):
    assert set(quick_gallery) == {label for label, *_ in demo_module.GALLERY}
    for reports in quick_gallery.values():
        assert reports  # non-empty sweep
        # Even the trimmed quick sweep must converge to every oracle.
        assert all(r.status == "pass" for r in reports)
        # Sample sizes are strictly increasing.
        ns = [r.n for r in reports]
        assert ns == sorted(ns)


def test_markdown_table_has_header_and_rows(demo_module, quick_gallery):
    label = next(iter(quick_gallery))
    table = demo_module.build_markdown_table(label, quick_gallery[label])
    assert label in table
    assert "| n |" in table
    assert "Oracle source" in table
    # One data row per sampled n.
    assert table.count("\n|") >= len(quick_gallery[label])


def test_create_convergence_plot_returns_figure(demo_module, quick_gallery):
    fig = demo_module.create_convergence_plot(quick_gallery)
    assert fig is not None
    assert len(fig.axes) >= len(quick_gallery)
    matplotlib.pyplot.close(fig)


def test_parse_args_defaults_and_overrides(demo_module):
    defaults = demo_module.parse_args([])
    assert defaults.quick is False
    assert defaults.seed == 0
    assert defaults.save is None

    custom = demo_module.parse_args(["--quick", "--seed", "7", "--save", "out.png"])
    assert custom.quick is True
    assert custom.seed == 7
    assert custom.save == "out.png"


def test_main_with_save(demo_module, tmp_path):
    out = tmp_path / "gallery.png"
    # quick sweep keeps it fast; --save avoids the interactive prompt.
    demo_module.main(["--quick", "--save", str(out)])
    assert out.exists() and out.stat().st_size > 0
