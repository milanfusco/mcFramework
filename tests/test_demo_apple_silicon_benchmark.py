"""Tests for runtime controls in demo_apple_silicon_benchmark.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")


def _load_demo_module():
    """Load demo module directly from file path."""
    matplotlib.use("Agg", force=True)
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "demos" / "demo_apple_silicon_benchmark.py"
    spec = importlib.util.spec_from_file_location("demo_apple_silicon_benchmark", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module.__name__] = module
    spec.loader.exec_module(module)
    return module


def _fake_result(module, backend: str, torch_device: str, n_simulations: int):
    backend_name = backend if backend != "torch" else f"torch-{torch_device}"
    return module.BenchmarkResult(
        backend=backend_name,
        n_simulations=n_simulations,
        execution_time=1.0,
        mean_estimate=3.14,
        throughput=float(n_simulations),
    )


def test_quick_mode_skips_large_sequential_runs(monkeypatch):
    module = _load_demo_module()
    calls = []

    def fake_run_benchmark(sim, n_simulations, backend, torch_device, n_workers, warmup=False):
        calls.append((backend, n_simulations, warmup))
        if warmup:
            return None
        return _fake_result(module, backend, torch_device, n_simulations)

    monkeypatch.setattr(module, "run_benchmark", fake_run_benchmark)
    results = module.run_benchmark_suite(
        sim=object(),
        simulation_sizes=[1_000_000, 5_000_000],
        backends=[("sequential", "cpu"), ("thread", "cpu")],
        n_workers=4,
        quick=True,
        max_sequential_size=1_000_000,
    )

    executed = {(b, n) for (b, n, warmup) in calls if not warmup}
    assert ("sequential", 5_000_000) not in executed
    assert ("sequential", 1_000_000) in executed
    assert ("thread", 1_000_000) in executed
    assert ("thread", 5_000_000) in executed
    assert [r.n_simulations for r in results["sequential"]] == [1_000_000]
    assert [r.n_simulations for r in results["thread"]] == [1_000_000, 5_000_000]


def test_non_quick_mode_keeps_full_execution_matrix(monkeypatch):
    module = _load_demo_module()
    calls = []

    def fake_run_benchmark(sim, n_simulations, backend, torch_device, n_workers, warmup=False):
        calls.append((backend, n_simulations, warmup))
        if warmup:
            return None
        return _fake_result(module, backend, torch_device, n_simulations)

    monkeypatch.setattr(module, "run_benchmark", fake_run_benchmark)
    module.run_benchmark_suite(
        sim=object(),
        simulation_sizes=[1_000_000, 5_000_000],
        backends=[("sequential", "cpu"), ("thread", "cpu")],
        n_workers=4,
        quick=False,
        max_sequential_size=1_000_000,
    )

    executed = {(b, n) for (b, n, warmup) in calls if not warmup}
    assert ("sequential", 1_000_000) in executed
    assert ("sequential", 5_000_000) in executed
    assert ("thread", 1_000_000) in executed
    assert ("thread", 5_000_000) in executed


def test_custom_max_sequential_size_override(monkeypatch):
    module = _load_demo_module()
    calls = []

    def fake_run_benchmark(sim, n_simulations, backend, torch_device, n_workers, warmup=False):
        calls.append((backend, n_simulations, warmup))
        if warmup:
            return None
        return _fake_result(module, backend, torch_device, n_simulations)

    monkeypatch.setattr(module, "run_benchmark", fake_run_benchmark)
    module.run_benchmark_suite(
        sim=object(),
        simulation_sizes=[100_000, 200_000],
        backends=[("sequential", "cpu"), ("thread", "cpu")],
        n_workers=4,
        quick=True,
        max_sequential_size=100_000,
    )

    executed = {(b, n) for (b, n, warmup) in calls if not warmup}
    assert ("sequential", 100_000) in executed
    assert ("sequential", 200_000) not in executed


def test_reporting_handles_missing_sequential_large_points(monkeypatch):
    module = _load_demo_module()

    def fake_run_benchmark(sim, n_simulations, backend, torch_device, n_workers, warmup=False):
        if warmup:
            return None
        return _fake_result(module, backend, torch_device, n_simulations)

    monkeypatch.setattr(module, "run_benchmark", fake_run_benchmark)
    results = module.run_benchmark_suite(
        sim=object(),
        simulation_sizes=[1_000_000, 5_000_000],
        backends=[("sequential", "cpu"), ("thread", "cpu")],
        n_workers=4,
        quick=True,
        max_sequential_size=1_000_000,
    )

    speedups = module.calculate_speedups(results, baseline="sequential")
    assert [n for n, _ in speedups["thread"]] == [1_000_000]
    summary = module.create_summary_table(results, [1_000_000, 5_000_000])
    assert "N/A" in summary


def test_parse_args_defaults_and_overrides():
    module = _load_demo_module()
    default_args = module.parse_args([])
    assert default_args.quick is False
    assert default_args.max_sequential_size == 1_000_000

    custom_args = module.parse_args(["--quick", "--max-sequential-size", "123456"])
    assert custom_args.quick is True
    assert custom_args.max_sequential_size == 123456
