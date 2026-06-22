"""Tests for the mcframework.benchmark subsystem."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest

import mcframework.benchmark as bm
from mcframework.benchmark import (
    BackendSpec,
    BenchmarkReport,
    BenchmarkResult,
    benchmark_run,
    default_backends,
    run_suite,
    system_info,
)


def _res(backend: str, n: int, t: float) -> BenchmarkResult:
    return BenchmarkResult(
        backend=backend, device="cpu", n_simulations=n,
        execution_time=t, throughput=n / t, mean_estimate=3.14,
    )


def _fake_sysinfo() -> dict:
    return {
        "torch_version": None, "mps_available": False, "cuda_available": False,
        "platform": "t", "machine": "x", "processor": "x", "python": "3.13", "cpu_count": 1,
    }


def _minimal_report() -> BenchmarkReport:
    return BenchmarkReport(results=[_res("sequential", 1000, 1.0)])


def _stub_main(monkeypatch, *, run_suite_fn=None, backends=None):
    monkeypatch.setattr(bm, "run_suite", run_suite_fn or (lambda *a, **k: _minimal_report()))
    monkeypatch.setattr(
        bm, "default_backends",
        lambda **kw: backends if backends is not None else [BackendSpec("sequential", "sequential")],
    )
    monkeypatch.setattr(bm, "system_info", _fake_sysinfo)


# --------------------------------------------------------------------------- #
# BenchmarkReport: speedups / summary / best / by_backend / to_dict
# --------------------------------------------------------------------------- #


def test_speedups_relative_to_baseline():
    report = BenchmarkReport(
        results=[
            _res("sequential", 1000, 1.0),
            _res("thread", 1000, 0.5),
            _res("sequential", 2000, 2.0),
            _res("thread", 2000, 0.5),
        ],
        sizes=[1000, 2000],
    )
    speedups = report.speedups()
    assert dict(speedups["thread"]) == {1000: 2.0, 2000: 4.0}
    assert dict(speedups["sequential"]) == {1000: 1.0, 2000: 1.0}


def test_speedups_empty_without_baseline():
    report = BenchmarkReport(results=[_res("thread", 1000, 0.5)])
    assert report.speedups() == {}


def test_summary_table_handles_missing_points_with_na():
    # thread is missing the 2000 cell -> summary must show N/A, not crash.
    report = BenchmarkReport(
        results=[
            _res("sequential", 1000, 1.0),
            _res("sequential", 2000, 2.0),
            _res("thread", 1000, 0.5),
        ],
        sizes=[1000, 2000],
    )
    table = report.summary_table()
    assert "N/A" in table
    assert "BENCHMARK SUMMARY" in table
    # Speedup only exists where both backend and baseline have a usable time.
    assert dict(report.speedups()["thread"]) == {1000: 2.0}


def test_by_backend_sorted_by_size():
    report = BenchmarkReport(results=[_res("thread", 2000, 1.0), _res("thread", 1000, 1.0)])
    rows = report.by_backend()["thread"]
    assert [r.n_simulations for r in rows] == [1000, 2000]


def test_best_returns_fastest_at_largest_size():
    report = BenchmarkReport(
        results=[
            _res("sequential", 2000, 2.0),
            _res("thread", 2000, 0.4),
            _res("torch-cpu", 2000, 0.1),
        ]
    )
    best = report.best()
    assert best is not None
    assert best.backend == "torch-cpu"


def test_best_none_when_empty():
    assert BenchmarkReport().best() is None


def test_to_dict_json_roundtrip():
    report = BenchmarkReport(
        results=[_res("sequential", 1000, 1.0), _res("thread", 1000, 0.5)],
        sizes=[1000],
        system={"machine": "test"},
    )
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["system"]["machine"] == "test"
    assert payload["sizes"] == [1000]
    assert len(payload["results"]) == 2
    assert payload["results"][0]["backend"] == "sequential"
    assert "peak_memory_mb" in payload["results"][0]


# --------------------------------------------------------------------------- #
# system_info / default_backends device gating
# --------------------------------------------------------------------------- #


def test_system_info_keys():
    info = system_info()
    for key in ("platform", "machine", "python", "cpu_count", "torch_available",
                "mps_available", "cuda_available"):
        assert key in info
    assert json.dumps(info)  # JSON-serializable


def test_default_backends_without_torch(monkeypatch):
    monkeypatch.setattr(bm, "_torch_version", lambda: None)
    labels = [s.label for s in default_backends()]
    assert labels == ["sequential", "thread", "process"]


def test_default_backends_with_mps(monkeypatch):
    monkeypatch.setattr(bm, "_torch_version", lambda: "2.9")
    monkeypatch.setattr(bm, "is_mps_available", lambda: True)
    monkeypatch.setattr(bm, "is_cuda_available", lambda: False)
    labels = [s.label for s in default_backends()]
    assert labels == ["sequential", "thread", "process", "torch-cpu", "torch-mps"]


def test_default_backends_cuda_gated(monkeypatch):
    monkeypatch.setattr(bm, "_torch_version", lambda: "2.9")
    monkeypatch.setattr(bm, "is_mps_available", lambda: False)
    monkeypatch.setattr(bm, "is_cuda_available", lambda: True)
    # CUDA is off by default even when available.
    assert "torch-cuda" not in [s.label for s in default_backends()]
    # Opt in explicitly.
    labels = [s.label for s in default_backends(include_cuda=True)]
    assert "torch-cuda" in labels


# --------------------------------------------------------------------------- #
# benchmark_run
# --------------------------------------------------------------------------- #


class _FakeSim:
    """Minimal simulation double: records calls, returns a fixed mean/time."""

    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.run_calls = 0
        self.seeds: list[int] = []

    def set_seed(self, seed: int) -> None:
        self.seeds.append(seed)

    def run(self, n, *, backend, torch_device, n_workers, compute_stats):
        self.run_calls += 1
        if self.fail:
            raise RuntimeError("backend unavailable")
        return SimpleNamespace(mean=3.14159, execution_time=0.001)


def test_benchmark_run_returns_result():
    sim = _FakeSim()
    spec = BackendSpec("torch-cpu", "torch", "cpu")
    result = benchmark_run(sim, 1000, spec, repeats=3)
    assert result is not None
    assert result.backend == "torch-cpu"
    assert result.device == "cpu"
    assert result.n_simulations == 1000
    assert result.throughput > 0
    assert sim.run_calls == 3  # repeats honored


def test_benchmark_run_warmup_returns_none():
    sim = _FakeSim()
    assert benchmark_run(sim, 1000, BackendSpec("seq", "sequential"), warmup=True) is None
    assert sim.run_calls == 1


def test_benchmark_run_returns_none_on_failure():
    sim = _FakeSim(fail=True)
    assert benchmark_run(sim, 1000, BackendSpec("seq", "sequential")) is None


# --------------------------------------------------------------------------- #
# run_suite quick / max-sequential skip logic (ported from the old demo tests)
# --------------------------------------------------------------------------- #


def _fake_benchmark_run_factory(calls):
    def fake(sim, n, spec, *, n_workers=None, repeats=1, seed=42, warmup=False):
        calls.append((spec.backend, n, warmup))
        if warmup:
            return None
        return _res(spec.label, n, 1.0)
    return fake


def test_run_suite_quick_skips_large_sequential(monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setattr(bm, "benchmark_run", _fake_benchmark_run_factory(calls))
    specs = [BackendSpec("sequential", "sequential"), BackendSpec("thread", "thread")]
    report = run_suite(object(), [1_000_000, 5_000_000], specs,
                       quick=True, max_sequential_size=1_000_000)
    executed = {(b, n) for b, n, warmup in calls if not warmup}
    assert ("sequential", 5_000_000) not in executed
    assert ("sequential", 1_000_000) in executed
    assert ("thread", 5_000_000) in executed
    assert [r.n_simulations for r in report.by_backend()["sequential"]] == [1_000_000]


def test_run_suite_full_matrix(monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setattr(bm, "benchmark_run", _fake_benchmark_run_factory(calls))
    specs = [BackendSpec("sequential", "sequential"), BackendSpec("thread", "thread")]
    run_suite(object(), [1_000_000, 5_000_000], specs, quick=False)
    executed = {(b, n) for b, n, warmup in calls if not warmup}
    assert executed == {
        ("sequential", 1_000_000), ("sequential", 5_000_000),
        ("thread", 1_000_000), ("thread", 5_000_000),
    }


def test_run_suite_custom_max_sequential_size(monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setattr(bm, "benchmark_run", _fake_benchmark_run_factory(calls))
    specs = [BackendSpec("sequential", "sequential")]
    run_suite(object(), [100_000, 200_000], specs, quick=True, max_sequential_size=100_000)
    executed = {(b, n) for b, n, warmup in calls if not warmup}
    assert ("sequential", 100_000) in executed
    assert ("sequential", 200_000) not in executed


def test_run_suite_progress_callback(monkeypatch):
    monkeypatch.setattr(bm, "benchmark_run", _fake_benchmark_run_factory([]))
    seen: list[tuple] = []
    specs = [BackendSpec("sequential", "sequential"), BackendSpec("thread", "thread")]
    run_suite(object(), [1000], specs, progress=lambda label, done, total: seen.append((done, total)))
    assert seen == [(1, 2), (2, 2)]


# --------------------------------------------------------------------------- #
# plot_benchmarks
# --------------------------------------------------------------------------- #


def test_plot_benchmarks_returns_figure():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    report = BenchmarkReport(
        results=[
            _res("sequential", 1000, 1.0), _res("sequential", 2000, 2.0),
            _res("thread", 1000, 0.5), _res("thread", 2000, 0.5),
        ],
        sizes=[1000, 2000],
        system=system_info(),
    )
    fig = plot = bm.plot_benchmarks(report)
    assert len(fig.axes) == 4
    matplotlib.pyplot.close(plot)


def test_plot_benchmarks_raises_on_empty():
    pytest.importorskip("matplotlib")
    with pytest.raises(ValueError, match="no results"):
        bm.plot_benchmarks(BenchmarkReport())


def test_best_none_when_size_has_no_candidates():
    report = BenchmarkReport(results=[_res("thread", 1000, 1.0)])
    assert report.best(n=2000) is None


def test_summary_table_omits_speedup_section_without_baseline():
    report = BenchmarkReport(results=[_res("thread", 1000, 1.0)])
    table = report.summary_table()
    assert "Speedup vs Sequential" not in table
    assert "Best performer" in table


def test_summary_table_includes_best_speedup_line():
    report = BenchmarkReport(
        results=[_res("sequential", 1000, 1.0), _res("thread", 1000, 0.5)],
    )
    table = report.summary_table()
    assert "Speedup:" in table
    assert "faster than sequential" in table


def test_torch_version_none_when_import_fails(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert bm._torch_version() is None


def test_default_backends_respects_include_torch_false(monkeypatch):
    monkeypatch.setattr(bm, "_torch_version", lambda: "2.9")
    labels = [s.label for s in default_backends(include_torch=False)]
    assert labels == ["sequential", "thread", "process"]


def test_benchmark_run_infinite_throughput_on_zero_time(monkeypatch):
    class _InstantSim:
        def set_seed(self, seed):
            pass

        def run(self, n, *, backend, torch_device, n_workers, compute_stats):
            return SimpleNamespace(mean=3.14)

    monkeypatch.setattr(bm.time, "perf_counter", lambda: 0.0)
    result = benchmark_run(_InstantSim(), 1000, BackendSpec("seq", "sequential"))
    assert result is not None
    assert result.throughput == float("inf")


def test_benchmark_run_warmup_failure_is_silent(caplog):
    sim = _FakeSim(fail=True)
    with caplog.at_level(logging.WARNING, logger="mcframework.benchmark"):
        assert benchmark_run(sim, 1000, BackendSpec("seq", "sequential"), warmup=True) is None
    assert caplog.text == ""


def test_run_suite_omits_failed_cells(monkeypatch):
    def fake(sim, n, spec, *, n_workers=None, repeats=1, seed=42, warmup=False):
        if warmup:
            return None
        return None if spec.backend == "thread" else _res(spec.label, n, 1.0)

    monkeypatch.setattr(bm, "benchmark_run", fake)
    specs = [BackendSpec("sequential", "sequential"), BackendSpec("thread", "thread")]
    report = run_suite(object(), [1000], specs)
    assert [r.backend for r in report.results] == ["sequential"]


def test_color_map_assigns_fallback_for_unknown_labels():
    colors = bm._color_map(["custom-a", "custom-b"])
    assert colors["custom-a"] != colors["custom-b"]


def test_format_system_info_lists_accelerators():
    text = bm._format_system_info({"torch_version": "2.9", "mps_available": True, "cuda_available": True,
                                     "platform": "Darwin", "machine": "arm64", "processor": "arm",
                                     "python": "3.13", "cpu_count": 8})
    assert "MPS, CUDA" in text


def test_plot_benchmarks_hides_bar_without_non_baseline():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    report = BenchmarkReport(results=[_res("sequential", 1000, 1.0), _res("sequential", 2000, 2.0)])
    fig = bm.plot_benchmarks(report)
    assert not fig.axes[2].get_visible()
    matplotlib.pyplot.close(fig)


def test_plot_benchmarks_hides_scale_with_single_speedup_point():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    report = BenchmarkReport(
        results=[_res("sequential", 1000, 1.0), _res("thread", 1000, 0.5)],
        system={"machine": "test", "cpu_count": 4, "python": "3.13"},
    )
    fig = bm.plot_benchmarks(report)
    assert not fig.axes[3].get_visible()
    matplotlib.pyplot.close(fig)


def test_plot_benchmarks_custom_subtitle():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    report = BenchmarkReport(
        results=[_res("sequential", 1000, 1.0), _res("thread", 1000, 0.5)],
    )
    fig = bm.plot_benchmarks(report, subtitle="custom subtitle")
    assert fig is not None
    matplotlib.pyplot.close(fig)


def test_parse_args_defaults_and_overrides():
    defaults = bm._parse_args([])
    assert defaults.quick is False
    assert defaults.repeats == 1
    assert defaults.save is None

    custom = bm._parse_args(["--quick", "--sizes", "1000,2000", "--json", "out.json", "--no-show"])
    assert custom.quick is True
    assert custom.sizes == "1000,2000"
    assert custom.json == "out.json"
    assert custom.no_show is True


def test_main_default_sizes(monkeypatch):
    captured: dict = {}

    def capture_sizes(sim, sizes, specs, **kw):
        captured["sizes"] = sizes
        return BenchmarkReport()

    _stub_main(monkeypatch, run_suite_fn=capture_sizes, backends=[])
    bm.main(["--no-show"])
    assert captured["sizes"] == list(bm._DEFAULT_SIZES)


def test_main_quick_no_show(monkeypatch, capsys):
    captured: dict = {}

    def capture_sizes(sim, sizes, specs, **kw):
        captured["sizes"] = sizes
        return _minimal_report()

    _stub_main(monkeypatch, run_suite_fn=capture_sizes)
    rc = bm.main(["--quick", "--no-show"])
    assert rc == 0
    assert captured["sizes"] == list(bm._QUICK_SIZES)
    assert "SYSTEM INFO" in capsys.readouterr().out


def test_main_default_sizes_and_json(tmp_path, monkeypatch):
    out = tmp_path / "report.json"
    _stub_main(monkeypatch)
    rc = bm.main(["--json", str(out), "--no-show"])
    assert rc == 0
    assert out.exists()


def test_main_custom_sizes(monkeypatch):
    captured: dict = {}

    def capture_sizes(sim, sizes, specs, **kw):
        captured["sizes"] = sizes
        return BenchmarkReport()

    _stub_main(monkeypatch, run_suite_fn=capture_sizes, backends=[])
    bm.main(["--sizes", "1000,2000", "--no-show"])
    assert captured["sizes"] == [1000, 2000]


def test_main_plot_import_error(monkeypatch, capsys):
    _stub_main(monkeypatch)

    def _no_plot(*args, **kwargs):
        raise ImportError("no matplotlib")

    monkeypatch.setattr(bm, "plot_benchmarks", _no_plot)
    rc = bm.main(["--save", "out.png"])
    assert rc == 0
    assert "[plot skipped]" in capsys.readouterr().out


def test_main_show_without_save(monkeypatch):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    shown = {"called": False}
    _stub_main(monkeypatch)
    monkeypatch.setattr(bm, "plot_benchmarks", lambda report: matplotlib.pyplot.figure())
    monkeypatch.setattr(matplotlib.pyplot, "show", lambda: shown.__setitem__("called", True))
    bm.main([])
    assert shown["called"]


def test_main_savefig_and_show(monkeypatch, tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    out = tmp_path / "bench.png"
    fake_fig = matplotlib.pyplot.figure()
    shown = {"called": False}

    _stub_main(monkeypatch)
    monkeypatch.setattr(bm, "plot_benchmarks", lambda report: fake_fig)
    monkeypatch.setattr(matplotlib.pyplot, "show", lambda: shown.__setitem__("called", True))
    bm.main(["--save", str(out)])
    assert out.exists()
    assert shown["called"]

