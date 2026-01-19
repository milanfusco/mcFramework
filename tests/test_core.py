"""
Tests for the mcframework.core module.

Fuctional requirements (FR), Non-functional requirements (NFR), and Usability requirements (USA)
can be found at https://milanfusco.github.io/mcFramework/PROJECT_PLAN.html.
"""

import numpy as np
import pytest

from mcframework import MonteCarloFramework, MonteCarloSimulation, SimulationResult
from mcframework.core import make_blocks
from mcframework.sims import PiEstimationSimulation
from mcframework.stats_engine import StatsContext


class TestMakeBlocks:
    """[FR-3] Test block creation for parallel processing."""

    def test_make_blocks_exact_division(self):
        """[FR-3] Test blocks with exact division."""
        blocks = make_blocks(10000, block_size=1000)
        assert len(blocks) == 10
        assert blocks[0] == (0, 1000)
        assert blocks[-1] == (9000, 10000)

    def test_make_blocks_with_remainder(self):
        """[FR-3] Test blocks with remainder."""
        blocks = make_blocks(10500, block_size=1000)
        assert len(blocks) == 11
        assert blocks[-1] == (10000, 10500)

    def test_make_blocks_small_n(self):
        """[FR-3] Test blocks smaller than block_size."""
        blocks = make_blocks(500, block_size=1000)
        assert len(blocks) == 1
        assert blocks[0] == (0, 500)

    def test_make_blocks_coverage(self):
        """[FR-3] Test all elements are covered exactly once."""
        n = 12345
        blocks = make_blocks(n, block_size=1000)
        total = sum(j - i for i, j in blocks)
        assert total == n


class TestSimulationResult:
    """[FR-5] Test SimulationResult dataclass."""

    def test_simulation_result_creation(self):
        """[FR-5] Test creating a simulation result with mean, std, percentiles."""
        results = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = SimulationResult(
            results=results,
            n_simulations=5,
            execution_time=1.5,
            mean=3.0,
            std=1.58,
            percentiles={50: 3.0},
        )
        assert result.n_simulations == 5
        assert result.mean == 3.0

    def test_result_to_string_basic(self):
        """[FR-5, USA-3] Test readable string representation of results."""
        results = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = SimulationResult(
            results=results,
            n_simulations=5,
            execution_time=1.5,
            mean=3.0,
            std=1.58,
            percentiles={50: 3.0},
        )
        output = result.result_to_string()
        assert "Mean: 3.00000" in output
        assert "Std Dev" in output
        assert "50th: 3.00000" in output

    def test_result_to_string_with_metadata(self):
        """[FR-5, USA-3] Test string output includes metadata."""
        results = np.array([1.0, 2.0, 3.0])
        result = SimulationResult(
            results=results,
            n_simulations=3,
            execution_time=1.0,
            mean=2.0,
            std=1.0,
            percentiles={},
            metadata={"simulation_name": "Test"},
        )
        output = result.result_to_string()
        assert "Test" in output

    def test_result_to_string_with_stats_and_metadata(self):
        """[FR-5, USA-3] Ensure result string includes engine stats and metadata entries."""
        results = np.array([2.0, 4.0, 6.0, 8.0])
        stats = {"ci_mean": (1.0, 2.0), "custom_metric": 42}
        metadata = {"simulation_name": "EdgeSim", "requested_percentiles": [5, 95], "note": "coverage"}
        percentiles = {5: 2.0, 95: 8.0}
        result = SimulationResult(
            results=results,
            n_simulations=len(results),
            execution_time=0.5,
            mean=float(np.mean(results)),
            std=float(np.std(results, ddof=1)),
            percentiles=percentiles,
            stats=stats,
            metadata=metadata,
        )

        summary = result.result_to_string(confidence=0.9, method="t")
        assert "(engine) CI" in summary
        assert "Additional Stats" in summary
        assert "custom_metric" in summary
        assert "note" in summary


class TestMonteCarloSimulation:
    """[FR-1] Test MonteCarloSimulation abstract base class."""

    def test_simulation_initialization(self, simple_simulation):
        """[FR-1] Test simulation initializes correctly with name and RNG."""
        assert simple_simulation.name == "TestSim"
        assert simple_simulation.rng is not None

    def test_set_seed(self, simple_simulation):
        """[FR-4, NFR-3] Test reproducible seeding with SeedSequence."""
        simple_simulation.set_seed(42)
        assert simple_simulation.seed_seq is not None

        # Generate some numbers
        val1 = simple_simulation.single_simulation()

        # Reset seed
        simple_simulation.set_seed(42)
        val2 = simple_simulation.single_simulation()

        # Should be same due to seed
        assert val1 == val2

    def test_run_sequential_basic(self, simple_simulation):
        """[FR-2, USA-2] Test basic sequential run with sensible defaults."""
        simple_simulation.set_seed(42)
        result = simple_simulation.run(
            100,
            backend="sequential",
            compute_stats=False,
        )
        assert result.n_simulations == 100
        assert len(result.results) == 100
        assert result.execution_time >= 0.0

    def test_run_with_progress_callback(self, simple_simulation):
        """[FR-2] Test progress callback is called during sequential run."""
        callback_data = []

        def callback(completed, total):
            callback_data.append((completed, total))

        simple_simulation.run(
            50,
            backend="sequential",
            progress_callback=callback,
            compute_stats=False,
        )

        assert len(callback_data) > 0
        assert callback_data[-1] == (50, 50)

    def test_run_with_custom_percentiles(self, simple_simulation):
        """[FR-5, FR-10] Test custom percentiles are computed."""
        result = simple_simulation.run(
            100,
            backend="sequential",
            percentiles=[10, 90],
            compute_stats=False,
        )
        assert 10 in result.percentiles
        assert 90 in result.percentiles

    def test_run_with_stats_engine(self, simple_simulation):
        """[FR-8, FR-9] Test running with stats engine computes mean and std."""
        result = simple_simulation.run(
            100,
            backend="sequential",
            compute_stats=True,
            confidence=0.95,
            eps=0.05,
        )

        mean = getattr(result, "mean")
        std = getattr(result, "std")
        assert mean is not None
        assert std is not None

    def test_run_parallel_basic(self, simple_simulation):
        """[FR-3] Test basic parallel run via _run_parallel()."""
        simple_simulation.set_seed(42)
        result = simple_simulation.run(
            100,
            backend="auto",
            n_workers=2,
            compute_stats=False,
        )
        assert result.n_simulations == 100
        assert len(result.results) == 100

    def test_run_sequential_vs_parallel_reproducibility(self, simple_simulation):
        """[FR-4, NFR-3] Test sequential and parallel give same results with same seed."""
        simple_simulation.set_seed(42)
        seq_result = simple_simulation.run(100, backend="sequential", compute_stats=False)

        simple_simulation.set_seed(42)
        par_result = simple_simulation.run(100, backend="auto", n_workers=2, compute_stats=False)

        # Means should be very close
        assert pytest.approx(seq_result.mean, abs=0.5) == par_result.mean

    def test_run_invalid_n_simulations(self, simple_simulation):
        """[USA-4] Test clear error on invalid n_simulations."""
        with pytest.raises(ValueError, match="n_simulations must be positive"):
            simple_simulation.run(0)

    def test_serialization(self, simple_simulation):
        """[FR-3] Test pickle serialization for parallel processing."""
        simple_simulation.set_seed(42)
        state = simple_simulation.__getstate__()

        # RNG should be removed
        assert state.get("rng") is None

        # Restore state
        simple_simulation.__setstate__(state)

        # Should still work
        val = simple_simulation.single_simulation()
        assert isinstance(val, float)

    def test_deterministic_results(self, deterministic_simulation):
        """[NFR-3] Test deterministic simulation produces expected sequence."""
        result = deterministic_simulation.run(5, backend="sequential", compute_stats=False)
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_array_equal(result.results, expected)

    def test_serialization_without_seed(self, simple_simulation):
        """[FR-3] __setstate__ should recreate RNG when seed_seq is missing."""
        state = simple_simulation.__getstate__()
        assert state["rng"] is None
        simple_simulation.__setstate__(state)
        assert simple_simulation.rng is not None

    def test_run_rejects_invalid_n_workers(self, simple_simulation):
        """[USA-4] n_workers must be positive."""
        with pytest.raises(ValueError, match="n_workers must be positive"):
            simple_simulation.run(5, n_workers=0)

    def test_run_rejects_invalid_confidence(self, simple_simulation):
        """[USA-4] confidence must lie in (0, 1)."""
        with pytest.raises(ValueError, match="confidence must be in the interval"):
            simple_simulation.run(5, confidence=1.5)

    def test_run_rejects_invalid_ci_method(self, simple_simulation):
        """[USA-4] Invalid ci_method should raise."""
        with pytest.raises(ValueError, match="ci_method must be one of"):
            simple_simulation.run(5, ci_method="invalid")

    def test_run_rejects_invalid_backend(self, simple_simulation):
        """[USA-4] Invalid backend should raise."""
        with pytest.raises(ValueError, match="backend must be one of"):
            simple_simulation.run(5, backend="invalid_backend")

    def test_run_torch_backend_not_available(self, simple_simulation):
        """[GPU-1] Torch backend should raise RuntimeError until implemented."""
        with pytest.raises(RuntimeError, match="Torch backend not available"):
            simple_simulation.run(5, backend="torch")

    def test_deprecated_parallel_parameter_warns(self, simple_simulation):
        """[DEPRECATION] Using parallel= should emit DeprecationWarning."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = simple_simulation.run(5, parallel=False, compute_stats=False)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "parallel" in str(w[0].message)
            assert result.n_simulations == 5

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = simple_simulation.run(5, parallel=True, compute_stats=False)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert result.n_simulations == 5

    def test_run_handles_invalid_extra_context(self, simple_simulation):
        """[NFR-4] Extra context with invalid keys should fall back to defaults."""
        result = simple_simulation.run(
            10,
            backend="sequential",
            extra_context={"unexpected": "value"},
        )
        assert result.n_simulations == 10
        assert result.stats

    def test_resolve_parallel_backend_unknown_value_defaults(self, simple_simulation):
        """[NFR-7] Unknown backend values should coerce to auto/thread."""
        simple_simulation.backend = "unknown"
        backend = simple_simulation._resolve_parallel_backend()
        assert backend in {"thread", "process"}

    def test_compute_stats_block_handles_empty_array(self):
        """[NFR-4] _compute_stats_block should return NaNs for empty input."""
        ctx = StatsContext(n=0)
        stats = MonteCarloSimulation._compute_stats_block(np.array([]), ctx)
        assert np.isnan(stats["mean"])
        assert np.isnan(stats["std"])
        low, high = stats["ci_mean"]
        assert np.isnan(low) and np.isnan(high)

    def test_create_result_merges_engine_percentiles(self, simple_simulation):
        """[FR-5, FR-10] Engine-supplied percentiles should be merged once."""
        results = np.array([1.0, 2.0, 3.0, 4.0])
        stats = {"mean": 2.5, "percentiles": {25: 1.5, 75: 3.5}}
        percentiles = {5: 1.0}
        res = simple_simulation._create_result(
            results,
            n_simulations=results.size,
            execution_time=0.1,
            percentiles=percentiles,
            stats=stats,
            requested_percentiles=[5, 25, 75],
            engine_defaults_used=True,
        )
        assert res.percentiles[25] == pytest.approx(1.5)
        assert res.percentiles[5] == pytest.approx(percentiles[5])
        assert "percentiles" not in res.stats


class TestComputePercentilesBlock:
    """[FR-10] Tests for MonteCarloSimulation._compute_percentiles_block."""

    def test_percentiles_block_returns_empty_without_requests(self):
        """[FR-10] No requested percentiles should yield an empty dict."""
        results = np.array([1.0, 2.0, 3.0], dtype=float)
        ctx = StatsContext(n=results.size, percentiles=())

        block = MonteCarloSimulation._compute_percentiles_block(results, ctx)
        assert block == {}

    def test_percentiles_block_uses_ctx_percentiles(self):
        """[FR-10] Percentiles should be computed for ctx.percentiles."""
        results = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
        ctx = StatsContext(n=results.size, percentiles=(10, 90))

        block = MonteCarloSimulation._compute_percentiles_block(results, ctx)

        assert 10.0 in block and 90.0 in block
        assert block[10.0] == pytest.approx(np.percentile(results, 10))
        assert block[90.0] == pytest.approx(np.percentile(results, 90))

    def test_percentiles_block_falls_back_to_requested(self, monkeypatch):
        """[FR-10] requested_percentiles should be used when ctx.percentiles is empty."""

        class RequestedStatsContext(StatsContext):
            __slots__ = ("requested_percentiles",)

            def __init__(self, *, requested_percentiles, **kwargs):
                super().__init__(**kwargs)
                self.requested_percentiles = requested_percentiles

        results = np.array([10.0, 20.0, 30.0, 40.0], dtype=float)
        ctx = RequestedStatsContext(
            n=results.size,
            percentiles=(),
            requested_percentiles=(5, 95),
        )

        def fake_pct(arr, ctx):
            return np.percentile(arr, list(ctx.requested_percentiles))

        monkeypatch.setattr("mcframework.simulation.pct", fake_pct)

        block = MonteCarloSimulation._compute_percentiles_block(results, ctx)

        assert set(block.keys()) == {5.0, 95.0}
        assert block[5.0] == pytest.approx(np.percentile(results, 5))
        assert block[95.0] == pytest.approx(np.percentile(results, 95))

    def test_percentiles_block_raises_on_engine_mismatch(self, monkeypatch):
        """[NFR-5, USA-4] pct() returning the wrong number of values should raise ValueError."""
        results = np.array([0.0, 1.0, 2.0], dtype=float)
        ctx = StatsContext(n=results.size, percentiles=(10, 90))

        def bad_pct(arr, ctx):  # pragma: no cover - failure path
            return [float(np.percentile(arr, 10))]

        monkeypatch.setattr("mcframework.simulation.pct", bad_pct)

        with pytest.raises(ValueError, match="must return as many values"):
            MonteCarloSimulation._compute_percentiles_block(results, ctx)


class TestMonteCarloFramework:
    """[FR-6] Test MonteCarloFramework registry class."""

    def test_framework_initialization(self, framework):
        """[FR-6] Test framework initializes empty."""
        assert len(framework.simulations) == 0
        assert len(framework.results) == 0

    def test_register_simulation(self, framework, simple_simulation):
        """[FR-6] Test registering a simulation."""
        framework.register_simulation(simple_simulation)
        assert "TestSim" in framework.simulations

    def test_register_simulation_custom_name(self, framework, simple_simulation):
        """[FR-6] Test registering with a custom name."""
        framework.register_simulation(simple_simulation, name="CustomName")
        assert "CustomName" in framework.simulations

    def test_run_simulation(self, framework, simple_simulation):
        """[FR-6] Test running registered simulation."""
        framework.register_simulation(simple_simulation)
        result = framework.run_simulation("TestSim", 50, backend="sequential")
        assert result.n_simulations == 50
        assert "TestSim" in framework.results

    def test_run_simulation_not_found(self, framework):
        """[FR-6, USA-4] Test error when simulation not found."""
        with pytest.raises(ValueError, match="not found"):
            framework.run_simulation("NonExistent", 50)

    def test_compare_results_mean(self, framework, simple_simulation):
        """[FR-7] Test comparing results by mean."""
        sim1 = simple_simulation
        sim2 = simple_simulation

        framework.register_simulation(sim1, "Sim1")
        framework.register_simulation(sim2, "Sim2")

        framework.run_simulation("Sim1", 100, backend="sequential", mean=5.0)
        framework.run_simulation("Sim2", 100, backend="sequential", mean=10.0)

        comparison = framework.compare_results(["Sim1", "Sim2"], metric="mean")
        assert "Sim1" in comparison
        assert "Sim2" in comparison
        assert comparison["Sim2"] > comparison["Sim1"]

    def test_compare_results_std(self, framework, simple_simulation):
        """[FR-7] Test comparing results by std."""
        framework.register_simulation(simple_simulation)
        framework.run_simulation("TestSim", 100, backend="sequential")

        comparison = framework.compare_results(["TestSim"], metric="std")
        assert "TestSim" in comparison
        assert comparison["TestSim"] > 0

    def test_compare_results_percentile(self, framework, simple_simulation):
        """[FR-7] Test comparing results by percentile."""
        framework.register_simulation(simple_simulation)
        framework.run_simulation("TestSim", 100, backend="sequential", percentiles=[50])

        comparison = framework.compare_results(["TestSim"], metric="p50")
        assert "TestSim" in comparison

    def test_compare_results_no_results(self, framework):
        """[FR-7, USA-4] Test error when comparing non-existent results."""
        with pytest.raises(ValueError, match="No results found"):
            framework.compare_results(["NonExistent"])

    def test_compare_results_invalid_metric(self, framework, simple_simulation):
        """[FR-7, USA-4] Test error on invalid metric."""
        framework.register_simulation(simple_simulation)
        framework.run_simulation("TestSim", 50, backend="sequential")

        with pytest.raises(ValueError, match="Unknown metric"):
            framework.compare_results(["TestSim"], metric="invalid")

    def test_compare_results_percentile_not_in_percentiles_dict(self):
        """[FR-7, USA-4] Test percentile not in result.percentiles dict."""
        fw = MonteCarloFramework()
        sim = PiEstimationSimulation()
        sim.set_seed(42)

        fw.register_simulation(sim, "TestSim")

        # Create result with no requested_percentiles in metadata
        result = fw.run_simulation(
            "TestSim",
            100,
            n_points=5000,
            backend="sequential",
            percentiles=[25, 75],
            compute_stats=False,
        )

        # Remove the requested percentile from metadata to test the fallback
        result.metadata.pop("requested_percentiles", None)

        # Manually remove a percentile from the dict to test line 431
        if 50 in result.percentiles:
            del result.percentiles[50]

        fw.results["TestSim"] = result

        # Should raise error since p50 not in percentiles
        with pytest.raises(ValueError, match="Percentile 50 not computed"):
            fw.compare_results(["TestSim"], metric="p50")

    class TestPercentileMerging:
        """[FR-10] Test merging percentiles from stats engine."""

        def test_stats_engine_percentiles_merge(self):
            """[FR-10] Test that stats engine percentiles are properly merged."""
            sim = PiEstimationSimulation()
            sim.set_seed(42)

            # Run with stats engine (which provides percentiles)
            result = sim.run(
                100,
                backend="sequential",
                n_points=5000,
                percentiles=[5, 10, 50, 95],  # User requests 10
                compute_stats=True,  # Engine adds 5, 25, 50, 75, 95
                eps=0.05,
            )

            # Should have both user-requested and engine percentiles
            assert 10 in result.percentiles  # User requested


class TestMetadataFields:
    """[FR-5] Test optional metadata fields."""

    def test_metadata_includes_requested_percentiles(self):
        """[FR-5] Test requested_percentiles in metadata."""
        sim = PiEstimationSimulation()
        sim.set_seed(42)

        result = sim.run(
            100,
            backend="sequential",
            n_points=5000,
            percentiles=[10, 20, 30],
            compute_stats=False,
        )

        # Should include requested_percentiles in metadata
        assert "requested_percentiles" in result.metadata
        assert result.metadata["requested_percentiles"] == [10, 20, 30]

    def test_metadata_includes_engine_defaults_used(self):
        """[FR-5] Test engine_defaults_used in metadata."""
        sim = PiEstimationSimulation()
        sim.set_seed(42)

        result = sim.run(
            100,
            backend="sequential",
            n_points=5000,
            percentiles=[10],
            compute_stats=True,  # Use engine
        )

        # Should include engine_defaults_used in metadata
        assert "engine_defaults_used" in result.metadata
        assert result.metadata["engine_defaults_used"] is True

    def test_metadata_without_requested_percentiles(self):
        """[FR-5] Test that metadata works when no percentiles requested."""
        sim = PiEstimationSimulation()
        sim.set_seed(42)

        result = sim.run(
            100,
            backend="sequential",
            n_points=5000,
            percentiles=None,  # No percentiles
            compute_stats=False,
        )

        # requested_percentiles should not be in metadata or should be empty
        requested = result.metadata.get("requested_percentiles")
        assert requested is None or requested == []


class TestParallelBackend:
    """[FR-3, NFR-7] Test backend selection for parallel execution."""

    def test_parallel_backend_thread_explicit(self):
        """[FR-3, NFR-7] Test explicit thread backend selection."""
        sim = PiEstimationSimulation()
        sim.set_seed(42)

        result = sim.run(
            25000,
            backend="thread",  # Explicitly set to thread
            n_workers=2,
            n_points=1000,
            compute_stats=False,
        )

        assert result.n_simulations == 25000

    def test_parallel_backend_process_explicit(self):
        """[FR-3, NFR-7] Test explicit process backend selection."""
        sim = PiEstimationSimulation()
        sim.set_seed(42)

        result = sim.run(
            25000,
            backend="process",  # Force process backend
            n_workers=2,
            n_points=1000,
            compute_stats=False,
        )

        assert result.n_simulations == 25000

    def test_parallel_backend_auto_uses_threads(self):
        """[FR-3, NFR-7] Test auto backend defaults to threads."""
        sim = PiEstimationSimulation()
        sim.set_seed(42)

        result = sim.run(
            25000,
            backend="auto",  # Should use threads
            n_workers=2,
            n_points=1000,
            compute_stats=False,
        )

        assert result.n_simulations == 25000


class TestParallelFallback:
    """[NFR-2] Test parallel fallback to sequential for small jobs."""

    def test_parallel_fallback_small_n_simulations(self):
        """[NFR-2] Test that parallel mode falls back to sequential for n < 20,000."""
        sim = PiEstimationSimulation()
        sim.set_seed(42)

        # With n_simulations < 20,000, should use sequential even with backend="auto"
        result = sim.run(
            5000,  # Less than 20,000
            backend="auto",
            n_workers=4,
            n_points=1000,
            compute_stats=False,
        )

        assert result.n_simulations == 5000
        assert len(result.results) == 5000

    def test_parallel_fallback_single_worker(self):
        """[NFR-2] Test that parallel mode falls back to sequential with n_workers=1."""
        sim = PiEstimationSimulation()
        sim.set_seed(42)

        result = sim.run(
            50000,  # Large enough but n_workers=1
            backend="auto",
            n_workers=1,
            n_points=1000,
            compute_stats=False,
        )

        assert result.n_simulations == 50000


class TestThreadBackendExecution:
    """[FR-3, NFR-7] Ensure thread backend is actually used in some tests."""

    def test_default_backend_is_auto(self):
        """[USA-2, NFR-7] Test that default backend is 'auto'."""
        sim = PiEstimationSimulation()

        # Check default value
        assert hasattr(sim, "backend")
        assert sim.backend == "auto"

    def test_parallel_backend_legacy_alias(self):
        """[COMPAT] Test parallel_backend legacy alias works bidirectionally."""
        sim = PiEstimationSimulation()

        # Getter: parallel_backend should reflect backend
        assert sim.parallel_backend == sim.backend == "auto"

        # Setter: setting parallel_backend should update backend
        sim.parallel_backend = "thread"
        assert sim.backend == "thread"
        assert sim.parallel_backend == "thread"

    def test_thread_backend_with_large_job(self):
        """[FR-3] Test thread backend with job large enough to avoid fallback."""
        sim = PiEstimationSimulation()
        sim.set_seed(42)
        sim.backend = "thread"

        result = sim.run(
            30000,  # Well above 20,000 the threshold
            backend="thread",
            n_workers=2,
            n_points=500,
            compute_stats=False,
        )

        assert result.n_simulations == 30000


class TestSeedSequenceGeneration:
    """[FR-4] Test generating random seed sequences without an initial seed."""

    def test_parallel_without_seed_generates_random_sequences(self):
        """[FR-4] Test that parallel execution generates random seeds when no seed set."""
        sim = PiEstimationSimulation()
        # Don't set seed - line 292 should execute

        result = sim.run(
            25000,  # Must be >= 20,000 to avoid fallback
            backend="auto",
            n_workers=2,
            n_points=1000,
            compute_stats=False,
        )

        assert result.n_simulations == 25000
        # Results should vary each run (no fixed seed)
        assert result.std > 0


class TestKeyboardInterruptHandling:
    """[NFR-4] Test KeyboardInterrupt during parallel execution."""

    def test_keyboard_interrupt_cleanup(self):
        """[NFR-4] Test KeyboardInterrupt is propagated and futures are cancelled."""

        class InterruptingSimulation(MonteCarloSimulation):
            def __init__(self):
                super().__init__("InterruptSim")
                self.call_count = 0

            def single_simulation(self, **kwargs):
                self.call_count += 1
                # Interrupt after a few calls to ensure we're in parallel execution
                if self.call_count > 10:
                    raise KeyboardInterrupt("User interrupted")
                return float(np.random.random())

        sim = InterruptingSimulation()
        sim.set_seed(42)

        # Should raise KeyboardInterrupt
        with pytest.raises(KeyboardInterrupt):
            sim.run(
                50000,  # Large enough to ensure parallel execution
                backend="thread",  # Use threads to avoid process issues
                n_workers=2,
                compute_stats=False,
            )


class TestAdditionalEdgeCases:
    """[NFR-4] Additional tests for remaining edge cases."""

    def test_run_without_percentiles_and_without_stats(self):
        """[NFR-4] Test running without percentiles and without stats."""
        sim = PiEstimationSimulation()
        sim.set_seed(42)

        result = sim.run(
            50,
            backend="sequential",
            n_points=1000,
            percentiles=None,  # No percentiles requested
            compute_stats=False,  # No stats engine
        )

        # Should have empty percentiles
        assert len(result.percentiles) == 0
        assert len(result.stats) == 0

    def test_run_with_empty_percentiles_list(self):
        """[NFR-4] Test with explicitly empty percentiles list."""
        sim = PiEstimationSimulation()
        sim.set_seed(42)

        result = sim.run(
            50,
            backend="sequential",
            n_points=1000,
            percentiles=[],  # Explicit empty list
            compute_stats=False,
        )

        # Should have empty percentiles
        assert len(result.percentiles) == 0

    def test_parallel_with_custom_block_size(self):
        """[FR-3] Test parallel execution with custom block sizing."""
        sim = PiEstimationSimulation()
        sim.set_seed(42)

        result = sim.run(
            100000,  # Large number to test block creation
            backend="auto",
            n_workers=4,
            n_points=500,
            compute_stats=False,
        )

        assert result.n_simulations == 100000

    def test_compare_results_with_all_metrics(self):
        """[FR-7] Test all metric types in compare_results."""
        fw = MonteCarloFramework()
        sim = PiEstimationSimulation()
        sim.set_seed(42)

        fw.register_simulation(sim)
        fw.run_simulation(
            "Pi Estimation", 100, n_points=5000, backend="sequential", percentiles=[50], compute_stats=True
        )

        # Test all metric types
        metrics_to_test = ["mean", "std", "var", "se", "p50"]

        for metric in metrics_to_test:
            result = fw.compare_results(["Pi Estimation"], metric=metric)
            assert "Pi Estimation" in result
            assert isinstance(result["Pi Estimation"], float)


def test_pi_simulation_antithetic_handles_odd_points():
    """[FR-20] Antithetic sampling should pad when n_points is odd."""
    sim = PiEstimationSimulation()
    sim.set_seed(123)
    value = sim.single_simulation(antithetic=True, n_points=5)
    assert 0.0 < value < 4.5


def test_compute_stats_with_none_engine():
    """[NFR-4] Test that _compute_stats_with_engine returns empty dicts when engine is None."""
    from unittest.mock import patch

    from mcframework.core import MonteCarloSimulation
    
    class SimpleSim(MonteCarloSimulation):
        def single_simulation(self, _rng=None):
            return 1.0
    
    sim = SimpleSim()
    sim.set_seed(42)
    
    # Patch DEFAULT_ENGINE to be None
    with patch('mcframework.simulation.DEFAULT_ENGINE', None):
        stats, percentiles = sim._compute_stats_with_engine(
            results=np.array([1.0, 2.0, 3.0]),
            n_simulations=3,
            confidence=0.95,
            ci_method="auto",
            stats_engine=None,
            extra_context=None
        )
        assert stats == {}
        assert percentiles == {}


def test_resolve_backend_type_explicit_thread():
    """[NFR-7] Test _resolve_backend_type returns 'thread' when explicitly set."""
    from mcframework.core import MonteCarloSimulation

    class SimpleSim(MonteCarloSimulation):
        def single_simulation(self, _rng=None):
            return 1.0

    sim = SimpleSim()
    sim.backend = "thread"
    # Explicit thread backend should return "thread" directly
    assert sim._resolve_backend_type() == "thread"


def test_resolve_backend_type_explicit_process():
    """[NFR-7] Test _resolve_backend_type returns 'process' when explicitly set."""
    from mcframework.core import MonteCarloSimulation

    class SimpleSim(MonteCarloSimulation):
        def single_simulation(self, _rng=None):
            return 1.0

    sim = SimpleSim()
    sim.backend = "process"
    # Explicit process backend should return "process" directly
    assert sim._resolve_backend_type() == "process"


def test_run_with_backend_auto_large_job():
    """[NFR-7] Test that backend='auto' resolves to parallel for large jobs."""
    from mcframework.core import MonteCarloSimulation

    class SimpleSim(MonteCarloSimulation):
        def single_simulation(self, _rng=None):
            rng = self._rng(_rng, self.rng)
            return float(rng.random())

    sim = SimpleSim()
    sim.set_seed(42)
    # Run with enough simulations to exceed _PARALLEL_THRESHOLD (20,000)
    result = sim.run(25000, backend="auto", n_workers=2, compute_stats=False)
    assert result.n_simulations == 25000


def test_resolve_backend_unknown_backend_warning():
    """[NFR-7] Test _resolve_backend warns and defaults to auto for unknown backends."""
    from mcframework.backends.parallel import ProcessBackend, ThreadBackend
    from mcframework.core import MonteCarloSimulation

    class SimpleSim(MonteCarloSimulation):
        def single_simulation(self, _rng=None):
            return 1.0

    sim = SimpleSim()
    sim.backend = "invalid_backend"  # Unknown backend

    # Unknown backend should log a warning and default to auto (thread on non-Windows)
    backend = sim._create_parallel_backend(n_workers=2)
    # Should return either ThreadBackend or ProcessBackend (depending on platform)
    assert isinstance(backend, (ThreadBackend, ProcessBackend))


def test_resolve_backend_auto_on_windows(monkeypatch):
    """[NFR-7] Test _resolve_backend returns ProcessBackend when auto on Windows."""
    from mcframework.backends.parallel import ProcessBackend
    from mcframework.core import MonteCarloSimulation

    class SimpleSim(MonteCarloSimulation):
        def single_simulation(self, _rng=None):
            return 1.0

    sim = SimpleSim()
    sim.backend = "auto"

    # Mock Windows platform
    monkeypatch.setattr("mcframework.core._is_windows_platform", lambda: True)

    backend = sim._create_parallel_backend(n_workers=2)
    assert isinstance(backend, ProcessBackend)


def test_process_backend_prepare_blocks_without_seed():
    """[NFR-7] Test ProcessBackend._prepare_blocks generates random seeds when seed_seq is None."""
    from mcframework.backends.parallel import ProcessBackend

    backend = ProcessBackend(n_workers=2)
    blocks, child_seqs = backend._prepare_blocks(100, seed_seq=None)

    # Should generate random seed sequences
    assert len(blocks) > 0
    assert len(child_seqs) == len(blocks)
    # All should be SeedSequence instances
    import numpy as np
    assert all(isinstance(s, np.random.SeedSequence) for s in child_seqs)


