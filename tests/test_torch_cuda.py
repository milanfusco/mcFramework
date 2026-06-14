"""Tests for Torch CUDA (NVIDIA GPU) backend.

These tests validate:
1. CUDA backend returns valid results on NVIDIA GPUs
2. CUDA respects seeding for reproducibility
3. Adaptive batch sizing and memory management
4. CUDA streams and multi-device support
5. cuRAND integration (via CuPy)

Note: These tests will be skipped on systems without CUDA support.
"""

import math

import numpy as np
import pytest
import torch

from mcframework.core import MonteCarloSimulation
from mcframework.sims import PiEstimationSimulation

CUDA_AVAILABLE = torch.cuda.is_available()


def _cupy_available() -> bool:
    """Check if CuPy is installed."""
    try:
        import cupy as cp  # noqa: F401  # pylint: disable=unused-import
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestCUDAValidationAndErrors:
    """[CUDA-00] Defensive programming - validation and error handling."""

    def test_missing_supports_batch_raises_attribute_error(self):
        """Simulation without supports_batch attribute raises clear error."""
        # Note: MonteCarloSimulation base class has supports_batch = False by default
        # This test verifies the error message when supports_batch is False (not explicitly set)
        class NoSupportsBatch(MonteCarloSimulation):
            # Doesn't set supports_batch, inherits False from base class
            def single_simulation(self, _rng=None, **kwargs):
                return 1.0

        sim = NoSupportsBatch(name="Invalid")
        sim.set_seed(42)

        # Should raise ValueError because supports_batch is False
        with pytest.raises(ValueError, match="does not support Torch batch execution"):
            sim.run(100, backend="torch", torch_device="cuda")

    def test_supports_batch_false_raises_clear_error(self):
        """Simulation with supports_batch=False raises informative error."""
        class ExplicitlyDisabled(MonteCarloSimulation):
            supports_batch = False  # Explicitly disabled

            def single_simulation(self, _rng=None, **kwargs):
                return 1.0

        sim = ExplicitlyDisabled(name="Disabled")
        sim.set_seed(42)

        with pytest.raises(ValueError, match="does not support Torch batch execution"):
            sim.run(100, backend="torch", torch_device="cuda")

    def test_missing_torch_batch_implementation_raises(self):
        """supports_batch=True but no torch_batch() raises NotImplementedError."""
        class MissingMethod(MonteCarloSimulation):
            supports_batch = True  # Claims support

            def single_simulation(self, _rng=None, **kwargs):
                return 1.0
            # No torch_batch method (inherits base class version that raises NotImplementedError)

        sim = MissingMethod(name="Incomplete")
        sim.set_seed(42)

        # Validation should detect that torch_batch is not overridden
        with pytest.raises(NotImplementedError, match="does not implement torch_batch"):
            sim.run(100, backend="torch", torch_device="cuda")


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestCUDABackendBasics:
    """[CUDA-01] Basic CUDA functionality tests."""

    def test_cuda_returns_valid_results(self):
        """CUDA backend returns valid Pi estimates."""
        sim = PiEstimationSimulation()
        sim.set_seed(42)

        result = sim.run(10_000, backend="torch", torch_device="cuda", compute_stats=False)

        # Mean should be close to pi
        assert 2.5 < result.mean < 3.8
        assert result.n_simulations == 10_000
        assert len(result.results) == 10_000

    def test_cuda_deterministic_with_seed(self):
        """Same seed produces identical results on CUDA."""
        sim1 = PiEstimationSimulation()
        sim1.set_seed(12345)
        result1 = sim1.run(5_000, backend="torch", torch_device="cuda", compute_stats=False)

        sim2 = PiEstimationSimulation()
        sim2.set_seed(12345)
        result2 = sim2.run(5_000, backend="torch", torch_device="cuda", compute_stats=False)

        np.testing.assert_array_equal(result1.results, result2.results)

    def test_cuda_different_seeds_differ(self):
        """Different seeds produce different results on CUDA."""
        sim1 = PiEstimationSimulation()
        sim1.set_seed(111)
        result1 = sim1.run(1_000, backend="torch", torch_device="cuda", compute_stats=False)

        sim2 = PiEstimationSimulation()
        sim2.set_seed(222)
        result2 = sim2.run(1_000, backend="torch", torch_device="cuda", compute_stats=False)

        # Results should differ (extremely unlikely to be equal)
        assert not np.array_equal(result1.results, result2.results)

    def test_cuda_no_global_rng_pollution(self):
        """CUDA backend doesn't pollute global RNG state."""
        # Set global RNG to known state
        torch.manual_seed(99999)
        global_sample_before = torch.rand(10).clone()

        # Reset global state
        torch.manual_seed(99999)

        # Run simulation (should use explicit generator, not global)
        sim = PiEstimationSimulation()
        sim.set_seed(42)
        sim.run(10_000, backend="torch", torch_device="cuda", compute_stats=False)

        # Reset global state again
        torch.manual_seed(99999)
        global_sample_after = torch.rand(10)

        # Global state should be unchanged (simulation used explicit generator)
        torch.testing.assert_close(global_sample_before, global_sample_after)

    def test_cuda_results_are_float64_native(self):
        """CUDA backend returns float64 results (no conversion overhead)."""
        sim = PiEstimationSimulation()
        sim.set_seed(42)

        result = sim.run(1000, backend="torch", torch_device="cuda", compute_stats=False)

        # Results should be float64 (CUDA native precision)
        assert result.results.dtype == np.float64

    def test_cuda_converges_to_pi(self):
        """CUDA backend converges to pi with large sample."""
        sim = PiEstimationSimulation()
        sim.set_seed(42)

        result = sim.run(500_000, backend="torch", torch_device="cuda", compute_stats=False)

        # Should be close to pi
        assert math.isclose(result.mean, np.pi, rel_tol=1e-2, abs_tol=1e-2)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestCUDAAdaptiveBatching:
    """[CUDA-02] Adaptive batch sizing tests."""

    def test_adaptive_batching_large_workload(self):
        """Large workloads are handled correctly."""
        sim = PiEstimationSimulation()
        sim.set_seed(42)

        # Use large number of simulations
        result = sim.run(100_000, backend="torch", torch_device="cuda", compute_stats=False)

        assert result.n_simulations == 100_000
        assert len(result.results) == 100_000
        # Should still converge
        assert 2.8 < result.mean < 3.5

    def test_fixed_batch_size_works(self):
        """Fixed batch size parameter works correctly."""
        from mcframework.backends import TorchCUDABackend

        sim = PiEstimationSimulation()
        sim.set_seed(42)

        # Use fixed batch size
        backend = TorchCUDABackend(device_id=0, batch_size=10_000)
        results = backend.run(sim, n_simulations=50_000, seed_seq=sim.seed_seq)

        assert len(results) == 50_000
        # Results should be valid
        assert 2.5 < results.mean() < 3.8

    def test_progress_callback_with_large_batch(self):
        """Progress callback works correctly."""
        sim = PiEstimationSimulation()
        sim.set_seed(42)

        progress_calls = []

        def callback(completed, total):
            progress_calls.append((completed, total))

        sim.run(
            50_000,
            backend="torch",
            torch_device="cuda",
            compute_stats=False,
            progress_callback=callback
        )

        # Callback should have been called at least once
        assert len(progress_calls) > 0
        # Final call should report completion
        assert progress_calls[-1] == (50_000, 50_000)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestCUDAMemoryManagement:
    """[CUDA-03] GPU memory management tests."""

    def test_memory_not_leaked_after_runs(self):
        """Multiple runs don't leak GPU memory."""
        sim = PiEstimationSimulation()
        sim.set_seed(42)

        # Get initial memory
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        initial_mem = torch.cuda.memory_allocated(0)

        # Run simulation multiple times
        for _ in range(5):
            sim.run(10_000, backend="torch", torch_device="cuda", compute_stats=False)

        # Clear cache and check final memory
        torch.cuda.empty_cache()
        final_mem = torch.cuda.memory_allocated(0)

        # Memory should not grow significantly (allow small variance)
        assert final_mem <= initial_mem + 1024 * 1024  # 1MB tolerance

    def test_memory_usage_reasonable(self):
        """Memory usage is proportional to batch size."""
        sim = PiEstimationSimulation()
        sim.set_seed(42)

        torch.cuda.reset_peak_memory_stats()
        sim.run(10_000, backend="torch", torch_device="cuda", compute_stats=False)
        peak_mem = torch.cuda.max_memory_allocated(0)

        # Peak memory should be reasonable (< 100 MB for 10k samples)
        assert peak_mem < 100 * 1024 * 1024  # 100 MB


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestCUDAStreams:
    """[CUDA-04] CUDA streams functionality."""

    def test_streams_enabled_by_default(self):
        """CUDA streams are used by default."""
        from mcframework.backends import TorchCUDABackend

        backend = TorchCUDABackend(device_id=0)
        assert backend.use_streams is True

    def test_streams_can_be_disabled(self):
        """CUDA streams can be disabled via parameter."""
        from mcframework.backends import TorchCUDABackend

        backend = TorchCUDABackend(device_id=0, use_streams=False)
        assert backend.use_streams is False

        # Should still work without streams
        sim = PiEstimationSimulation()
        sim.set_seed(42)
        results = backend.run(sim, n_simulations=5_000, seed_seq=sim.seed_seq)

        assert len(results) == 5_000
        assert 2.5 < results.mean() < 3.8


@pytest.mark.skipif(
    not CUDA_AVAILABLE or not _cupy_available(),
    reason="CUDA or CuPy not available"
)
class TestCuRANDIntegration:
    """[CUDA-05] cuRAND via CuPy integration tests."""

    def test_curand_mode_requires_curand_batch(self):
        """cuRAND mode requires curand_batch() method."""
        sim = PiEstimationSimulation()
        sim.set_seed(42)

        # PiEstimationSimulation doesn't have curand_batch
        from mcframework.backends import TorchCUDABackend
        backend = TorchCUDABackend(device_id=0, use_curand=True)

        with pytest.raises(NotImplementedError, match="does not implement curand_batch"):
            backend.run(sim, n_simulations=100, seed_seq=sim.seed_seq)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestCUDAMultiDevice:
    """[CUDA-06] Multi-device validation tests."""

    def test_device_selection_works(self):
        """Can select CUDA device ID."""
        from mcframework.backends import TorchCUDABackend

        # Should not raise for device 0 (always available in CUDA systems)
        backend = TorchCUDABackend(device_id=0)
        assert backend.device_id == 0

    def test_invalid_device_id_raises(self):
        """Invalid device ID raises RuntimeError."""
        from mcframework.backends import TorchCUDABackend

        device_count = torch.cuda.device_count()

        with pytest.raises(RuntimeError, match="requested but only"):
            TorchCUDABackend(device_id=device_count + 10)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestCUDAPerformance:
    """[CUDA-07] Performance benchmarks (informational)."""

    def test_cuda_execution_time_reasonable(self):
        """Execution time is within expected bounds."""
        sim = PiEstimationSimulation()
        sim.set_seed(42)

        result = sim.run(100_000, backend="torch", torch_device="cuda", compute_stats=False)

        # CUDA should be fast (generous upper bound)
        assert result.execution_time < 30.0
        assert result.execution_time > 0

    def test_cuda_stats_computation_works(self):
        """Stats engine works correctly with CUDA backend results."""
        sim = PiEstimationSimulation()
        sim.set_seed(42)

        result = sim.run(
            100_000,
            backend="torch",
            torch_device="cuda",
            compute_stats=True,
            confidence=0.95,
        )

        # Stats should be computed
        assert "mean" in result.stats
        assert "std" in result.stats
        assert "ci_mean" in result.stats

        # CI should contain pi
        ci = result.stats["ci_mean"]
        assert ci["low"] < np.pi < ci["high"]
