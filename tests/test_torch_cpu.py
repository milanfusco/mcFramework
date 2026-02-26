"""Tests for Torch CPU backend execution.

These tests validate:
1. Torch CPU results match NumPy results within tolerance
2. Torch backend respects reproducibility via seeding
3. Proper error handling for unsupported simulations
4. Statistics computation works correctly with Torch results
"""

import math

import numpy as np
import pytest
import torch

from mcframework.core import MonteCarloSimulation
from mcframework.sims import PiEstimationSimulation


class TestTorchBackendBasics:
    """[GPU-01] Basic Torch backend functionality tests."""

    def test_pi_torch_cpu_returns_valid_results(self):
        """[GPU-01] Torch CPU backend returns valid Pi estimates."""
        sim = PiEstimationSimulation()
        sim.set_seed(42)

        result = sim.run(10_000, backend="torch", compute_stats=False)

        # Mean should be close to pi
        assert 2.5 < result.mean < 3.8
        assert result.n_simulations == 10_000
        assert len(result.results) == 10_000

    def test_torch_cpu_deterministic_with_seed(self):
        """[GPU-01] Torch CPU backend produces deterministic results with same seed."""
        sim1 = PiEstimationSimulation()
        sim1.set_seed(12345)
        result1 = sim1.run(5_000, backend="torch", compute_stats=False)

        sim2 = PiEstimationSimulation()
        sim2.set_seed(12345)
        result2 = sim2.run(5_000, backend="torch", compute_stats=False)

        np.testing.assert_array_equal(result1.results, result2.results)

    def test_torch_cpu_different_seeds_differ(self):
        """[GPU-01] Torch CPU backend produces different results with different seeds."""
        sim1 = PiEstimationSimulation()
        sim1.set_seed(111)
        result1 = sim1.run(1_000, backend="torch", compute_stats=False)

        sim2 = PiEstimationSimulation()
        sim2.set_seed(222)
        result2 = sim2.run(1_000, backend="torch", compute_stats=False)

        # Results should differ (extremely unlikely to be equal)
        assert not np.array_equal(result1.results, result2.results)


class TestTorchNumPyParity:
    """[GPU-02] Test that Torch CPU matches NumPy results statistically."""

    @pytest.mark.slow
    def test_torch_cpu_matches_numpy_statistically(self):
        """[GPU-02] Torch CPU and NumPy backends converge to same value (pi)."""
        n_sims = 100_000

        # NumPy path (sequential backend)
        sim_np = PiEstimationSimulation()
        sim_np.set_seed(42)
        # Force scalar path by disabling batch support temporarily
        sim_np.supports_batch = False
        result_np = sim_np.run(n_sims, backend="sequential", compute_stats=False)

        # Torch CPU path
        sim_torch = PiEstimationSimulation()
        sim_torch.set_seed(42)
        result_torch = sim_torch.run(n_sims, backend="torch", compute_stats=False)

        # Both should be close to pi
        assert math.isclose(result_np.mean, np.pi, rel_tol=1e-2, abs_tol=1e-2)
        assert math.isclose(result_torch.mean, np.pi, rel_tol=1e-2, abs_tol=1e-2)

        # And close to each other
        assert math.isclose(result_np.mean, result_torch.mean, rel_tol=1e-2, abs_tol=1e-2)

    def test_torch_stats_computation_works(self):
        """[GPU-02] Stats engine works correctly with Torch backend results."""
        sim = PiEstimationSimulation()
        sim.set_seed(42)

        result = sim.run(
            100_000,
            backend="torch",
            compute_stats=True,
            confidence=0.95,
        )

        # Stats should be computed
        assert "mean" in result.stats
        assert "std" in result.stats
        assert "ci_mean" in result.stats

        # CI should contain pi (ci_mean is a dict with 'low' and 'high' keys)
        ci = result.stats["ci_mean"]
        assert ci["low"] < np.pi < ci["high"]

    def test_torch_percentiles_work(self):
        """[GPU-02] Percentile computation works with Torch backend."""
        sim = PiEstimationSimulation()
        sim.set_seed(42)

        result = sim.run(
            50_000,
            backend="torch",
            compute_stats=True,
            percentiles=[5, 25, 50, 75, 95],
        )

        # Percentiles should be computed
        assert 5 in result.percentiles
        assert 50 in result.percentiles
        assert 95 in result.percentiles

        # For Pi estimation, values are binary (0 or 4), so median is 4.0
        # (more than ~78.5% of values are inside the circle)
        # Just verify percentiles are valid values from the distribution
        assert result.percentiles[50] in (0.0, 4.0)
        assert result.percentiles[5] in (0.0, 4.0)
        assert result.percentiles[95] in (0.0, 4.0)


class TestTorchBackendErrors:
    """[GPU-03] Test error handling for Torch backend."""

    def test_torch_backend_rejects_unsupported_simulation(self):
        """[GPU-03] Torch backend raises error for simulations without supports_batch."""

        class NoTorchSim(MonteCarloSimulation):
            supports_batch = False

            def single_simulation(self, _rng=None, **kwargs):
                rng = self._rng(_rng, self.rng)
                return float(rng.random())

        sim = NoTorchSim(name="NoTorchSim")
        sim.set_seed(42)

        with pytest.raises(ValueError, match="does not support Torch batch"):
            sim.run(100, backend="torch")

    def test_torch_batch_not_implemented_raises(self):
        """[GPU-03] Calling torch_batch on base class raises NotImplementedError."""

        class PartialTorchSim(MonteCarloSimulation):
            supports_batch = True  # Claims support but doesn't implement

            def single_simulation(self, _rng=None, **kwargs):
                return 1.0

        sim = PartialTorchSim(name="PartialTorchSim")
        sim.set_seed(42)

        with pytest.raises(NotImplementedError):
            sim.run(100, backend="torch")


class TestTorchBatchMethod:
    """[GPU-04] Test torch_batch method directly."""

    def test_torch_batch_returns_correct_shape(self):
        """[GPU-04] torch_batch returns tensor of correct shape."""
        sim = PiEstimationSimulation()
        device = torch.device("cpu")
        generator = torch.Generator(device=device)
        generator.manual_seed(42)

        result = sim.torch_batch(1000, device=device, generator=generator)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (1000,)
        # torch_batch returns float32 for MPS compatibility
        # Framework promotes to float64 after moving to CPU
        assert result.dtype == torch.float32

    def test_torch_batch_values_are_valid(self):
        """[GPU-04] torch_batch returns valid Pi indicator values."""
        sim = PiEstimationSimulation()
        device = torch.device("cpu")
        generator = torch.Generator(device=device)
        generator.manual_seed(42)

        result = sim.torch_batch(10_000, device=device, generator=generator)

        # Values should be 0.0 or 4.0 (indicator * 4)
        unique_vals = torch.unique(result)
        assert len(unique_vals) == 2
        assert 0.0 in unique_vals.tolist()
        assert 4.0 in unique_vals.tolist()

        # Mean should be close to pi
        assert math.isclose(result.mean().item(), np.pi, rel_tol=0.1, abs_tol=0.1)

    def test_torch_batch_explicit_generator_determinism(self):
        """[GPU-04] Same generator seed produces identical results."""
        sim = PiEstimationSimulation()
        device = torch.device("cpu")

        # First run
        gen1 = torch.Generator(device=device)
        gen1.manual_seed(12345)
        result1 = sim.torch_batch(5_000, device=device, generator=gen1)

        # Second run with same seed
        gen2 = torch.Generator(device=device)
        gen2.manual_seed(12345)
        result2 = sim.torch_batch(5_000, device=device, generator=gen2)

        torch.testing.assert_close(result1, result2)

    def test_torch_batch_different_generators_differ(self):
        """[GPU-04] Different generator seeds produce different results."""
        sim = PiEstimationSimulation()
        device = torch.device("cpu")

        gen1 = torch.Generator(device=device)
        gen1.manual_seed(111)
        result1 = sim.torch_batch(1_000, device=device, generator=gen1)

        gen2 = torch.Generator(device=device)
        gen2.manual_seed(222)
        result2 = sim.torch_batch(1_000, device=device, generator=gen2)

        assert not torch.equal(result1, result2)


class TestTorchBackendIntegration:
    """[GPU-05] Integration tests for Torch backend with framework features."""

    def test_torch_backend_with_framework_metadata(self):
        """[GPU-05] Torch backend populates result metadata correctly."""
        sim = PiEstimationSimulation()
        sim.set_seed(42)

        result = sim.run(10_000, backend="torch", compute_stats=True)

        assert result.metadata["simulation_name"] == "Pi Estimation"
        assert result.metadata["n"] == 10_000
        assert result.metadata["seed_entropy"] is not None

    def test_torch_backend_execution_time_recorded(self):
        """[GPU-05] Torch backend records execution time."""
        sim = PiEstimationSimulation()
        sim.set_seed(42)

        result = sim.run(50_000, backend="torch", compute_stats=False)

        assert result.execution_time > 0
        # Torch batch should be fast
        assert result.execution_time < 10.0  # Generous upper bound


class TestTorchCPUBackendDirect:
    """[GPU-10] Direct tests for TorchCPUBackend class."""

    def test_cpu_backend_direct_instantiation(self):
        """TorchCPUBackend can be instantiated directly."""
        from mcframework.backends import TorchCPUBackend

        backend = TorchCPUBackend()

        assert backend.device_type == "cpu"
        assert backend.device == torch.device("cpu")

    def test_cpu_backend_direct_run(self):
        """TorchCPUBackend.run() works correctly."""
        from mcframework.backends import TorchCPUBackend

        backend = TorchCPUBackend()
        sim = PiEstimationSimulation()
        sim.set_seed(42)

        results = backend.run(sim, 5000, sim.seed_seq, None)

        assert len(results) == 5000
        assert results.dtype == np.float64

    def test_cpu_backend_with_progress_callback(self):
        """TorchCPUBackend calls progress callback on completion."""
        from mcframework.backends import TorchCPUBackend

        backend = TorchCPUBackend()
        sim = PiEstimationSimulation()
        sim.set_seed(42)

        callback_calls = []

        def progress_callback(completed, total):
            callback_calls.append((completed, total))

        results = backend.run(sim, 1000, sim.seed_seq, progress_callback)

        # Callback should be called at completion
        assert len(callback_calls) == 1
        assert callback_calls[0] == (1000, 1000)
        # Verify results are still valid
        assert len(results) == 1000

    def test_cpu_backend_rejects_unsupported_simulation(self):
        """TorchCPUBackend raises ValueError for unsupported simulation."""
        from mcframework.backends import TorchCPUBackend

        class UnsupportedSim(MonteCarloSimulation):
            supports_batch = False

            def single_simulation(self, _rng=None, **kwargs):
                return 1.0

        backend = TorchCPUBackend()
        sim = UnsupportedSim(name="Unsupported")
        sim.set_seed(42)

        with pytest.raises(ValueError, match="does not support Torch batch execution"):
            backend.run(sim, 100, sim.seed_seq, None)

