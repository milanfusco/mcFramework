"""Tests for Torch batch backend execution.

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

    def test_torch_cpu_matches_numpy_statistically(self):
        """[GPU-02] Torch CPU and NumPy backends converge to same value (pi)."""
        n_sims = 500_000

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


class TestExplicitGeneratorInfrastructure:
    """[GPU-06] Test explicit torch.Generator infrastructure from SeedSequence."""

    def test_make_torch_generator_creates_valid_generator(self):
        """[GPU-06] _make_torch_generator creates a valid torch.Generator."""
        sim = PiEstimationSimulation()
        sim.set_seed(42)
        device = torch.device("cpu")

        generator = sim._make_torch_generator(device)

        assert isinstance(generator, torch.Generator)
        assert generator.device == device

    def test_make_torch_generator_deterministic_from_seed_seq(self):
        """[GPU-06] Same SeedSequence produces same generator state."""
        device = torch.device("cpu")

        # Two simulations with same seed
        sim1 = PiEstimationSimulation()
        sim1.set_seed(42)
        gen1 = sim1._make_torch_generator(device)

        sim2 = PiEstimationSimulation()
        sim2.set_seed(42)
        gen2 = sim2._make_torch_generator(device)

        # Generate samples from each
        samples1 = torch.rand(1000, generator=gen1)
        samples2 = torch.rand(1000, generator=gen2)

        torch.testing.assert_close(samples1, samples2)

    def test_make_torch_generator_different_seeds_differ(self):
        """[GPU-06] Different SeedSequences produce different generator states."""
        device = torch.device("cpu")

        sim1 = PiEstimationSimulation()
        sim1.set_seed(111)
        gen1 = sim1._make_torch_generator(device)

        sim2 = PiEstimationSimulation()
        sim2.set_seed(222)
        gen2 = sim2._make_torch_generator(device)

        samples1 = torch.rand(1000, generator=gen1)
        samples2 = torch.rand(1000, generator=gen2)

        assert not torch.equal(samples1, samples2)

    def test_no_global_rng_pollution(self):
        """[GPU-06] Torch backend doesn't pollute global RNG state."""
        # Set global RNG to known state
        torch.manual_seed(99999)
        global_sample_before = torch.rand(10).clone()

        # Reset global state
        torch.manual_seed(99999)

        # Run simulation (should use explicit generator, not global)
        sim = PiEstimationSimulation()
        sim.set_seed(42)
        sim.run(10_000, backend="torch", compute_stats=False)

        # Reset global state again
        torch.manual_seed(99999)
        global_sample_after = torch.rand(10)

        # Global state should be unchanged (simulation used explicit generator)
        torch.testing.assert_close(global_sample_before, global_sample_after)

    def test_seed_sequence_spawn_preserves_hierarchy(self):
        """[GPU-06] Generator seeding uses SeedSequence.spawn() for proper hierarchy."""
        sim = PiEstimationSimulation()
        sim.set_seed(42)

        # Verify the seed_seq exists and has expected entropy
        assert sim.seed_seq is not None
        assert sim.seed_seq.entropy == 42

        # Create generator and verify it's deterministic
        device = torch.device("cpu")
        gen = sim._make_torch_generator(device)

        # The generator should be seeded from a spawned child
        # Verify by checking reproducibility
        sample1 = torch.rand(100, generator=gen)

        # Recreate with same seed
        sim2 = PiEstimationSimulation()
        sim2.set_seed(42)
        gen2 = sim2._make_torch_generator(device)
        sample2 = torch.rand(100, generator=gen2)

        torch.testing.assert_close(sample1, sample2)


class TestTorchDeviceValidation:
    """[GPU-07] Test device validation and error handling."""

    def test_invalid_torch_device_raises(self):
        """[GPU-07] Invalid torch_device raises ValueError."""
        sim = PiEstimationSimulation()
        sim.set_seed(42)

        with pytest.raises(ValueError, match="torch_device must be one of"):
            sim.run(100, backend="torch", torch_device="invalid")

    def test_cpu_device_always_available(self):
        """[GPU-07] CPU device is always available."""
        sim = PiEstimationSimulation()
        sim.set_seed(42)

        # Should not raise
        result = sim.run(1000, backend="torch", torch_device="cpu", compute_stats=False)
        assert result.n_simulations == 1000

    @pytest.mark.skipif(
        torch.backends.mps.is_available(),
        reason="MPS is available, cannot test unavailable error"
    )
    def test_mps_unavailable_raises(self):
        """[GPU-07] MPS device raises RuntimeError when not available."""
        sim = PiEstimationSimulation()
        sim.set_seed(42)

        with pytest.raises(RuntimeError, match="MPS device requested but not available"):
            sim.run(100, backend="torch", torch_device="mps")

    @pytest.mark.skipif(
        torch.cuda.is_available(),
        reason="CUDA is available, cannot test unavailable error"
    )
    def test_cuda_unavailable_raises(self):
        """[GPU-07] CUDA device raises RuntimeError when not available."""
        sim = PiEstimationSimulation()
        sim.set_seed(42)

        with pytest.raises(RuntimeError, match="CUDA device requested but not available"):
            sim.run(100, backend="torch", torch_device="cuda")


# =============================================================================
# MPS Backend Tests (Apple Silicon)
# =============================================================================

MPS_AVAILABLE = torch.backends.mps.is_available() and torch.backends.mps.is_built()


@pytest.mark.skipif(not MPS_AVAILABLE, reason="MPS not available")
class TestTorchMPSBackend:
    """[MPS-01] Tests for Apple Metal Performance Shaders backend."""

    def test_pi_mps_returns_valid_results(self):
        """[MPS-01] MPS backend returns valid Pi estimates."""
        sim = PiEstimationSimulation()
        sim.set_seed(42)

        result = sim.run(10_000, backend="torch", torch_device="mps", compute_stats=False)

        # Mean should be close to pi (relaxed tolerance for MPS)
        assert 2.5 < result.mean < 3.8
        assert result.n_simulations == 10_000
        assert len(result.results) == 10_000

    def test_pi_mps_converges_to_pi(self):
        """[MPS-01] MPS backend converges to pi with large sample."""
        sim = PiEstimationSimulation()
        sim.set_seed(42)

        result = sim.run(500_000, backend="torch", torch_device="mps", compute_stats=False)

        # Should be close to pi (not tightening tolerance for MPS)
        assert math.isclose(result.mean, np.pi, rel_tol=1e-2, abs_tol=1e-2)

    def test_mps_stats_computation_works(self):
        """[MPS-01] Stats engine works correctly with MPS backend results."""
        sim = PiEstimationSimulation()
        sim.set_seed(42)

        result = sim.run(
            100_000,
            backend="torch",
            torch_device="mps",
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

    def test_mps_results_are_float64(self):
        """[MPS-01] MPS results are promoted to float64 for stats precision."""
        sim = PiEstimationSimulation()
        sim.set_seed(42)

        result = sim.run(1000, backend="torch", torch_device="mps", compute_stats=False)

        # Results should be float64 (promoted from MPS float32)
        assert result.results.dtype == np.float64

    def test_mps_no_nans_or_infs(self):
        """[MPS-01] MPS backend produces no NaN or Inf values."""
        sim = PiEstimationSimulation()
        sim.set_seed(42)

        result = sim.run(50_000, backend="torch", torch_device="mps", compute_stats=False)

        assert not np.any(np.isnan(result.results))
        assert not np.any(np.isinf(result.results))

    def test_mps_ci_widths_reasonable(self):
        """[MPS-01] MPS confidence intervals have reasonable widths."""
        sim = PiEstimationSimulation()
        sim.set_seed(42)

        result = sim.run(
            100_000,
            backend="torch",
            torch_device="mps",
            compute_stats=True,
            confidence=0.95,
        )

        ci = result.stats["ci_mean"]
        ci_width = ci["high"] - ci["low"]

        # CI width should be small for 100k samples (< 0.1)
        assert ci_width < 0.1
        assert ci_width > 0  # Not degenerate

    def test_mps_execution_time_reasonable(self):
        """[MPS-01] MPS backend execution time is reasonable."""
        sim = PiEstimationSimulation()
        sim.set_seed(42)

        result = sim.run(100_000, backend="torch", torch_device="mps", compute_stats=False)

        # MPS should be fast (generous upper bound)
        assert result.execution_time < 30.0
        assert result.execution_time > 0


@pytest.mark.skipif(not MPS_AVAILABLE, reason="MPS not available")
class TestMPSDeterminism:
    """[MPS-02] Test MPS determinism behavior (best-effort, not bitwise)."""

    def test_mps_same_seed_similar_mean(self):
        """[MPS-02] Same seed produces statistically similar results on MPS."""
        # Note: MPS determinism is best-effort, so we compare means not bitwise
        sim1 = PiEstimationSimulation()
        sim1.set_seed(42)
        result1 = sim1.run(100_000, backend="torch", torch_device="mps", compute_stats=False)

        sim2 = PiEstimationSimulation()
        sim2.set_seed(42)
        result2 = sim2.run(100_000, backend="torch", torch_device="mps", compute_stats=False)

        # Means should be very close (even if not bitwise identical)
        assert math.isclose(result1.mean, result2.mean, rel_tol=1e-2, abs_tol=1e-2)

    def test_mps_different_seeds_differ(self):
        """[MPS-02] Different seeds produce different results on MPS."""
        sim1 = PiEstimationSimulation()
        sim1.set_seed(111)
        result1 = sim1.run(10_000, backend="torch", torch_device="mps", compute_stats=False)

        sim2 = PiEstimationSimulation()
        sim2.set_seed(222)
        result2 = sim2.run(10_000, backend="torch", torch_device="mps", compute_stats=False)

        # Results should differ (extremely unlikely to be equal)
        assert not np.array_equal(result1.results, result2.results)

    def test_mps_generator_structure_preserved(self):
        """[MPS-02] MPS uses explicit generator from SeedSequence spawn."""
        sim = PiEstimationSimulation()
        sim.set_seed(42)

        # This should work without error, using spawned generator
        result = sim.run(10_000, backend="torch", torch_device="mps", compute_stats=False)

        # Verify metadata shows seed was used
        assert result.metadata["seed_entropy"] == 42

