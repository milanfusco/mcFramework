"""Tests for Torch MPS (Apple Metal Performance Shaders) backend.

These tests validate:
1. MPS backend returns valid results on Apple Silicon
2. MPS respects seeding for reproducibility (best-effort)
3. Stats computation works correctly with MPS results
4. Proper float32 -> float64 promotion for precision
"""

import math

import numpy as np
import pytest
import torch

from mcframework.core import MonteCarloSimulation
from mcframework.sims import PiEstimationSimulation

from mcframework.backends.torch_mps import is_mps_available, validate_mps_device

MPS_AVAILABLE = torch.backends.mps.is_available() and torch.backends.mps.is_built()


def test_is_mps_available_false_when_not_built(monkeypatch):
    """is_mps_available() returns False when MPS is available but not built."""
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: False)
    assert not is_mps_available()


def test_validate_mps_device_raises_when_not_built(monkeypatch):
    """validate_mps_device() raises RuntimeError when MPS is not built."""
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: False)
    with pytest.raises(RuntimeError, match="not built with MPS support"):
        validate_mps_device()


def test_is_mps_available_false_when_torch_missing(monkeypatch):
    """is_mps_available() returns False when torch cannot be imported."""
    from mcframework.backends import torch_mps

    def _no_torch():
        raise ImportError("no torch")

    monkeypatch.setattr(torch_mps, "import_torch", _no_torch)
    assert not is_mps_available()


def test_validate_mps_device_raises_when_not_available(monkeypatch):
    """validate_mps_device() raises RuntimeError when MPS is not available."""
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="MPS device requested but not available"):
        validate_mps_device()


@pytest.mark.skipif(not MPS_AVAILABLE, reason="MPS not available")
class TestTorchMPSBackendDirect:
    """[GPU-11] Direct tests for TorchMPSBackend class."""

    def test_mps_backend_direct_instantiation(self):
        """TorchMPSBackend can be instantiated directly."""
        from mcframework.backends import TorchMPSBackend

        backend = TorchMPSBackend()

        assert backend.device_type == "mps"
        assert backend.device == torch.device("mps")

    def test_mps_backend_direct_run(self):
        """TorchMPSBackend.run() works correctly."""
        from mcframework.backends import TorchMPSBackend

        backend = TorchMPSBackend()
        sim = PiEstimationSimulation()
        sim.set_seed(42)

        results = backend.run(sim, 5000, sim.seed_seq, None)

        assert len(results) == 5000
        # Results should be float64 after conversion
        assert results.dtype == np.float64

    def test_mps_backend_with_progress_callback(self):
        """TorchMPSBackend calls progress callback on completion."""
        from mcframework.backends import TorchMPSBackend

        backend = TorchMPSBackend()
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

    def test_mps_backend_rejects_unsupported_simulation(self):
        """TorchMPSBackend raises ValueError for unsupported simulation."""
        from mcframework.backends import TorchMPSBackend

        class UnsupportedSim(MonteCarloSimulation):
            supports_batch = False

            def single_simulation(self, _rng=None, **kwargs):
                return 1.0

        backend = TorchMPSBackend()
        sim = UnsupportedSim(name="Unsupported")
        sim.set_seed(42)

        with pytest.raises(ValueError, match="does not support Torch batch execution"):
            backend.run(sim, 100, sim.seed_seq, None)

    def test_mps_backend_rejects_non_positive_max_batch_size(self):
        """TorchMPSBackend rejects invalid max_batch_size values."""
        from mcframework.backends import TorchMPSBackend

        with pytest.raises(ValueError, match="max_batch_size must be a positive integer"):
            TorchMPSBackend(max_batch_size=0)
        with pytest.raises(ValueError, match="max_batch_size must be a positive integer"):
            TorchMPSBackend(max_batch_size=-1)

    def test_mps_backend_chunked_progress_callback_monotonic(self):
        """Chunked MPS runs report monotonic progress and final completion."""
        from mcframework.backends import TorchMPSBackend

        backend = TorchMPSBackend(max_batch_size=1_000)
        sim = PiEstimationSimulation()
        sim.set_seed(42)
        calls = []

        results = backend.run(sim, 3_500, sim.seed_seq, lambda c, t: calls.append((c, t)))

        assert len(results) == 3_500
        assert results.dtype == np.float64
        assert len(calls) == 4
        assert calls[-1] == (3_500, 3_500)
        assert all(calls[i][0] < calls[i + 1][0] for i in range(len(calls) - 1))


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

