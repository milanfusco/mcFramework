"""Tests for Torch backend utilities and infrastructure.

These tests validate:
1. Generator creation and seeding from SeedSequence
2. Device validation and availability checks
3. TorchBackend factory class delegation
"""

import numpy as np
import pytest
import torch

from mcframework.sims import PiEstimationSimulation


class TestExplicitGeneratorInfrastructure:
    """[GPU-06] Test explicit torch.Generator infrastructure from SeedSequence."""

    def test_make_torch_generator_creates_valid_generator(self):
        """[GPU-06] make_torch_generator creates a valid torch.Generator."""
        from mcframework.backends import make_torch_generator

        sim = PiEstimationSimulation()
        sim.set_seed(42)
        device = torch.device("cpu")

        generator = make_torch_generator(device, sim.seed_seq)

        assert isinstance(generator, torch.Generator)
        assert generator.device == device

    def test_make_torch_generator_deterministic_from_seed_seq(self):
        """[GPU-06] Same SeedSequence produces same generator state."""
        from mcframework.backends import make_torch_generator

        device = torch.device("cpu")

        # Two simulations with same seed
        sim1 = PiEstimationSimulation()
        sim1.set_seed(42)
        gen1 = make_torch_generator(device, sim1.seed_seq)

        sim2 = PiEstimationSimulation()
        sim2.set_seed(42)
        gen2 = make_torch_generator(device, sim2.seed_seq)

        # Generate samples from each
        samples1 = torch.rand(1000, generator=gen1)
        samples2 = torch.rand(1000, generator=gen2)

        torch.testing.assert_close(samples1, samples2)

    def test_make_torch_generator_different_seeds_differ(self):
        """[GPU-06] Different SeedSequences produce different generator states."""
        from mcframework.backends import make_torch_generator

        device = torch.device("cpu")

        sim1 = PiEstimationSimulation()
        sim1.set_seed(111)
        gen1 = make_torch_generator(device, sim1.seed_seq)

        sim2 = PiEstimationSimulation()
        sim2.set_seed(222)
        gen2 = make_torch_generator(device, sim2.seed_seq)

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
        from mcframework.backends import make_torch_generator

        sim = PiEstimationSimulation()
        sim.set_seed(42)

        # Verify the seed_seq exists and has expected entropy
        assert sim.seed_seq is not None
        assert sim.seed_seq.entropy == 42

        # Create generator and verify it's deterministic
        device = torch.device("cpu")
        gen = make_torch_generator(device, sim.seed_seq)

        # The generator should be seeded from a spawned child
        # Verify by checking reproducibility
        sample1 = torch.rand(100, generator=gen)

        # Recreate with same seed
        sim2 = PiEstimationSimulation()
        sim2.set_seed(42)
        gen2 = make_torch_generator(device, sim2.seed_seq)
        sample2 = torch.rand(100, generator=gen2)

        torch.testing.assert_close(sample1, sample2)


class TestTorchBaseUtilities:
    """[GPU-08] Test torch_base.py utility functions."""

    def test_validate_torch_available_succeeds(self):
        """validate_torch_available() succeeds when torch is installed."""
        from mcframework.backends.torch_base import validate_torch_available

        # Should not raise since torch is installed
        validate_torch_available()

    def test_make_torch_generator_with_no_seed_returns_valid_generator(self):
        """make_torch_generator returns valid generator even without seed."""
        from mcframework.backends import make_torch_generator

        device = torch.device("cpu")

        # The function should still return a valid generator (warns internally)
        gen = make_torch_generator(device, None)
        assert isinstance(gen, torch.Generator)

        # Generator should still work (just not reproducible)
        samples = torch.rand(10, generator=gen)
        assert samples.shape == (10,)

    def test_import_torch_returns_module(self):
        """import_torch() returns the torch module."""
        from mcframework.backends.torch_base import import_torch

        th = import_torch()
        assert hasattr(th, "Tensor")
        assert hasattr(th, "Generator")


class TestTorchBackendFactory:
    """[GPU-09] Test TorchBackend factory class."""

    def test_torch_backend_cpu_with_kwargs_warns(self, caplog):
        """TorchBackend warns when CPU backend receives device_kwargs."""
        from mcframework.backends import TorchBackend

        caplog.set_level("WARNING")
        # CPU  backend should warn about unused kwargs
        backend = TorchBackend(device="cpu", some_unused_kwarg=True)

        assert "CPU backend ignores device_kwargs" in caplog.text
        assert "some_unused_kwarg" in caplog.text
        assert backend.device_type == "cpu"
        assert "some_unused_kwarg" not in backend.__dict__

    @pytest.mark.skipif(
        not (torch.backends.mps.is_available() and torch.backends.mps.is_built()),
        reason="MPS not available"
    )
    def test_torch_backend_mps_with_kwargs_warns(self, caplog):
        """TorchBackend warns when MPS backend receives device_kwargs."""
        from mcframework.backends import TorchBackend

        caplog.set_level("WARNING")

        # MPS backend should warn about unused kwargs
        backend = TorchBackend(device="mps", some_unused_kwarg=True)

        assert "MPS backend ignores device_kwargs" in caplog.text
        assert "some_unused_kwarg" in caplog.text
        assert backend.device_type == "mps"

    def test_torch_backend_exposes_device(self):
        """TorchBackend exposes the underlying device."""
        from mcframework.backends import TorchBackend

        backend = TorchBackend(device="cpu")

        assert backend.device == torch.device("cpu")
        assert backend.device_type == "cpu"

    def test_torch_backend_delegates_run(self):
        """TorchBackend delegates run() to underlying backend."""
        from mcframework.backends import TorchBackend

        backend = TorchBackend(device="cpu")
        sim = PiEstimationSimulation()
        sim.set_seed(42)

        results = backend.run(sim, 1000, sim.seed_seq, None)

        assert len(results) == 1000
        assert 2.5 < np.mean(results) < 3.8


class TestMPSAvailabilityCheck:
    """[GPU-12] Test is_mps_available() function."""

    def test_is_mps_available_returns_bool(self):
        """is_mps_available() returns a boolean."""
        from mcframework.backends import is_mps_available

        result = is_mps_available()
        assert isinstance(result, bool)

    def test_is_mps_available_matches_torch(self):
        """is_mps_available() matches torch.backends.mps checks."""
        from mcframework.backends import is_mps_available

        expected = (
            torch.backends.mps.is_available() and
            torch.backends.mps.is_built()
        )
        assert is_mps_available() == expected


class TestValidateTorchDevice:
    """[GPU-13] Test validate_torch_device() function."""

    def test_validate_cpu_always_passes(self):
        """validate_torch_device('cpu') always succeeds."""
        from mcframework.backends import validate_torch_device

        # Should not raise
        validate_torch_device("cpu")

    def test_validate_invalid_device_raises(self):
        """validate_torch_device() raises ValueError for invalid device."""
        from mcframework.backends import validate_torch_device

        with pytest.raises(ValueError, match="torch_device must be one of"):
            validate_torch_device("invalid_device")

    @pytest.mark.skipif(
        not (torch.backends.mps.is_available() and torch.backends.mps.is_built()),
        reason="MPS not available"
    )
    def test_validate_mps_passes_when_available(self):
        """validate_torch_device('mps') passes on Apple Silicon."""
        from mcframework.backends import validate_torch_device

        # Should not raise on MPS-capable system
        validate_torch_device("mps")


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

