"""Verify that the base class stub and CUDA backend agree on method name."""

import pytest

from mcframework.simulation import MonteCarloSimulation


def test_base_class_has_curand_batch():
    """The base class must define curand_batch (not cupy_batch) to match CUDA dispatch."""
    assert hasattr(MonteCarloSimulation, "curand_batch"), (
        "MonteCarloSimulation must define curand_batch() so the CUDA backend "
        "can discover it. If you renamed it, update torch_cuda.py too."
    )


def test_curand_batch_raises_not_implemented():
    class Dummy(MonteCarloSimulation):
        def single_simulation(self, **kw):
            return 0.0

    sim = Dummy(name="dummy")
    with pytest.raises(NotImplementedError):
        sim.curand_batch(10, 0, None)


def test_rng_null_guard_raises():
    """_rng(None, None) must raise ValueError when no RNG is available."""
    class Dummy(MonteCarloSimulation):
        def single_simulation(self, **kw):
            return 0.0

    sim = Dummy(name="d")
    with pytest.raises(ValueError, match="No RNG available"):
        sim._rng(None, None)
