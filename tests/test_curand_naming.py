"""Verify that the base class stub and CUDA backend agree on method name."""

import warnings

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
    try:
        sim.curand_batch(10, 0, None)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("curand_batch should raise NotImplementedError by default")


def test_curand_batch_delegates_to_legacy_cupy_batch():
    class LegacyDummy(MonteCarloSimulation):
        def single_simulation(self, **kw):
            return 0.0

        def cupy_batch(self, n, *, device, rng):
            return ("legacy", n, device, rng)

    sim = LegacyDummy(name="legacy")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = sim.curand_batch(10, 2, None)
    assert out[0] == "legacy"
    assert out[1] == 10
    assert any("cupy_batch() is deprecated" in str(w.message) for w in caught)


def test_legacy_cupy_batch_shim_delegates_to_curand_batch():
    class NewDummy(MonteCarloSimulation):
        def single_simulation(self, **kw):
            return 0.0

        def curand_batch(self, n, device_id, rng):
            return ("new", n, device_id, rng)

    sim = NewDummy(name="new")
    with pytest.deprecated_call(match="cupy_batch\\(\\) is deprecated"):
        out = sim.cupy_batch(7, device="cuda:3", rng=None)
    assert out == ("new", 7, 3, None)


def test_rng_null_guard_raises():
    """_rng(None, None) must raise ValueError when no RNG is available."""
    class Dummy(MonteCarloSimulation):
        def single_simulation(self, **kw):
            return 0.0

    sim = Dummy(name="d")
    with pytest.raises(ValueError, match="No RNG available"):
        sim._rng(None, None)


def test_curand_batch_legacy_fallback_without_torch(monkeypatch):
    """When torch is not importable, curand_batch passes device_id as-is to cupy_batch."""
    import builtins

    class LegacyNoTorch(MonteCarloSimulation):
        def single_simulation(self, **kw):
            return 0.0

        def cupy_batch(self, n, *, device, rng):
            return ("legacy", n, device, rng)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)

    sim = LegacyNoTorch(name="lt")
    monkeypatch.setattr(builtins, "__import__", fake_import)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        out = sim.curand_batch(5, 1, None)
    assert out == ("legacy", 5, 1, None)


def test_cupy_batch_shim_device_int():
    """cupy_batch shim handles device passed as int."""
    class NewSim(MonteCarloSimulation):
        def single_simulation(self, **kw):
            return 0.0

        def curand_batch(self, n, device_id, rng):
            return ("new", n, device_id, rng)

    sim = NewSim(name="ns")
    with pytest.deprecated_call(match="cupy_batch\\(\\) is deprecated"):
        out = sim.cupy_batch(3, device=0, rng=None)
    assert out == ("new", 3, 0, None)


def test_cupy_batch_shim_device_with_index_attr():
    """cupy_batch shim handles device object with .index attribute."""
    class FakeDevice:
        def __init__(self, idx):
            self.index = idx

    class NewSim(MonteCarloSimulation):
        def single_simulation(self, **kw):
            return 0.0

        def curand_batch(self, n, device_id, rng):
            return ("new", n, device_id, rng)

    sim = NewSim(name="ns")
    with pytest.deprecated_call(match="cupy_batch\\(\\) is deprecated"):
        out = sim.cupy_batch(4, device=FakeDevice(2), rng=None)
    assert out == ("new", 4, 2, None)


def test_cupy_batch_shim_device_plain_string():
    """cupy_batch shim falls back to device_id=0 for strings without a colon."""
    class NewSim(MonteCarloSimulation):
        def single_simulation(self, **kw):
            return 0.0

        def curand_batch(self, n, device_id, rng):
            return ("new", n, device_id, rng)

    sim = NewSim(name="ns")
    with pytest.deprecated_call(match="cupy_batch\\(\\) is deprecated"):
        out = sim.cupy_batch(2, device="mps", rng=None)
    assert out == ("new", 2, 0, None)


def test_cupy_batch_shim_device_with_none_index():
    """cupy_batch shim defaults to device_id=0 when device.index is None."""
    class DeviceWithNoneIndex:
        index = None

    class NewSim(MonteCarloSimulation):
        def single_simulation(self, **kw):
            return 0.0

        def curand_batch(self, n, device_id, rng):
            return ("new", n, device_id, rng)

    sim = NewSim(name="ns")
    with pytest.deprecated_call(match="cupy_batch\\(\\) is deprecated"):
        out = sim.cupy_batch(1, device=DeviceWithNoneIndex(), rng=None)
    assert out == ("new", 1, 0, None)
