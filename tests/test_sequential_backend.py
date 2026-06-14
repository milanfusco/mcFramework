"""Direct tests for mcframework.backends.sequential.SequentialBackend."""

from __future__ import annotations

import numpy as np

from mcframework.backends.sequential import SequentialBackend
from mcframework.core import MonteCarloSimulation


class ConstantSim(MonteCarloSimulation):
    """Returns a fixed value regardless of RNG."""

    def single_simulation(self, _rng=None, **kwargs):
        return 1.0


class RngSim(MonteCarloSimulation):
    """Returns a normal draw from the provided RNG."""

    def single_simulation(self, _rng=None, **kwargs):
        rng = self._rng(_rng, self.rng)
        return float(rng.normal())


class TestSequentialBackend:
    def test_basic_run_no_seed(self):
        backend = SequentialBackend()
        sim = ConstantSim(name="const")
        results = backend.run(sim, 10, seed_seq=None, progress_callback=None)
        assert results.shape == (10,)
        np.testing.assert_array_equal(results, 1.0)

    def test_deterministic_with_seed(self):
        backend = SequentialBackend()
        sim = RngSim(name="rng")

        seq1 = np.random.SeedSequence(99)
        seq2 = np.random.SeedSequence(99)

        r1 = backend.run(sim, 50, seed_seq=seq1, progress_callback=None)
        r2 = backend.run(sim, 50, seed_seq=seq2, progress_callback=None)
        np.testing.assert_array_equal(r1, r2)

    def test_different_seeds_differ(self):
        backend = SequentialBackend()
        sim = RngSim(name="rng")

        r1 = backend.run(sim, 100, seed_seq=np.random.SeedSequence(1), progress_callback=None)
        r2 = backend.run(sim, 100, seed_seq=np.random.SeedSequence(2), progress_callback=None)
        assert not np.array_equal(r1, r2)

    def test_progress_callback(self):
        backend = SequentialBackend()
        sim = ConstantSim(name="const")
        calls = []
        backend.run(sim, 200, seed_seq=None, progress_callback=lambda c, t: calls.append((c, t)))
        assert len(calls) > 0
        assert calls[-1] == (200, 200)

    def test_kwargs_forwarded(self):
        class KwargSim(MonteCarloSimulation):
            def single_simulation(self, value=0.0, _rng=None, **kwargs):
                return float(value)

        backend = SequentialBackend()
        sim = KwargSim(name="kw")
        results = backend.run(sim, 5, seed_seq=None, progress_callback=None, value=42.0)
        np.testing.assert_array_equal(results, 42.0)

    def test_progress_callback_with_seed(self):
        """Progress callback fires in the seed_seq-is-not-None branch."""
        backend = SequentialBackend()
        sim = ConstantSim(name="const")
        calls = []
        seed = np.random.SeedSequence(42)
        backend.run(
            sim, 200,
            seed_seq=seed,
            progress_callback=lambda c, t: calls.append((c, t)),
        )
        assert len(calls) > 0
        assert calls[-1] == (200, 200)
