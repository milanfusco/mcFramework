import time

import numpy as np

from mcframework.core import _worker_run_chunk


class TestPerformance:
    """Test performance characteristics"""

    #import time
    # def test_parallel_faster_than_sequential(self, simple_simulation):
    #     """Test parallel execution is faster for large runs"""
    #     n = 150_000

    #     # Sequential
    #     simple_simulation.set_seed(42)
    #     start = time.time()
    #     _ = simple_simulation.run(n, parallel=False, compute_stats=False) # noqa: F841
    #     seq_time = time.time() - start

    #     # Parallel
    #     simple_simulation.set_seed(42)
    #     start = time.time()
    #     _ = simple_simulation.run(n, parallel=True, n_workers=3, compute_stats=False)
    #     par_time = time.time() - start

    #     # Parallel should be faster (allowing for overhead)
    #     # This is a soft check since timing can be variable
    #     # assert par_time < seq_time * 1.5  # Some speedup expected"""

    def test_auto_backend_resolves_per_platform(self, simple_simulation, monkeypatch):
        """Auto backend should prefer processes on Windows, threads elsewhere."""
        simple_simulation.parallel_backend = "auto"

        monkeypatch.setattr("mcframework.core._is_windows_platform", lambda: True)
        assert simple_simulation._resolve_parallel_backend() == "process"

        monkeypatch.setattr("mcframework.core._is_windows_platform", lambda: False)
        assert simple_simulation._resolve_parallel_backend() == "thread"


    def test_memory_efficiency_streaming(self, simple_simulation):
        """Test large simulations don't crash due to memory"""
        # This should work without OOM
        result = simple_simulation.run(
            50000,
            parallel=True,
            n_workers=2,
            compute_stats=False
        )
        assert result.n_simulations == 50000

    def test_worker_run_chunk(self, simple_simulation):
        """Test worker function directly"""
        seed_seq = np.random.SeedSequence(42)
        results = _worker_run_chunk(simple_simulation, 100, seed_seq, {})
        assert len(results) == 100
        assert all(isinstance(r, float) for r in results)
