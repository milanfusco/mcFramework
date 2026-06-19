
import numpy as np

from mcframework.backends.base import worker_run_chunk
from mcframework.backends.parallel import ThreadBackend


def test_thread_backend_invokes_progress_callback(simple_simulation):
    """[NFR-7] ThreadBackend reports progress per completed block and reaches the total."""
    simple_simulation.set_seed(42)
    seed_seq = np.random.SeedSequence(42)
    calls: list[tuple[int, int]] = []

    backend = ThreadBackend(n_workers=2)
    results = backend.run(
        simple_simulation,
        n_simulations=100,
        seed_seq=seed_seq,
        progress_callback=lambda done, total: calls.append((done, total)),
    )

    assert results.shape == (100,)
    assert calls, "progress_callback was never invoked"
    assert all(total == 100 for _, total in calls)
    # Progress is monotonically non-decreasing and ends at the total.
    assert [done for done, _ in calls] == sorted(done for done, _ in calls)
    assert calls[-1][0] == 100


class TestPerformance:
    """[NFR-1, NFR-7] Test performance characteristics."""

    #import time
    # def test_parallel_faster_than_sequential(self, simple_simulation):
    #     """Test parallel execution is faster for large runs"""
    #     n = 150_000

    #     # Sequential
    #     simple_simulation.set_seed(42)
    #     start = time.time()
    #     _ = simple_simulation.run(n, backend="sequential", compute_stats=False) # noqa: F841
    #     seq_time = time.time() - start

    #     # Parallel
    #     simple_simulation.set_seed(42)
    #     start = time.time()
    #     _ = simple_simulation.run(n, backend="auto", n_workers=3, compute_stats=False)
    #     par_time = time.time() - start

    #     # Parallel should be faster (allowing for overhead)
    #     # This is a soft check since timing can be variable
    #     # assert par_time < seq_time * 1.5  # Some speedup expected"""

    def test_auto_backend_resolves_per_platform(self, simple_simulation, monkeypatch):
        """[NFR-7] Auto backend should prefer processes on Windows, threads elsewhere."""
        simple_simulation.backend = "auto"

        monkeypatch.setattr("mcframework.simulation.is_windows_platform", lambda: True)
        assert simple_simulation._resolve_backend_type() == "process"

        monkeypatch.setattr("mcframework.simulation.is_windows_platform", lambda: False)
        assert simple_simulation._resolve_backend_type() == "thread"


    def test_memory_efficiency_streaming(self, simple_simulation):
        """[NFR-1] Test large simulations don't crash due to memory."""
        # This should work without OOM
        result = simple_simulation.run(
            50000,
            backend="auto",
            n_workers=2,
            compute_stats=False
        )
        assert result.n_simulations == 50000

    def test_worker_run_chunk(self, simple_simulation):
        """[FR-3] Test worker function directly."""
        seed_seq = np.random.SeedSequence(42)
        results = worker_run_chunk(simple_simulation, 100, seed_seq, {})
        assert len(results) == 100
        assert all(isinstance(r, float) for r in results)
