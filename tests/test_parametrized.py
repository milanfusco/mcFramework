import pytest

from mcframework.core import make_blocks
from mcframework.stats_engine import autocrit, ci_mean


class TestParametrized:
    """Parametrized tests for various scenarios"""

    @pytest.mark.parametrize("n_sims", [1, 10, 100, 1000])
    def test_various_simulation_sizes(self, simple_simulation, n_sims):
        """Test different simulation sizes"""
        result = simple_simulation.run(n_sims, backend="sequential", compute_stats=False)
        assert result.n_simulations == n_sims
        assert len(result.results) == n_sims

    @pytest.mark.parametrize("confidence", [0.90, 0.95, 0.99])
    def test_various_confidence_levels(self, sample_data, confidence):
        """Test different confidence levels"""
        ctx = {"n": len(sample_data), "confidence": confidence, "ci_method": "z"}
        result = ci_mean(sample_data, ctx)
        assert result["confidence"] == confidence

    @pytest.mark.parametrize("method", ["z", "t", "auto"])
    def test_various_ci_methods(self, sample_data, method):
        """Test different CI calculation methods"""
        crit, kind = autocrit(0.95, len(sample_data), method)
        assert crit > 0
        assert kind in ["z", "t"]

    @pytest.mark.parametrize("n_workers", [1, 2, 4])
    def test_various_worker_counts(self, simple_simulation, n_workers):
        """Test different numbers of workers"""
        result = simple_simulation.run(
            200,
            backend="auto",
            n_workers=n_workers,
            compute_stats=False
        )
        assert result.n_simulations == 200

    @pytest.mark.parametrize("block_size", [100, 1000, 10000])
    def test_various_block_sizes(self, block_size):
        """Test different block sizes"""
        blocks = make_blocks(10000, block_size=block_size)
        total = sum(j - i for i, j in blocks)
        assert total == 10000
