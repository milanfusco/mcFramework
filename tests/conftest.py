import multiprocessing as mp

import numpy as np
import pytest

from mcframework.core import MonteCarloFramework, MonteCarloSimulation


class SimpleSim(MonteCarloSimulation):
    """A simple sim that *uses the simulation RNG* (not the global)."""
    def single_simulation(self, mean: float = 0.0, std: float = 1.0, _rng=None, **kwargs):
        rng = self._rng(_rng, self.rng)
        return float(rng.normal(mean, std))

class DeterministicSim(MonteCarloSimulation):
    """Deterministic simulation that returns incrementing integers."""
    def __init__(self):
        super().__init__("DeterministicSim")
        self.counter = 0
    def single_simulation(self, _rng=None, **kwargs):
        self.counter += 1
        return float(self.counter)


@pytest.fixture(autouse=True)
def _stable_seed():
    # Keep global state stable for any legacy code that still touches np.random.*
    np.random.seed(42)

@pytest.fixture(scope="session", autouse=True)
def _set_spawn_start_method():
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass  # already set

@pytest.fixture
def sample_data():
    """Fixture providing sample data for testing"""
    np.random.seed(42)
    return np.random.normal(5.0, 2.0, 1000)


@pytest.fixture
def simple_simulation():
    """Provide a simple simulation instance."""
    return SimpleSim(name="TestSim")


@pytest.fixture
def deterministic_simulation():
    """Provide a deterministic simulation instance."""
    return DeterministicSim()


@pytest.fixture
def framework():
    """Provide a framework with default state."""
    return MonteCarloFramework()


@pytest.fixture
def ctx_basic():
    """Basic context for stats engine tests"""
    return {
        "n": 1000,
        "confidence": 0.95,
        "nan_policy": "propagate",
        "ci_method": "auto",
        "percentiles": (5, 25, 50, 75, 95),
        "target": 0.0,
        "eps": 0.5,
    }

