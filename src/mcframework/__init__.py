"""mcframework package public API."""

from .core import (
    MonteCarloFramework,
    MonteCarloSimulation,
    SimulationResult,
)
from .sims import PiEstimationSimulation, PortfolioSimulation
from .stats_engine import (
    DEFAULT_ENGINE,
    FnMetric,
    StatsEngine,
    StatsContext
)
from .utils import autocrit, t_crit, z_crit

__all__ = [
    "SimulationResult",
    "MonteCarloSimulation",
    "MonteCarloFramework",
    "PiEstimationSimulation",
    "PortfolioSimulation",
    "StatsEngine",
    "StatsContext",
    "FnMetric",
    "DEFAULT_ENGINE",
    "z_crit",
    "t_crit",
    "autocrit",
]

__version__ = "0.1.0"
