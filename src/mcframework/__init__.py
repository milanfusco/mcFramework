"""mcframework package public API."""

# Profiling submodule (imported as submodule, not exposed at top level)
from . import profiling
from .core import MonteCarloFramework, MonteCarloSimulation, SimulationResult
from .sims import (
    BlackScholesPathSimulation,
    BlackScholesSimulation,
    PiEstimationSimulation,
    PortfolioSimulation,
)
from .stats_engine import DEFAULT_ENGINE, FnMetric, StatsContext, StatsEngine
from .utils import autocrit, t_crit, z_crit

__all__ = [
    "SimulationResult",
    "MonteCarloSimulation",
    "MonteCarloFramework",
    "PiEstimationSimulation",
    "PortfolioSimulation",
    "BlackScholesSimulation",
    "BlackScholesPathSimulation",
    "StatsEngine",
    "StatsContext",
    "FnMetric",
    "DEFAULT_ENGINE",
    "z_crit",
    "t_crit",
    "autocrit",
    "profiling",
]

__version__ = "0.2.0"
