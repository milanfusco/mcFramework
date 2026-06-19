"""mcframework package public API."""

import logging
from importlib.metadata import PackageNotFoundError, version

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
from .validation import ConvergenceReport, validate_convergence

# Library logging best practice: attach a no-op handler to the package root so
# importing mcframework never emits "No handlers could be found" warnings and
# never imposes logging configuration on the host application.
logging.getLogger(__name__).addHandler(logging.NullHandler())

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
    "ConvergenceReport",
    "validate_convergence",
    "z_crit",
    "t_crit",
    "autocrit",
    "profiling",
]

try:
    __version__ = version("mcframework")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
