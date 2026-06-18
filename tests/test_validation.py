"""
Tests for the oracle/convergence validation harness (``mcframework.validation``).

These exercise the reusable engine with tiny inline simulations whose true mean is
known in closed form, so the behavior is exact and seed-deterministic.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.random import Generator

from mcframework import ConvergenceReport, validate_convergence
from mcframework.core import MonteCarloSimulation


class UniformMeanSim(MonteCarloSimulation):
    """Each draw is Uniform(0, 1); E[draw] = 0.5 is the oracle."""

    reference_source = "Closed form: E[Uniform(0,1)] = 1/2"

    def __init__(self):
        super().__init__("Uniform Mean")

    def single_simulation(self, _rng: Generator | None = None, **kwargs) -> float:
        rng = self._rng(_rng, self.rng)
        return float(rng.uniform(0.0, 1.0))

    def analytic_reference(self, **params) -> float:
        return 0.5


class NoOracleSim(MonteCarloSimulation):
    """A simulation that declares no analytic reference."""

    def __init__(self):
        super().__init__("No Oracle")

    def single_simulation(self, _rng: Generator | None = None, **kwargs) -> float:
        rng = self._rng(_rng, self.rng)
        return float(rng.uniform(0.0, 1.0))


class WrongOracleSim(UniformMeanSim):
    """Same Uniform(0,1) draws but a deliberately wrong oracle far from 0.5."""

    def analytic_reference(self, **params) -> float:
        return 5.0


def test_pass_within_tolerance():
    report = validate_convergence(UniformMeanSim(), 20_000, seed=0)
    assert isinstance(report, ConvergenceReport)
    assert report.status == "pass"
    assert report.within_tol
    assert report.oracle == 0.5
    assert report.abs_error is not None and report.abs_error < 0.05
    assert report.reference_source.startswith("Closed form")
    # The 99% CI should bracket the true mean for a correct estimator.
    assert report.within_ci


def test_no_oracle_path():
    report = validate_convergence(NoOracleSim(), 1_000)
    assert report.status == "no-oracle"
    assert report.oracle is None
    assert report.estimate is None
    assert not report.within_tol


def test_wrong_oracle_fails():
    report = validate_convergence(WrongOracleSim(), 20_000, seed=0)
    assert report.status == "fail"
    assert not report.within_tol
    assert not report.within_ci
    # A wrong oracle should be many standard errors away.
    assert report.abs_error is not None and report.se is not None
    assert report.abs_error > report.sigma_tol * report.se


def test_reproducible_at_fixed_seed():
    a = validate_convergence(UniformMeanSim(), 5_000, seed=123)
    b = validate_convergence(UniformMeanSim(), 5_000, seed=123)
    assert a.estimate == b.estimate
    assert a.se == b.se


def test_rel_error_and_se_are_consistent():
    report = validate_convergence(UniformMeanSim(), 10_000, seed=7)
    assert report.rel_error == pytest.approx(report.abs_error / abs(report.oracle))
    # se = std / sqrt(n); std for Uniform(0,1) is 1/sqrt(12) ≈ 0.289, so se ≈ 0.00289.
    assert report.se == pytest.approx(1.0 / np.sqrt(12) / np.sqrt(10_000), rel=0.1)
