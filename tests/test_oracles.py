"""
Regression tests pinning the built-in simulations to their analytic references.

Each oracle-backed simulation must converge to its known closed-form answer. These
are the guardrails that would have caught, e.g., the American LSM look-ahead bug:
a wrong estimator lands many standard errors away from the oracle and fails.
"""

from __future__ import annotations

import math

import pytest

from mcframework import (
    BlackScholesPathSimulation,
    BlackScholesSimulation,
    PiEstimationSimulation,
    PortfolioSimulation,
    validate_convergence,
)

# (id, sim factory, params, n) for sims that declare an analytic_reference.
ORACLE_CASES = [
    ("pi", PiEstimationSimulation, {}, 20_000),
    ("portfolio_gbm", PortfolioSimulation, {}, 50_000),
    ("bs_path", BlackScholesPathSimulation, {}, 50_000),
    ("bs_european_call", BlackScholesSimulation, {"option_type": "call"}, 50_000),
    ("bs_european_put", BlackScholesSimulation, {"option_type": "put"}, 50_000),
]


@pytest.mark.parametrize(
    "factory,params,n",
    [c[1:] for c in ORACLE_CASES],
    ids=[c[0] for c in ORACLE_CASES],
)
def test_builtin_sims_converge_to_oracle(factory, params, n):
    report = validate_convergence(factory(), n, seed=0, **params)
    assert report.status == "pass", (
        f"{factory.__name__} estimate {report.estimate} drifted from oracle "
        f"{report.oracle} by {report.abs_error} (> {report.sigma_tol} * SE={report.se})"
    )
    assert report.within_tol
    assert report.reference_source  # every oracle-backed sim cites its source
    assert report.reference_kind == "closed-form"  # built-ins are all closed-form


def test_pi_oracle_value():
    assert PiEstimationSimulation().analytic_reference() == math.pi


def test_bs_european_matches_textbook_price():
    # Hull-style benchmark: S0=K=100, r=5%, sigma=20%, T=1 -> call ≈ 10.4506.
    sim = BlackScholesSimulation()
    price = sim.analytic_reference(S0=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type="call")
    assert price == pytest.approx(10.4506, abs=1e-3)


def test_american_has_no_oracle():
    sim = BlackScholesSimulation()
    assert sim.analytic_reference(exercise_type="american") is None
    report = validate_convergence(sim, 1_000, exercise_type="american")
    assert report.status == "no-oracle"


def test_portfolio_arithmetic_branch_has_no_oracle():
    sim = PortfolioSimulation()
    assert sim.analytic_reference(use_gbm=False) is None
    report = validate_convergence(sim, 1_000, use_gbm=False)
    assert report.status == "no-oracle"
