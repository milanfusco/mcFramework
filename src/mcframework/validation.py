r"""
Convergence validation against analytic references ("oracles").

A Monte Carlo simulation is only as trustworthy as your ability to check it. This
module turns that check into a first-class, repeatable operation: given a simulation
that declares a known answer via
:meth:`~mcframework.core.MonteCarloSimulation.analytic_reference`, it runs the
simulation and asserts that the estimate converges to that answer.

The implementation is deliberately thin: it reuses the stats engine's existing
``target`` machinery (:func:`~mcframework.stats_engine.bias_to_target`,
:func:`~mcframework.stats_engine.mse_to_target`) and the confidence interval from
:func:`~mcframework.stats_engine.ci_mean`. The oracle is simply passed through as the
context ``target``.

Two complementary checks are reported:

- ``within_ci``  -- the oracle lies inside the computed confidence interval.
- ``within_tol`` -- the absolute error is within ``sigma_tol`` standard errors.

``within_tol`` at a fixed seed is the recommended CI gate: it is statistically
principled, essentially never flaky, yet a genuine implementation bug lands many
standard errors away and fails loudly.

Example
-------
>>> import math
>>> from mcframework.validation import validate_convergence
>>> # report = validate_convergence(PiSim(), 200_000)  # doctest: +SKIP
>>> # report.status, round(report.estimate, 2)  # doctest: +SKIP
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .simulation import MonteCarloSimulation

__all__ = ["ConvergenceReport", "validate_convergence"]


@dataclass(frozen=True)
class ConvergenceReport:
    r"""
    Outcome of a convergence check against a simulation's analytic reference.

    Attributes
    ----------
    status : {"pass", "fail", "no-oracle"}
        ``"no-oracle"`` if the simulation does not declare an
        :meth:`~mcframework.core.MonteCarloSimulation.analytic_reference`;
        otherwise ``"pass"``/``"fail"`` from ``within_tol``.
    n : int
        Number of simulation draws used.
    oracle : float or None
        The known reference value (``None`` when no oracle exists).
    estimate : float or None
        The Monte Carlo estimate of the mean.
    abs_error : float or None
        ``|estimate - oracle|``.
    rel_error : float or None
        ``abs_error / |oracle|`` (``None`` when ``oracle`` is ``0`` or missing).
    se : float or None
        Standard error of the mean.
    ci_low, ci_high : float or None
        Confidence-interval endpoints for the mean.
    within_ci : bool
        Whether ``oracle`` lies within ``[ci_low, ci_high]``.
    within_tol : bool
        Whether ``abs_error <= sigma_tol * se``.
    sigma_tol : float
        Standard-error multiplier used for ``within_tol``.
    confidence : float
        Confidence level used for the interval.
    reference_source : str
        Citation for the oracle, copied from the simulation.
    reference_kind : str
        Kind of reference: ``"closed-form"``, ``"benchmark"``, ``"limit"``, or ``""``.
    """

    status: str
    n: int
    oracle: float | None = None
    estimate: float | None = None
    abs_error: float | None = None
    rel_error: float | None = None
    se: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    within_ci: bool = False
    within_tol: bool = False
    sigma_tol: float = 5.0
    confidence: float = 0.99
    reference_source: str = ""
    reference_kind: str = ""


def _ci_bounds(ci: Any) -> tuple[float, float] | None:
    """Extract (low, high) from the engine's ci_mean dict or the legacy tuple form."""
    if isinstance(ci, dict) and "low" in ci and "high" in ci:
        return float(ci["low"]), float(ci["high"])
    if isinstance(ci, (tuple, list)) and len(ci) == 2:
        return float(ci[0]), float(ci[1])
    return None


def validate_convergence(
    sim: MonteCarloSimulation,
    n: int,
    *,
    sigma_tol: float = 5.0,
    confidence: float = 0.99,
    seed: int = 0,
    **params: Any,
) -> ConvergenceReport:
    r"""
    Check that a simulation's Monte Carlo estimate converges to its analytic reference.

    Parameters
    ----------
    sim : MonteCarloSimulation
        The simulation to validate. Must implement
        :meth:`~mcframework.core.MonteCarloSimulation.analytic_reference` to be
        checkable; otherwise a ``"no-oracle"`` report is returned.
    n : int
        Number of simulation draws.
    sigma_tol : float, default 5.0
        Tolerance for ``ConvergenceReport.within_tol``, expressed as a multiple of
        the standard error. The default of ``5`` sigma makes a fixed-seed check
        essentially non-flaky while still catching real bugs.
    confidence : float, default 0.99
        Confidence level for the interval used by ``ConvergenceReport.within_ci``.
    seed : int, default 0
        Seed applied via :meth:`~mcframework.core.MonteCarloSimulation.set_seed`
        for reproducible validation.
    **params : Any
        Forwarded to both :meth:`~mcframework.core.MonteCarloSimulation.analytic_reference`
        and :meth:`~mcframework.core.MonteCarloSimulation.run`, so the oracle and the
        run use identical parameters.

    Returns
    -------
    ConvergenceReport
        The structured outcome. ``status`` is ``"no-oracle"`` when no reference is
        declared, else ``"pass"``/``"fail"`` based on ``ConvergenceReport.within_tol``.

    Notes
    -----
    The oracle is threaded through the stats engine as the context ``target`` (via
    ``run(..., extra_context={"target": oracle})``), which also populates
    ``bias_to_target``/``mse_to_target`` in the result for free.
    """
    oracle = sim.analytic_reference(**params)
    reference_source = getattr(sim, "reference_source", "")
    reference_kind = getattr(sim, "reference_kind", "")

    if oracle is None:
        return ConvergenceReport(
            status="no-oracle",
            n=n,
            sigma_tol=sigma_tol,
            confidence=confidence,
            reference_source=reference_source,
            reference_kind=reference_kind,
        )

    oracle = float(oracle)
    sim.set_seed(seed)
    res = sim.run(
        n,
        confidence=confidence,
        extra_context={"target": oracle},
        **params,
    )

    estimate = float(res.mean)
    abs_error = abs(estimate - oracle)
    rel_error = abs_error / abs(oracle) if oracle != 0.0 else None
    se = float(res.std) / np.sqrt(max(1, n))

    bounds = _ci_bounds(res.stats.get("ci_mean"))
    ci_low: float | None = None
    ci_high: float | None = None
    within_ci = False
    if bounds is not None:
        ci_low, ci_high = bounds
        within_ci = ci_low <= oracle <= ci_high

    within_tol = abs_error <= sigma_tol * se

    return ConvergenceReport(
        status="pass" if within_tol else "fail",
        n=n,
        oracle=oracle,
        estimate=estimate,
        abs_error=abs_error,
        rel_error=rel_error,
        se=se,
        ci_low=ci_low,
        ci_high=ci_high,
        within_ci=within_ci,
        within_tol=within_tol,
        sigma_tol=sigma_tol,
        confidence=confidence,
        reference_source=reference_source,
        reference_kind=reference_kind,
    )
