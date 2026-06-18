Validation Module
=================

.. currentmodule:: mcframework.validation

The validation module turns "does my simulation converge to the right answer?" into a
single, repeatable call. Given a simulation that declares an analytic reference
("oracle") via :meth:`~mcframework.core.MonteCarloSimulation.analytic_reference`, it
runs the simulation and reports whether the Monte Carlo estimate lands on the known
value.

Overview
--------

A Monte Carlo estimator :math:`\widehat{\theta}_n` is trustworthy only insofar as you
can check it. When a closed-form answer :math:`\theta` exists, that answer is an
*oracle*: a checkable ground truth. This module threads the oracle through the stats
engine's existing ``target`` machinery and reports two complementary checks:

- ``within_ci``  -- the oracle lies inside the computed confidence interval.
- ``within_tol`` -- the absolute error is within ``sigma_tol`` standard errors:

.. math::

   \left| \widehat{\theta}_n - \theta \right| \le k \cdot \mathrm{SE},
   \qquad \mathrm{SE} = \frac{s}{\sqrt{n}},

with :math:`k = \texttt{sigma\_tol}` (default 5). At a fixed seed this check is
essentially never flaky, yet a genuine implementation bug lands many standard errors
away and fails loudly. That makes it an ideal CI gate.

Usage
-----

.. code-block:: python

   from mcframework import validate_convergence, PiEstimationSimulation

   report = validate_convergence(PiEstimationSimulation(), 200_000, seed=0)
   print(report.status)      # "pass"
   print(report.estimate)    # ≈ 3.14159
   print(report.abs_error)   # small multiple of the standard error

If a simulation has no oracle, the report's ``status`` is ``"no-oracle"`` and no run is
performed — a deliberate signal that the simulation is research, not a verifiable demo:

.. code-block:: python

   report = validate_convergence(SomeResearchSim(), 10_000)
   assert report.status == "no-oracle"

Adding an oracle to a custom simulation
---------------------------------------

Override :meth:`~mcframework.core.MonteCarloSimulation.analytic_reference` to return the
known expected value of a single draw, and set ``reference_source`` to cite it:

.. code-block:: python

   import math
   from mcframework import MonteCarloSimulation

   class PiSim(MonteCarloSimulation):
       reference_source = "Closed form: 4 * P(X^2 + Y^2 <= 1) = pi"

       def single_simulation(self, _rng=None, **kwargs):
           ...

       def analytic_reference(self, **params):
           return math.pi


Module Reference
----------------

.. automodule:: mcframework.validation
   :no-members:
   :no-undoc-members:
   :no-inherited-members:

Classes
~~~~~~~

.. autosummary::
   :toctree: generated
   :recursive:
   :nosignatures:

   ConvergenceReport

Functions
~~~~~~~~~

.. autosummary::
   :toctree: generated
   :recursive:
   :nosignatures:

   validate_convergence


See Also
--------

- :doc:`stats_engine`: Provides the ``target``-based metrics and ``ci_mean`` reused here
- :meth:`~mcframework.core.MonteCarloSimulation.analytic_reference`: The oracle hook
- :func:`~mcframework.stats_engine.ci_mean`: Parametric CI for the mean
