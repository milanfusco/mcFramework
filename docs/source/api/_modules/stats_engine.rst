Stats Engine
============

.. currentmodule:: mcframework.stats_engine

Statistical metrics and confidence interval computations for Monte Carlo simulation results.

.. contents:: On This Page
   :local:
   :depth: 2

----

Quick Start
-----------

.. code-block:: python

   from mcframework.stats_engine import DEFAULT_ENGINE, StatsContext
   import numpy as np

   # Your simulation results
   data = np.random.normal(100, 15, size=10_000)

   # Configure and compute
   ctx = StatsContext(n=len(data), confidence=0.95)
   result = DEFAULT_ENGINE.compute(data, ctx)

   print(f"Mean: {result.metrics['mean']:.2f}")
   print(f"95% CI: [{result.metrics['ci_mean']['low']:.2f}, {result.metrics['ci_mean']['high']:.2f}]")

----

Configuration
-------------

StatsContext
~~~~~~~~~~~~

The :class:`StatsContext` dataclass configures all metric computations:

.. code-block:: python

   ctx = StatsContext(
       n=10_000,              # Sample size (required)
       confidence=0.95,       # CI confidence level
       ci_method="auto",      # "auto", "z", "t", or "bootstrap"
       percentiles=(5, 50, 95),
       nan_policy="propagate",  # or "omit"
   )

.. autosummary::
   :toctree: generated
   :nosignatures:

   StatsContext

See :class:`StatsContext` for full attribute documentation with examples.

**Quick Reference:**

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Field
     - Default
     - Description
   * - ``n``
     - (required)
     - Declared sample size
   * - ``confidence``
     - 0.95
     - Confidence level ∈ (0, 1)
   * - ``ci_method``
     - "auto"
     - CI method: "auto", "z", "t", "bootstrap"
   * - ``percentiles``
     - (5,25,50,75,95)
     - Quantiles to compute
   * - ``nan_policy``
     - "propagate"
     - "propagate" or "omit" non-finite values
   * - ``ddof``
     - 1
     - Degrees of freedom for std
   * - ``target``
     - None
     - Target value for bias/MSE metrics
   * - ``eps``
     - None
     - Error tolerance for Chebyshev/Markov
   * - ``n_bootstrap``
     - 10,000
     - Bootstrap resamples
   * - ``bootstrap``
     - "percentile"
     - Bootstrap method: "percentile" or "bca"
   * - ``rng``
     - None
     - Seed or Generator for reproducibility

**Helper Properties:**

.. code-block:: python

   ctx.alpha           # Tail probability: 1 - confidence
   ctx.q_bound()       # Percentile bounds: (2.5, 97.5) for 95% CI
   ctx.eff_n(len(x))   # Effective sample size
   ctx.with_overrides(confidence=0.99)  # Create modified copy

----

Metrics
-------

Descriptive Statistics
~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:

   mean
   std
   percentiles
   skew
   kurtosis

**Usage:**

.. code-block:: python

   from mcframework.stats_engine import mean, std, percentiles

   m = mean(data, ctx)           # Sample mean
   s = std(data, ctx)            # Sample std (ddof=1)
   pcts = percentiles(data, ctx) # {5: ..., 25: ..., 50: ..., 75: ..., 95: ...}

Confidence Intervals
~~~~~~~~~~~~~~~~~~~~

Three methods for computing CIs on the mean:

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * - Method
     - Function
     - Best For
   * - **Parametric**
     - :func:`ci_mean`
     - Large samples, normal-ish data
   * - **Bootstrap**
     - :func:`ci_mean_bootstrap`
     - Non-normal distributions
   * - **Chebyshev**
     - :func:`ci_mean_chebyshev`
     - Distribution-free guarantees

.. autosummary::
   :toctree: generated
   :nosignatures:

   ci_mean
   ci_mean_bootstrap
   ci_mean_chebyshev

**Parametric CI** (z or t):

.. code-block:: python

   from mcframework.stats_engine import ci_mean

   result = ci_mean(data, ctx)
   # {'confidence': 0.95, 'method': 'z', 'low': 99.5, 'high': 100.5, 'se': 0.15, 'crit': 1.96}

**Bootstrap CI**:

.. code-block:: python

   from mcframework.stats_engine import ci_mean_bootstrap

   ctx = StatsContext(n=len(data), bootstrap="bca", rng=42)
   result = ci_mean_bootstrap(data, ctx)
   # {'confidence': 0.95, 'method': 'bootstrap-bca', 'low': 99.4, 'high': 100.6}

Target-Based Metrics
~~~~~~~~~~~~~~~~~~~~

Metrics that compare results to a known target value:

.. autosummary::
   :toctree: generated
   :nosignatures:

   bias_to_target
   mse_to_target
   markov_error_prob
   chebyshev_required_n

**Usage** (requires ``target`` and/or ``eps`` in context):

.. code-block:: python

   ctx = StatsContext(n=len(data), target=100.0, eps=0.5)
   
   bias = bias_to_target(data, ctx)      # Mean - target
   mse = mse_to_target(data, ctx)        # Mean squared error
   prob = markov_error_prob(data, ctx)   # P(|mean - target| >= eps)
   req_n = chebyshev_required_n(data, ctx)  # Required n for precision

----

Engine
------

The :class:`StatsEngine` orchestrates multiple metrics at once:

.. code-block:: python

   from mcframework.stats_engine import StatsEngine, FnMetric, mean, std, ci_mean

   engine = StatsEngine([
       FnMetric("mean", mean),
       FnMetric("std", std),
       FnMetric("ci_mean", ci_mean),
   ])

   result = engine.compute(data, ctx)
   print(result.metrics)   # {'mean': 100.1, 'std': 15.2, 'ci_mean': {...}}
   print(result.skipped)   # Metrics skipped due to missing context
   print(result.errors)    # Metrics that raised errors

.. autosummary::
   :toctree: generated
   :nosignatures:

   StatsEngine
   FnMetric
   ComputeResult

**Default Engine:**

A pre-configured engine with all standard metrics:

.. code-block:: python

   from mcframework.stats_engine import DEFAULT_ENGINE

   result = DEFAULT_ENGINE.compute(data, ctx)

Includes: mean, std, percentiles, skew, kurtosis, ci_mean, ci_mean_bootstrap,
ci_mean_chebyshev, chebyshev_required_n, markov_error_prob, bias_to_target, mse_to_target.

.. autosummary::
   :toctree: generated
   :nosignatures:

   build_default_engine

----

Enumerations
------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   CIMethod
   NanPolicy
   BootstrapMethod

.. code-block:: python

   from mcframework.stats_engine import CIMethod, NanPolicy, BootstrapMethod

   ctx = StatsContext(
       n=100,
       ci_method=CIMethod.auto,      # or "auto"
       nan_policy=NanPolicy.omit,    # or "omit"
       bootstrap=BootstrapMethod.bca # or "bca"
   )

----

Exceptions
----------

.. autosummary::
   :toctree: generated
   :nosignatures:

   MissingContextError
   InsufficientDataError

.. code-block:: python

   from mcframework.stats_engine import MissingContextError

   try:
       bias_to_target(data, StatsContext(n=100))  # Missing 'target'
   except MissingContextError as e:
       print(f"Missing field: {e}")

----

Custom Metrics
--------------

Create custom metrics with :class:`FnMetric`:

.. code-block:: python

   import numpy as np
   from mcframework.stats_engine import FnMetric, StatsEngine, StatsContext

   def coefficient_of_variation(x, ctx):
       """CV = std / mean"""
       m, s = float(np.mean(x)), float(np.std(x, ddof=1))
       return s / m if m != 0 else float('nan')

   def interquartile_range(x, ctx):
       """IQR = Q3 - Q1"""
       q1, q3 = np.percentile(x, [25, 75])
       return float(q3 - q1)

   # Build custom engine
   engine = StatsEngine([
       FnMetric("cv", coefficient_of_variation, doc="Coefficient of variation"),
       FnMetric("iqr", interquartile_range, doc="Interquartile range"),
   ])

   result = engine.compute(data, StatsContext(n=len(data)))

----

See Also
--------

- :doc:`core` — Simulation classes that use this engine
- :doc:`utils` — Critical value utilities (z_crit, t_crit)
