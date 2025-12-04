Utilities Module
================

.. currentmodule:: mcframework.utils

The utilities module provides critical value functions for constructing confidence
intervals. These are the building blocks used by the :doc:`stats_engine`.

Overview
--------

When constructing a two-sided confidence interval for the mean:

.. math::

   \bar{X} \pm c \cdot \frac{s}{\sqrt{n}}

the critical value :math:`c` determines the interval width. This module provides:

- :func:`z_crit` — Normal distribution critical values
- :func:`t_crit` — Student's t-distribution critical values  
- :func:`autocrit` — Automatic selection based on sample size


Critical Values
---------------

z Critical Value (Normal)
~~~~~~~~~~~~~~~~~~~~~~~~~

For large samples (:math:`n \ge 30`), use the normal approximation:

.. math::

   z_{\alpha/2} = \Phi^{-1}\left(1 - \frac{\alpha}{2}\right)

where :math:`\Phi^{-1}` is the inverse standard normal CDF and :math:`\alpha = 1 - \text{confidence}`.

.. code-block:: python

   from mcframework.utils import z_crit

   z_crit(0.95)   # 1.96 (95% CI)
   z_crit(0.99)   # 2.576 (99% CI)
   z_crit(0.90)   # 1.645 (90% CI)

**Common Values:**

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Confidence
     - α
     - z-critical
   * - 90%
     - 0.10
     - 1.645
   * - 95%
     - 0.05
     - 1.960
   * - 99%
     - 0.01
     - 2.576


t Critical Value (Student's t)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For small samples or when population variance is unknown, use the t-distribution:

.. math::

   t_{\alpha/2, \text{df}} = T_{\text{df}}^{-1}\left(1 - \frac{\alpha}{2}\right)

where :math:`\text{df} = n - 1` degrees of freedom.

.. code-block:: python

   from mcframework.utils import t_crit

   t_crit(0.95, df=9)    # 2.262 (n=10)
   t_crit(0.95, df=29)   # 2.045 (n=30)
   t_crit(0.95, df=99)   # 1.984 (n=100, approaches z)

The t critical value is always larger than z for finite df, yielding wider (more conservative) intervals.


Automatic Selection
~~~~~~~~~~~~~~~~~~~

The :func:`autocrit` function chooses between z and t based on sample size:

.. code-block:: python

   from mcframework.utils import autocrit

   # Small sample → use t
   crit, method = autocrit(0.95, n=15)
   print(f"{method}: {crit:.3f}")  # t: 2.145

   # Large sample → use z
   crit, method = autocrit(0.95, n=100)
   print(f"{method}: {crit:.3f}")  # z: 1.960

   # Force specific method
   crit, method = autocrit(0.95, n=100, method="t")
   print(f"{method}: {crit:.3f}")  # t: 1.984

**Selection Rules:**

- ``method="auto"`` (default): Use t if :math:`n < 30`, otherwise z
- ``method="z"``: Always use normal critical value
- ``method="t"``: Always use t with :math:`\text{df} = \max(1, n-1)`


Usage with Stats Engine
-----------------------

The stats engine uses these utilities internally:

.. code-block:: python

   from mcframework.stats_engine import ci_mean

   # ci_method controls which critical value is used
   ci_mean(data, {"n": 25, "ci_method": "auto"})   # Uses t (n < 30)
   ci_mean(data, {"n": 25, "ci_method": "z"})      # Forces z
   ci_mean(data, {"n": 100, "ci_method": "auto"})  # Uses z (n ≥ 30)


Mathematical Background
-----------------------

**Why the n < 30 threshold?**

The threshold comes from the convergence of the t-distribution to normal:

- At df=29, the 97.5th percentile differs from z by only ~4%
- Below df=10, the difference exceeds 10%
- The t-distribution accounts for additional uncertainty in estimating variance

**Coverage Probability:**

A confidence interval has "coverage" :math:`1 - \alpha` if:

.. math::

   \Pr\left(\mu \in \left[\bar{X} - c \cdot \text{SE}, \bar{X} + c \cdot \text{SE}\right]\right) = 1 - \alpha

Using t critical values with small samples ensures proper coverage even when the population variance is unknown.


Module Reference
----------------

.. automodule:: mcframework.utils
   :no-members:
   :no-undoc-members:
   :no-inherited-members:

Functions
~~~~~~~~~

.. autosummary::
   :toctree: generated
   :recursive:
   :nosignatures:

   z_crit
   t_crit
   autocrit
   _validate_confidence


See Also
--------

- :doc:`stats_engine` — Uses these utilities for confidence intervals
- :func:`~mcframework.stats_engine.ci_mean` — Parametric CI for the mean
- :func:`~mcframework.stats_engine.ci_mean_chebyshev` — Distribution-free alternative
