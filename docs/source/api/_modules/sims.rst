Simulations Module
==================

.. currentmodule:: mcframework.sims

The ``sims`` module provides ready-to-use Monte Carlo simulation classes covering
classic problems in probability and quantitative finance.

Overview
--------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Simulation
     - Use Case
   * - :class:`PiEstimationSimulation`
     - Estimate π via geometric probability (unit disk sampling)
   * - :class:`PortfolioSimulation`
     - Model investment growth under GBM or arithmetic returns
   * - :class:`BlackScholesSimulation`
     - Price European/American options with Greeks calculation
   * - :class:`BlackScholesPathSimulation`
     - Generate stock price paths for scenario analysis


Pi Estimation
-------------

Estimate π using the Monte Carlo integration identity:

.. math::

   \pi = 4 \cdot \Pr\left(X^2 + Y^2 \le 1 \mid X, Y \sim \text{Uniform}(-1, 1)\right)

**Example:**

.. code-block:: python

   from mcframework import PiEstimationSimulation

   sim = PiEstimationSimulation()
   sim.set_seed(42)
   
   result = sim.run(
       n_simulations=50_000,
       n_points=10_000,        # Points per simulation
       antithetic=True,        # Variance reduction
       backend="thread"
   )
   
   print(f"π ≈ {result.mean:.6f}")
   print(f"Std error: {result.std / (result.n_simulations ** 0.5):.6f}")

**Parameters:**

- ``n_points``: Number of random points thrown per simulation (default: 10,000)
- ``antithetic``: Enable antithetic sampling for variance reduction (default: False)


Portfolio Simulation
--------------------

Model portfolio growth under Geometric Brownian Motion:

.. math::

   V_T = V_0 \exp\left(\sum_{k=1}^{n} \left[(\mu - \tfrac{1}{2}\sigma^2)\Delta t + \sigma\sqrt{\Delta t} \cdot Z_k\right]\right)

where :math:`Z_k \sim \mathcal{N}(0, 1)` and :math:`\Delta t = 1/252` (daily steps).

**Example:**

.. code-block:: python

   from mcframework import PortfolioSimulation

   sim = PortfolioSimulation()
   sim.set_seed(2024)
   
   result = sim.run(
       n_simulations=100_000,
       initial_value=100_000,   # Starting capital
       annual_return=0.08,      # 8% expected return
       volatility=0.20,         # 20% annual volatility
       years=20,                # Investment horizon
       use_gbm=True,            # GBM dynamics
       backend="thread",
       percentiles=(5, 25, 50, 75, 95)
   )
   
   print(f"Expected terminal value: ${result.mean:,.0f}")
   print(f"Median: ${result.percentiles[50]:,.0f}")
   print(f"5% VaR: ${result.percentiles[5]:,.0f}")

**Parameters:**

- ``initial_value``: Starting wealth (default: 10,000)
- ``annual_return``: Expected annualized return μ (default: 0.07)
- ``volatility``: Annualized volatility σ (default: 0.20)
- ``years``: Investment horizon in years (default: 10)
- ``use_gbm``: Use geometric (True) or arithmetic (False) returns (default: True)


Black-Scholes Option Pricing
----------------------------

Price European and American options using Monte Carlo simulation with support for Greeks
calculation via finite differences.

**Stock Price Dynamics:**

.. math::

   dS_t = r S_t \, dt + \sigma S_t \, dW_t

**European Option Payoff:**

.. math::

   \Phi_{\text{call}}(S_T) = \max(S_T - K, 0), \quad
   \Phi_{\text{put}}(S_T) = \max(K - S_T, 0)

**Example - European Call:**

.. code-block:: python

   from mcframework import BlackScholesSimulation

   bs = BlackScholesSimulation()
   bs.set_seed(42)
   
   result = bs.run(
       n_simulations=100_000,
       S0=100.0,              # Current stock price
       K=105.0,               # Strike price
       T=0.5,                 # 6 months to maturity
       r=0.05,                # 5% risk-free rate
       sigma=0.25,            # 25% volatility
       option_type="call",
       exercise_type="european",
       backend="thread"
   )
   
   print(f"Call Price: ${result.mean:.4f}")
   print(f"95% CI: [${result.stats['ci_mean'][0]:.4f}, ${result.stats['ci_mean'][1]:.4f}]")

**Example - Greeks Calculation:**

.. code-block:: python

   greeks = bs.calculate_greeks(
       n_simulations=50_000,
       S0=100, K=100, T=1.0, r=0.05, sigma=0.20,
       option_type="call",
       exercise_type="european",
       backend="thread"
   )
   
   print(f"Delta: {greeks['delta']:.4f}")   # ∂V/∂S
   print(f"Gamma: {greeks['gamma']:.6f}")   # ∂²V/∂S²
   print(f"Vega:  {greeks['vega']:.4f}")    # ∂V/∂σ (per 1% vol move)
   print(f"Theta: {greeks['theta']:.4f}")   # -∂V/∂t (daily decay)
   print(f"Rho:   {greeks['rho']:.4f}")     # ∂V/∂r (per 1% rate move)

**American Options:**

For American options, the simulation uses the Longstaff-Schwartz LSM algorithm
with polynomial basis regression for early exercise decisions.

.. code-block:: python

   result = bs.run(
       n_simulations=50_000,
       S0=100, K=100, T=1.0, r=0.05, sigma=0.20,
       option_type="put",
       exercise_type="american",   # LSM algorithm
       n_steps=252,                # Daily time steps
       backend="thread"
   )

**Parameters:**

- ``S0``: Initial stock price (default: 100.0)
- ``K``: Strike price (default: 100.0)
- ``T``: Time to maturity in years (default: 1.0)
- ``r``: Risk-free interest rate (default: 0.05)
- ``sigma``: Volatility (default: 0.20)
- ``option_type``: ``"call"`` or ``"put"``
- ``exercise_type``: ``"european"`` or ``"american"``
- ``n_steps``: Number of time steps (default: 252)


Black-Scholes Path Simulation
-----------------------------

Generate stock price paths for visualization and scenario analysis:

.. code-block:: python

   from mcframework import BlackScholesPathSimulation
   import matplotlib.pyplot as plt

   path_sim = BlackScholesPathSimulation()
   path_sim.set_seed(42)
   
   # Generate 1000 paths
   paths = path_sim.simulate_paths(
       n_paths=1000,
       S0=100,
       r=0.05,
       sigma=0.25,
       T=1.0,
       n_steps=252
   )
   
   # Plot first 50 paths
   plt.figure(figsize=(10, 6))
   plt.plot(paths[:50].T, alpha=0.3)
   plt.xlabel("Trading Day")
   plt.ylabel("Stock Price")
   plt.title("Simulated GBM Paths")
   plt.show()


Module Reference
----------------

.. automodule:: mcframework.sims
   :no-members:
   :no-undoc-members:
   :no-inherited-members:

Simulation Classes
~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :recursive:

   PiEstimationSimulation
   PortfolioSimulation
   BlackScholesSimulation
   BlackScholesPathSimulation

Helper Functions
~~~~~~~~~~~~~~~~

These low-level functions power the Black-Scholes simulations:

.. autofunction:: _simulate_gbm_path

.. autofunction:: _european_payoff

.. autofunction:: _american_exercise_lsm


See Also
--------

- :doc:`core` — Base classes and framework
- :doc:`backends` — Execution backends (sequential, parallel, GPU)
- :doc:`stats_engine` — Statistical analysis of results
- ``demos/gui/`` — Interactive Black-Scholes GUI application
