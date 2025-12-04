.. mcframework documentation master file

================================
mcframework
================================

**Lightweight, reproducible, and deterministic Monte Carlo simulation framework**

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 🎯 Deterministic & Reproducible
      
      Parallel execution with SeedSequence spawning ensures identical results 
      regardless of worker scheduling or system load.

   .. grid-item-card:: ⚡ Parallel by Design
      
      Automatic backend selection (threads vs processes) optimized per platform.
      NumPy's GIL-releasing RNGs enable efficient thread-based parallelism.

   .. grid-item-card:: 📊 Rich Statistics Engine
      
      Built-in metrics: mean, std, percentiles, skew, kurtosis, and multiple 
      CI methods (z, t, bootstrap, Chebyshev).

   .. grid-item-card:: 🔌 Extensible Architecture
      
      Simple base class design—just implement ``single_simulation()`` and 
      the framework handles parallelism, statistics, and result management.

Quick Example
-------------

.. code-block:: python

   from mcframework import MonteCarloFramework, PiEstimationSimulation

   # Create and configure simulation
   sim = PiEstimationSimulation()
   sim.set_seed(42)  # Reproducible results

   # Register with framework and run
   fw = MonteCarloFramework()
   fw.register_simulation(sim)
   result = fw.run_simulation("Pi Estimation", n_simulations=50_000, parallel=True)

   # Access results
   print(f"π ≈ {result.mean:.6f}")
   print(f"95% CI: [{result.stats['ci_mean'][0]:.6f}, {result.stats['ci_mean'][1]:.6f}]")

.. code-block:: text
   :caption: Output

   π ≈ 3.141592
   95% CI: [3.140821, 3.142363]


Installation
------------

**From Source (Development)**

.. code-block:: bash

   git clone https://github.com/milanfusco/mcframework.git
   cd mcframework
   pip install -e .

**With Optional Dependencies**

.. code-block:: bash

   # All dependencies (dev, test, docs, gui)
   pip install -e ".[dev,test,docs,gui]"

   # Just testing
   pip install -e ".[test]"

   # GUI application (PySide6)
   pip install -e ".[gui]"


Architecture Overview
---------------------

The framework is organized into four main components:

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────┐
   │                    MonteCarloFramework                      │
   │         (Registry for managing multiple simulations)        │
   └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
   ┌─────────────────────────────────────────────────────────────┐
   │                   MonteCarloSimulation                      │
   │   • single_simulation() - implement your logic here         │
   │   • run() - sequential or parallel execution                │
   │   • set_seed() - reproducible RNG via SeedSequence          │
   └─────────────────────────────────────────────────────────────┘
                                │
              ┌─────────────────┴─────────────────┐
              ▼                                   ▼
   ┌─────────────────────┐            ┌─────────────────────────┐
   │    StatsEngine      │            │    SimulationResult     │
   │  • mean, std        │            │  • results array        │
   │  • percentiles      │            │  • mean, std            │
   │  • ci_mean (z/t)    │            │  • percentiles          │
   │  • bootstrap CI     │            │  • stats dict           │
   │  • Chebyshev CI     │            │  • metadata             │
   └─────────────────────┘            └─────────────────────────┘


Built-in Simulations
--------------------

The framework includes several ready-to-use simulation classes:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Simulation
     - Description
   * - :class:`~mcframework.sims.PiEstimationSimulation`
     - Estimate π using geometric probability on the unit disk
   * - :class:`~mcframework.sims.PortfolioSimulation`
     - Model portfolio growth under GBM or arithmetic returns
   * - :class:`~mcframework.sims.BlackScholesSimulation`
     - Price European/American options with Greeks calculation
   * - :class:`~mcframework.sims.BlackScholesPathSimulation`
     - Generate GBM stock price paths for analysis


Custom Simulation Example
-------------------------

Creating your own simulation requires just one method:

.. code-block:: python

   from mcframework import MonteCarloSimulation
   import numpy as np

   class DiceSumSimulation(MonteCarloSimulation):
       """Simulate the sum of N dice rolls."""
       
       def __init__(self):
           super().__init__("Dice Sum")
       
       def single_simulation(self, _rng=None, n_dice: int = 5) -> float:
           rng = self._rng(_rng, self.rng)  # Get thread-safe RNG
           return float(rng.integers(1, 7, size=n_dice).sum())

   # Use it
   sim = DiceSumSimulation()
   sim.set_seed(123)
   result = sim.run(100_000, n_dice=3, parallel=True, percentiles=(5, 50, 95))
   print(f"Expected sum of 3d6: {result.mean:.2f}")  # ~10.5


Cross-Platform Parallelism
--------------------------

The ``parallel_backend`` setting controls execution strategy:

- **"auto"** (default): Threads on POSIX (NumPy releases GIL), processes on Windows
- **"thread"**: Force thread-based parallelism
- **"process"**: Force process-based parallelism (uses spawn context)

.. code-block:: python

   sim = PiEstimationSimulation()
   sim.parallel_backend = "thread"  # Override default
   result = sim.run(100_000, parallel=True, n_workers=8)


.. toctree::
   :maxdepth: 2
   :caption: Guides

   getting-started

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/_modules/core
   api/_modules/stats_engine
   api/_modules/sims
   api/_modules/utils


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
