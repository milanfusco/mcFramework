Core Module
===========

.. currentmodule:: mcframework.core

The foundational classes for building and running Monte Carlo simulations with
parallel execution, reproducible seeding, and result aggregation.

.. contents:: On This Page
   :local:
   :depth: 2

----

Quick Start
-----------

.. code-block:: python

   from mcframework.core import MonteCarloSimulation
   import numpy as np

   class DiceSim(MonteCarloSimulation):
       def single_simulation(self, _rng=None, n_dice=2):
           rng = self._rng(_rng, self.rng)
           return float(rng.integers(1, 7, size=n_dice).sum())

   sim = DiceSim(name="2d6")
   sim.set_seed(42)
   result = sim.run(50_000, backend="thread")

   print(f"Mean: {result.mean:.2f}")          # ~7.0
   print(f"95% CI: {result.stats['ci_mean']}")

----

Classes
-------

SimulationResult
~~~~~~~~~~~~~~~~

Container for simulation outputs and computed statistics.

.. autosummary::
   :toctree: generated
   :nosignatures:

   SimulationResult

See :class:`SimulationResult` for full attribute documentation.

**Quick Reference:**

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Attribute
     - Description
   * - ``results``
     - Raw NumPy array of simulation values (length = ``n_simulations``)
   * - ``n_simulations``
     - Number of simulation draws performed
   * - ``execution_time``
     - Wall-clock time in seconds
   * - ``mean``
     - Sample mean :math:`\bar{X}`
   * - ``std``
     - Sample standard deviation (ddof=1)
   * - ``percentiles``
     - Dict mapping percentile keys to values, e.g., ``{5: 0.12, 50: 0.50, 95: 0.88}``
   * - ``stats``
     - Additional statistics from the stats engine (``ci_mean``, ``skew``, etc.)
   * - ``metadata``
     - Freeform dict with ``simulation_name``, ``timestamp``, ``seed_entropy``, etc.

**Usage:**

.. code-block:: python

   result = sim.run(10_000)
   
   # Access raw data
   raw = result.results  # np.ndarray
   
   # Access computed stats
   print(result.mean, result.std)
   print(result.percentiles[50])  # Median
   print(result.stats['ci_mean'])  # {'low': ..., 'high': ..., ...}
   
   # Pretty print
   print(result.result_to_string())


MonteCarloSimulation
~~~~~~~~~~~~~~~~~~~~

Abstract base class for defining simulations. Subclass and implement
:meth:`~MonteCarloSimulation.single_simulation`.

.. autosummary::
   :toctree: generated
   :nosignatures:

   MonteCarloSimulation

**Key Attributes:**

.. currentmodule:: mcframework.simulation

.. autosummary::
   :toctree: generated
   :template: autosummary/accessor_attribute.rst

   MonteCarloSimulation.supports_batch

.. currentmodule:: mcframework.core

**Key Methods:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Description
   * - ``single_simulation(**kwargs)``
     - **Abstract.** Implement to define your simulation logic. Return a ``float``.
   * - ``run(n, backend="auto", ...)``
     - Execute ``n`` simulations and return :class:`SimulationResult`.
   * - ``set_seed(seed)``
     - Initialize RNG with a seed for reproducibility.
   * - ``_rng(rng, default)``
     - Helper to select thread-local RNG inside ``single_simulation``.
   * - ``torch_batch(n, device, generator)``
     - **Optional.** Vectorized Torch implementation for GPU acceleration.
   * - ``supports_batch``
     - **Class attribute.** Set to ``True`` to enable Torch batch execution.

**Example Implementation:**

.. code-block:: python

   from mcframework.core import MonteCarloSimulation

   class PiEstimator(MonteCarloSimulation):
       """Estimate π using random points in a unit square."""
       
       def single_simulation(self, _rng=None, n_points=10_000):
           rng = self._rng(_rng, self.rng)
           x = rng.random(n_points)
           y = rng.random(n_points)
           inside = (x*x + y*y) <= 1.0
           return 4.0 * inside.mean()

   sim = PiEstimator(name="Pi")
   sim.set_seed(42)
   result = sim.run(1000, backend="thread")
   print(f"π ≈ {result.mean:.6f}")


MonteCarloFramework
~~~~~~~~~~~~~~~~~~~

Registry and runner for managing multiple simulations.

.. autosummary::
   :toctree: generated
   :nosignatures:

   MonteCarloFramework

**Usage:**

.. code-block:: python

   from mcframework.core import MonteCarloFramework

   framework = MonteCarloFramework()
   framework.register_simulation(sim1)
   framework.register_simulation(sim2)

   # Run simulations
   res1 = framework.run_simulation("Sim1", 10_000, backend="thread")
   res2 = framework.run_simulation("Sim2", 10_000)

   # Compare results
   comparison = framework.compare_results(["Sim1", "Sim2"], metric="mean")
   print(comparison)  # {'Sim1': 3.14, 'Sim2': 3.15}

----

Execution Backends
------------------

The framework supports multiple execution backends via the ``backend`` parameter:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Backend
     - Description
     - Best For
   * - **sequential**
     - Single-threaded execution
     - Debugging, small jobs (< 20K)
   * - **thread**
     - :class:`~concurrent.futures.ThreadPoolExecutor`
     - NumPy-heavy code (releases GIL)
   * - **process**
     - :class:`~concurrent.futures.ProcessPoolExecutor` with spawn
     - Python-bound code, Windows
   * - **torch**
     - GPU-accelerated batch execution
     - Large jobs (100K+), GPU available

**Auto Selection:**

- **Small jobs (< 20K):** Sequential execution
- **POSIX (macOS/Linux):** Defaults to threads (NumPy releases GIL)
- **Windows:** Defaults to processes (threads serialize under GIL)

.. code-block:: python

   # Explicit backend selection
   result = sim.run(100_000, backend="sequential")  # Single-threaded
   result = sim.run(100_000, backend="thread", n_workers=8)
   result = sim.run(100_000, backend="process", n_workers=4)
   result = sim.run(100_000, backend="auto")  # Platform default

   # GPU backends (requires pip install mcframework[gpu])
   result = sim.run(1_000_000, backend="torch", torch_device="cpu")
   result = sim.run(1_000_000, backend="torch", torch_device="mps")   # Apple Silicon
   result = sim.run(1_000_000, backend="torch", torch_device="cuda")  # NVIDIA GPU

----

Reproducibility
---------------

Reproducible results via NumPy's :class:`~numpy.random.SeedSequence`:

.. code-block:: python

   sim.set_seed(42)
   result1 = sim.run(10_000, backend="thread")

   sim.set_seed(42)
   result2 = sim.run(10_000, backend="thread")

   assert np.allclose(result1.results, result2.results)  # Identical!

Each parallel worker receives an independent child sequence via :meth:`~numpy.random.SeedSequence.spawn`,
ensuring deterministic streams regardless of scheduling order.

For GPU backends, explicit :class:`~torch.Generator` objects are seeded from the same
:class:`~numpy.random.SeedSequence`, preserving reproducibility across CPU and GPU execution.

----

Functions
---------

.. autosummary::
   :toctree: generated
   :nosignatures:

   make_blocks

**Chunking Helper:**

.. code-block:: python

   from mcframework.core import make_blocks

   blocks = make_blocks(100_000, block_size=10_000)
   # [(0, 10000), (10000, 20000), ..., (90000, 100000)]

----

See Also
--------

- :doc:`backends` — Execution backends (sequential, parallel, GPU)
- :doc:`stats_engine` — Statistical metrics and confidence intervals
- :doc:`sims` — Built-in simulation implementations (Pi, Portfolio, Black-Scholes)
- :doc:`utils` — Critical value utilities
