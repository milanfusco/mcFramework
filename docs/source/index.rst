.. mcframework documentation master file

================================
mcframework
================================

**Lightweight, reproducible, and deterministic Monte Carlo simulation framework**

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Deterministic & Reproducible
      
      Parallel execution with :class:`~numpy.random.SeedSequence` spawning ensures 
      identical results regardless of worker scheduling or system load.

   .. grid-item-card:: Parallel by Design
      
      Automatic backend selection (threads vs processes) optimized per platform.
      NumPy's GIL-releasing RNGs enable efficient thread-based parallelism.

   .. grid-item-card:: GPU Accelerated
      
      Optional `PyTorch <https://docs.pytorch.org/docs>`_ backends for massive speedups: Torch CPU,
      Apple `MPS <https://developer.apple.com/documentation/metalperformanceshaders>`_ (M1/M2/M3), and NVIDIA `CUDA <https://docs.nvidia.com/cuda/>`_ with adaptive batching.

   .. grid-item-card:: Rich Statistics Engine
      
      Built-in metrics: :meth:`~mcframework.stats_engine.mean`, :meth:`~mcframework.stats_engine.std`, :meth:`~mcframework.stats_engine.percentiles`, :meth:`~mcframework.stats_engine.skew`, :meth:`~mcframework.stats_engine.kurtosis`, :meth:`~mcframework.stats_engine.ci_mean`, and multiple 
      CI methods (:meth:`~mcframework.stats_engine.ci_mean`, :meth:`~mcframework.stats_engine.ci_mean_bootstrap`, :meth:`~mcframework.stats_engine.ci_mean_chebyshev`).

   .. grid-item-card:: Extensible Architecture
      
      Simple base class design—implement :meth:`~mcframework.core.MonteCarloSimulation.single_simulation` for scalar execution
      or :meth:`~mcframework.core.MonteCarloSimulation.torch_batch` for hardware-accelerated vectorized tensor operations. The framework
      handles execution, statistics, and result management.

   .. grid-item-card:: Multiple Execution Backends
      
      Choose from :class:`~mcframework.backends.SequentialBackend`, :class:`~mcframework.backends.ThreadBackend`, :class:`~mcframework.backends.ProcessBackend`, or :class:`~mcframework.backends.TorchBackend` backends. Each backend
      supports progress callbacks and respects reproducible seeding.

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
   result = fw.run_simulation("Pi Estimation", n_simulations=50_000, backend="thread")

   # Access results
   print(f"π ≈ {result.mean:.6f}")
   print(f"95% CI: [{result.stats['ci_mean'][0]:.6f}, {result.stats['ci_mean'][1]:.6f}]")

.. code-block:: text
   :caption: Output

   π ≈ 3.141592
   95% CI: [3.140821, 3.142363]

**GPU-Accelerated Example (17,000x faster):**

.. code-block:: python

   # Same simulation, GPU backend
   result = sim.run(1_000_000, backend="torch", torch_device="mps")  # Apple Silicon
   # result = sim.run(1_000_000, backend="torch", torch_device="cuda")  # NVIDIA GPU
   print(f"π ≈ {result.mean:.6f} (computed in {result.execution_time:.3f}s)")


Installation
------------

**From PyPI** (recommended)

.. code-block:: bash

   pip install mcframework

**From Source (Development)**

.. code-block:: bash

   git clone https://github.com/milanfusco/mcframework.git
   cd mcframework
   pip install -e .

**With Optional Dependencies**

.. code-block:: bash

   # GPU acceleration (PyTorch)
   pip install mcframework[gpu]

   # All dependencies (dev, test, docs, gui, gpu)
   pip install -e ".[dev,test,docs,gui,gpu]"

   # Just testing
   pip install -e ".[test]"

   # GUI application (PySide6)
   pip install -e ".[gui]"


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


Execution Backends
------------------

The ``backend`` parameter controls execution strategy:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Backend
     - Description
   * - ``"auto"``
     - Sequential for small jobs, parallel (thread/process) for large jobs
   * - ``"sequential"``
     - Single-threaded execution
   * - ``"thread"``
     - Thread-based parallelism (best when NumPy releases GIL)
   * - ``"process"``
     - Process-based parallelism (required on Windows for true parallelism)
   * - ``"torch"``
     - GPU-accelerated batch execution (requires ``supports_batch = True``)

.. code-block:: python

   sim = PiEstimationSimulation()
   sim.set_seed(42)

   # CPU backends
   result = sim.run(100_000, backend="thread", n_workers=8)
   result = sim.run(100_000, backend="process", n_workers=4)

   # GPU backends (requires pip install mcframework[gpu])
   result = sim.run(1_000_000, backend="torch", torch_device="cpu")   # Vectorized CPU
   result = sim.run(1_000_000, backend="torch", torch_device="mps")   # Apple Silicon
   result = sim.run(1_000_000, backend="torch", torch_device="cuda")  # NVIDIA GPU


.. toctree::
   :maxdepth: 2
   :caption: Guides

   getting-started

.. toctree::
   :maxdepth: 2
   :caption: Project Documentation

   ../PROJECT_PLAN
   ../SYSTEM_DESIGN

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/_modules/core
   api/_modules/backends
   api/_modules/stats_engine
   api/_modules/sims
   api/_modules/utils


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
