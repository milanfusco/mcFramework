Benchmark Module
================

.. currentmodule:: mcframework.benchmark

The benchmark module turns "how fast is each backend on my simulation?" into a single,
repeatable, plottable operation. Given a simulation, it times the same workload across
every available execution backend (sequential, thread, process, Torch CPU, and
Apple-Silicon MPS) and returns a structured, JSON-serializable report.

Overview
--------

Backends are described by :class:`BackendSpec`, so the benchmark matrix is *data*, not
code. :func:`default_backends` builds the device-aware default set (Torch / MPS / CUDA
backends are added only when their device is actually available), and :func:`run_suite`
sweeps the ``sizes`` × ``specs`` grid. The measurement core depends only on NumPy;
:func:`plot_benchmarks` imports :mod:`matplotlib` lazily, so plotting stays an optional
extra (``pip install mcframework[viz]``).

Adding a new accelerator later — NVIDIA CUDA, for instance — is a single extra
:class:`BackendSpec` plus a device gate in :func:`default_backends`; no new benchmark
script is required.

Usage
-----

.. code-block:: python

   from mcframework import PiEstimationSimulation, run_suite, default_backends
   from mcframework.benchmark import plot_benchmarks

   sim = PiEstimationSimulation()
   report = run_suite(sim, [1_000, 10_000, 100_000], default_backends())

   print(report.summary_table())     # execution time + speedup table (N/A-aware)
   fig = plot_benchmarks(report)      # four-panel performance figure
   data = report.to_dict()           # JSON-serializable artifact

The same workflow is available from the command line:

.. code-block:: bash

   mcframework-benchmark --quick --save backend_benchmark.png
   mcframework-benchmark --sizes 1000,10000,100000 --json report.json --no-show

Module Reference
----------------

.. automodule:: mcframework.benchmark
   :no-members:
   :no-undoc-members:
   :no-inherited-members:

Classes
~~~~~~~

.. autosummary::
   :toctree: generated
   :recursive:
   :nosignatures:

   BackendSpec
   BenchmarkResult
   BenchmarkReport

Functions
~~~~~~~~~

.. autosummary::
   :toctree: generated
   :recursive:
   :nosignatures:

   system_info
   default_backends
   benchmark_run
   run_suite
   plot_benchmarks
   main


See Also
--------

- :doc:`backends`: The execution backends being measured
- :doc:`../../guides/mps`: Apple-Silicon MPS backend details
- :doc:`../../guides/cuda`: NVIDIA CUDA backend (the next milestone)
- :mod:`mcframework.profiling`: Operator-level PyTorch profiling for kernel detail
