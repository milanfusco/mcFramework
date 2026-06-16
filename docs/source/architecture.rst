============
Architecture
============

This page describes how ``mcframework`` is put together: its package
structure, layered design, the design patterns it uses, the UML models,
and the data flow through a simulation run.

.. contents:: Table of Contents
   :depth: 2
   :local:

System Overview
===============

System Context
--------------

.. mermaid:: diagrams/system_context.mmd

Package Structure
-----------------

::

   mcframework/
   ├── __init__.py          # Public API exports
   ├── core.py              # SimulationResult, MonteCarloFramework
   ├── simulation.py        # MonteCarloSimulation base class
   ├── stats_engine.py      # StatsEngine, StatsContext, ComputeResult, metrics
   ├── utils.py             # z_crit, t_crit, autocrit
   ├── profiling.py         # PyTorch profiler integration
   ├── backends/
   │   ├── __init__.py      # Backend exports (lazy Torch imports)
   │   ├── base.py          # ExecutionBackend protocol, utilities
   │   ├── sequential.py    # Single-threaded execution
   │   ├── parallel.py      # Thread and process backends
   │   ├── torch.py         # TorchBackend factory (auto-selects device)
   │   ├── torch_base.py    # Shared Torch and cuRAND utilities
   │   ├── torch_cpu.py     # PyTorch CPU backend
   │   ├── torch_mps.py     # Apple Silicon MPS backend
   │   └── torch_cuda.py    # NVIDIA CUDA backend
   └── sims/
       ├── __init__.py      # Simulation catalog
       ├── pi.py            # PiEstimationSimulation
       ├── portfolio.py     # PortfolioSimulation
       └── black_scholes.py # BlackScholesSimulation, BlackScholesPathSimulation

Architectural Design
====================

Layered Architecture
--------------------

.. mermaid:: diagrams/layered_architecture.mmd

Component Diagram
-----------------

.. mermaid:: diagrams/component_diagram.mmd

Process View (Parallel Execution)
---------------------------------

.. mermaid:: diagrams/process_view.mmd

Design Patterns
===============

Template Method
---------------

**Location:** ``MonteCarloSimulation.run()``

**Purpose:** Define the skeleton of the simulation algorithm, deferring the
per-simulation step to subclasses.

.. mermaid:: diagrams/template_method_pattern.mmd

Strategy
--------

**Location:** ``StatsEngine`` with the ``Metric`` protocol

**Purpose:** Define a family of metrics, encapsulate each one, and make them
interchangeable.

.. mermaid:: diagrams/strategy_pattern.mmd

Registry
--------

**Location:** ``MonteCarloFramework``

**Purpose:** Maintain a collection of named simulations for lookup and
comparison.

.. mermaid:: diagrams/registry_pattern.mmd

Adapter
-------

**Location:** ``FnMetric``

**Purpose:** Convert a plain function into an object implementing the
``Metric`` protocol.

.. mermaid:: diagrams/adapter_pattern.mmd

Pattern Summary
---------------

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Pattern
     - Location
     - Benefit
   * - **Template Method**
     - ``MonteCarloSimulation.run()``
     - Reuse execution logic, customize only the simulation step
   * - **Strategy**
     - ``StatsEngine`` + ``Metric``
     - Pluggable metrics without changing the engine
   * - **Registry**
     - ``MonteCarloFramework``
     - Named lookup and comparison
   * - **Builder**
     - ``StatsContext.with_overrides()``
     - Fluent configuration
   * - **Adapter**
     - ``FnMetric``
     - Convert functions to protocol objects

UML Diagrams
============

Class Diagram (Core Module)
---------------------------

.. mermaid:: diagrams/class_diagram_core.mmd

Class Diagram (Stats Engine Module)
-----------------------------------

.. mermaid:: diagrams/class_diagram_stats.mmd

Sequence: Running a Simulation
------------------------------

.. mermaid:: diagrams/sequence_simulation.mmd

Sequence: Bootstrap Confidence Interval
---------------------------------------

.. mermaid:: diagrams/sequence_bootstrap.mmd

State: Simulation Lifecycle
---------------------------

.. mermaid:: diagrams/state_lifecycle.mmd

Data Flow
=========

Simulation Execution Flow
-------------------------

.. mermaid:: diagrams/execution_flow.mmd

Statistics Computation Flow
---------------------------

.. mermaid:: diagrams/stats_flow.mmd

Interface Design
================

Public API
----------

.. code-block:: python

   # Core classes
   from mcframework import (
       MonteCarloSimulation,   # base class for custom simulations
       SimulationResult,       # result container
       MonteCarloFramework,    # registry and runner
   )

   # Statistics
   from mcframework import (
       StatsEngine,            # metric orchestrator
       StatsContext,           # configuration
       FnMetric,               # metric adapter
       DEFAULT_ENGINE,         # pre-built engine
   )

   # Utilities
   from mcframework import z_crit, t_crit, autocrit

   # Built-in simulations
   from mcframework import (
       PiEstimationSimulation,
       PortfolioSimulation,
       BlackScholesSimulation,
       BlackScholesPathSimulation,
   )

Usage Examples
--------------

A minimal custom simulation only needs ``single_simulation()``:

.. code-block:: python

   from mcframework import MonteCarloSimulation

   class DiceSimulation(MonteCarloSimulation):
       def single_simulation(self, _rng=None, n_dice=2):
           rng = self._rng(_rng, self.rng)
           return float(rng.integers(1, 7, size=n_dice).sum())

   sim = DiceSimulation(name="2d6")
   sim.set_seed(42)
   result = sim.run(10_000, parallel=True)
   print(result.mean)  # ~7.0

Using the framework registry:

.. code-block:: python

   from mcframework import MonteCarloFramework, PiEstimationSimulation

   fw = MonteCarloFramework()
   fw.register_simulation(PiEstimationSimulation())
   result = fw.run_simulation("Pi Estimation", 100_000, n_points=10_000)
   print(result.result_to_string())

Module Reference
================

.. list-table::
   :header-rows: 1
   :widths: 22 42 36

   * - Module
     - Classes and functions
     - Purpose
   * - ``core.py``
     - ``SimulationResult``, ``MonteCarloFramework``
     - Result container and simulation registry
   * - ``simulation.py``
     - ``MonteCarloSimulation``
     - Base class and execution orchestration
   * - ``stats_engine.py``
     - ``StatsEngine``, ``StatsContext``, ``ComputeResult``, ``FnMetric``, metric functions
     - Statistical analysis
   * - ``utils.py``
     - ``z_crit``, ``t_crit``, ``autocrit``
     - Critical values
   * - ``profiling.py``
     - PyTorch profiler integration
     - Performance profiling
   * - ``backends/``
     - ``SequentialBackend``, ``ThreadBackend``, ``ProcessBackend``, ``TorchBackend``
     - Execution strategies
   * - ``sims/pi.py``
     - ``PiEstimationSimulation``
     - Pi estimation
   * - ``sims/portfolio.py``
     - ``PortfolioSimulation``
     - GBM wealth dynamics
   * - ``sims/black_scholes.py``
     - ``BlackScholesSimulation``, ``BlackScholesPathSimulation``
     - Option pricing
