=========================
System Design
=========================

System and Program Design
=========================

----

.. list-table::
   :header-rows: 1
   :widths: 30  50
   :align: center

   * - Name
     - Email
   * - Milan Fusco
     - mdfusco@student.ysu.edu
.. note::

   For requirements, stakeholders, and project plan, see :doc:`PROJECT_PLAN`

----

.. contents:: Table of Contents
   :depth: 2
   :local:

----

1. System Overview
------------------

1.1 System Context Diagram
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. mermaid:: diagrams/system_context.mmd

1.2 Package Structure
~~~~~~~~~~~~~~~~~~~~~

::

   mcframework/
   ├── __init__.py          # Public API exports
   ├── core.py              # MonteCarloSimulation, SimulationResult, MonteCarloFramework
   ├── stats_engine.py      # StatsEngine, StatsContext, ComputeResult, metrics
   ├── utils.py             # z_crit, t_crit, autocrit
   └── sims/
       ├── __init__.py      # Simulation catalog
       ├── pi.py            # PiEstimationSimulation
       ├── portfolio.py     # PortfolioSimulation
       └── black_scholes.py # BlackScholesSimulation, BlackScholesPathSimulation

----

2. Architectural Design
-----------------------

2.1 Layered Architecture
~~~~~~~~~~~~~~~~~~~~~~~~

.. mermaid:: diagrams/layered_architecture.mmd

2.2 Component Diagram
~~~~~~~~~~~~~~~~~~~~~

.. mermaid:: diagrams/component_diagram.mmd

2.3 Process View (Parallel Execution)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. mermaid:: diagrams/process_view.mmd

----

3. Design Patterns
------------------

3.1 Template Method Pattern
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``MonteCarloSimulation.run()``

**Purpose:** Define the skeleton of the simulation algorithm, deferring specific steps to subclasses.

.. mermaid:: diagrams/template_method_pattern.mmd

3.2 Strategy Pattern
~~~~~~~~~~~~~~~~~~~~

**Location:** ``StatsEngine`` with ``Metric`` protocol

**Purpose:** Define a family of algorithms (metrics), encapsulate each one, and make them interchangeable.

.. mermaid:: diagrams/strategy_pattern.mmd

3.3 Registry Pattern
~~~~~~~~~~~~~~~~~~~~

**Location:** ``MonteCarloFramework``

**Purpose:** Maintain a collection of named simulations for lookup and comparison.

.. mermaid:: diagrams/registry_pattern.mmd

3.4 Adapter Pattern
~~~~~~~~~~~~~~~~~~~

**Location:** ``FnMetric``

**Purpose:** Convert a plain function into an object implementing the ``Metric`` protocol.

.. mermaid:: diagrams/adapter_pattern.mmd

3.5 Pattern Summary
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Pattern
     - Location
     - Benefit
   * - **Template Method**
     - ``MonteCarloSimulation.run()``
     - Reuse execution logic, customize only simulation
   * - **Strategy**
     - ``StatsEngine`` + ``Metric``
     - Pluggable metrics without changing engine
   * - **Registry**
     - ``MonteCarloFramework``
     - Named lookup and comparison
   * - **Builder**
     - ``StatsContext.with_overrides()``
     - Fluent configuration
   * - **Adapter**
     - ``FnMetric``
     - Convert functions to protocol objects

----

4. UML Diagrams
---------------

4.1 Class Diagram (Core Module)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. mermaid:: diagrams/class_diagram_core.mmd

4.2 Class Diagram (Stats Engine Module)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. mermaid:: diagrams/class_diagram_stats.mmd

4.3 Sequence Diagram: Running a Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. mermaid:: diagrams/sequence_simulation.mmd

4.4 Sequence Diagram: Bootstrap Confidence Interval
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. mermaid:: diagrams/sequence_bootstrap.mmd

4.5 State Diagram: Simulation Lifecycle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. mermaid:: diagrams/state_lifecycle.mmd

----

5. Data Flow
------------

5.1 Simulation Execution Flow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. mermaid:: diagrams/execution_flow.mmd

5.2 Statistics Computation Flow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. mermaid:: diagrams/stats_flow.mmd

----

6. Interface Design
-------------------

6.1 Public API
~~~~~~~~~~~~~~

.. code-block:: python

   # Core classes
   from mcframework import (
       MonteCarloSimulation,  # ABC for custom simulations
       SimulationResult,       # Result container
       MonteCarloFramework,    # Registry and runner
   )

   # Statistics
   from mcframework import (
       StatsEngine,            # Metric orchestrator
       StatsContext,           # Configuration
       FnMetric,               # Metric adapter
       DEFAULT_ENGINE,         # Pre-built engine
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

6.2 Usage Examples
~~~~~~~~~~~~~~~~~~

**Minimal Custom Simulation:**

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

**Using the Framework:**

.. code-block:: python

   from mcframework import MonteCarloFramework, PiEstimationSimulation

   fw = MonteCarloFramework()
   fw.register_simulation(PiEstimationSimulation())
   result = fw.run_simulation("Pi Estimation", 100_000, n_points=10_000)
   print(result.result_to_string())

----

Appendix: Module Reference
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Module
     - Classes/Functions
     - Purpose
   * - ``core.py``
     - ``MonteCarloSimulation``, ``SimulationResult``, ``MonteCarloFramework``
     - Simulation execution
   * - ``stats_engine.py``
     - ``StatsEngine``, ``StatsContext``, ``ComputeResult``, ``FnMetric``, 12+ metric functions
     - Statistical analysis
   * - ``utils.py``
     - ``z_crit``, ``t_crit``, ``autocrit``
     - Critical values
   * - ``sims/pi.py``
     - ``PiEstimationSimulation``
     - π estimation
   * - ``sims/portfolio.py``
     - ``PortfolioSimulation``
     - GBM wealth
   * - ``sims/black_scholes.py``
     - ``BlackScholesSimulation``, ``BlackScholesPathSimulation``
     - Option pricing
