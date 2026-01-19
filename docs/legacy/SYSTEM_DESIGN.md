# System Design

## System and Program Design

### McFramework — Monte Carlo Simulation Framework

> *For requirements, stakeholders, and project plan, see [PROJECT_PLAN.md](PROJECT_PLAN.md)*

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architectural Design](#2-architectural-design)
3. [Design Patterns](#3-design-patterns)
4. [UML Diagrams](#4-uml-diagrams)
5. [Data Flow](#5-data-flow)
6. [Interface Design](#6-interface-design)

---

## 1. System Overview

### 1.1 System Context Diagram

```mermaid
flowchart TB
    subgraph External["External Dependencies"]
        numpy["NumPy<br/>(RNG, Arrays)"]
        scipy["SciPy<br/>(Statistics)"]
        python["Python 3.10+<br/>(Runtime)"]
        concurrent["concurrent.futures<br/>(Parallelism)"]
    end

    subgraph McFramework["McFramework Package"]
        core["core.py<br/>MonteCarloSimulation<br/>SimulationResult<br/>MonteCarloFramework"]
        stats["stats_engine.py<br/>StatsEngine<br/>StatsContext<br/>Metrics"]
        utils["utils.py<br/>z_crit, t_crit<br/>autocrit"]
        sims["sims/<br/>Pi, Portfolio<br/>Black-Scholes"]
    end

    subgraph Users["End Users"]
        researchers["Researchers"]
        students["Students"]
        quants["Quantitative<br/>Analysts"]
    end

    numpy --> core
    scipy --> stats
    python --> McFramework
    concurrent --> core

    core --> stats
    stats --> utils
    sims --> core

    McFramework --> Users
```

### 1.2 Package Structure

```
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
```

---

## 2. Architectural Design

### 2.1 Layered Architecture

```mermaid
flowchart TB
    subgraph Layer5["Application Layer"]
        user["User-defined simulations<br/>Custom scripts"]
    end

    subgraph Layer4["Framework Layer"]
        framework["MonteCarloFramework<br/>(registry, comparison, orchestration)"]
    end

    subgraph Layer3["Core Services Layer"]
        sim["MonteCarloSimulation<br/>(execution engine)"]
        engine["StatsEngine<br/>(statistical analysis)"]
    end

    subgraph Layer2["Utilities Layer"]
        utils2["utils.py<br/>(critical values, helpers)"]
    end

    subgraph Layer1["Infrastructure Layer"]
        infra["NumPy | SciPy | concurrent.futures | dataclasses"]
    end

    Layer5 --> Layer4
    Layer4 --> Layer3
    Layer3 --> Layer2
    Layer2 --> Layer1
```

### 2.2 Component Diagram

```mermaid
flowchart TB
    subgraph mcframework["mcframework package"]
        subgraph core_comp["<<component>> core"]
            MCS["MonteCarloSimulation<br/><<abstract>>"]
            SR["SimulationResult<br/><<dataclass>>"]
            MCF["MonteCarloFramework"]
        end

        subgraph stats_comp["<<component>> stats_engine"]
            SE["StatsEngine"]
            SC["StatsContext<br/><<dataclass>>"]
            CR["ComputeResult<br/><<dataclass>>"]
            FM["FnMetric<<generic>>"]
            MP["Metric<<protocol>>"]
        end

        subgraph utils_comp["<<component>> utils"]
            zcrit["z_crit()"]
            tcrit["t_crit()"]
            auto["autocrit()"]
        end

        subgraph sims_comp["<<component>> sims"]
            pi["PiEstimationSimulation"]
            port["PortfolioSimulation"]
            bs["BlackScholesSimulation"]
        end
    end

    core_comp -->|uses| stats_comp
    core_comp -->|uses| utils_comp
    stats_comp -->|uses| utils_comp
    sims_comp -->|extends| core_comp
```

### 2.3 Process View (Parallel Execution)

```mermaid
flowchart TB
    subgraph Main["Main Process"]
        run["simulation.run()"]
        validate["_validate_run_params()"]
        seed["spawn SeedSequences"]
        
        subgraph Execution["Execution Path"]
            seq["_run_sequential()"]
            par["_run_parallel()"]
        end
        
        subgraph Workers["Worker Pool (parallel=True)"]
            w1["Worker 1<br/>SeedSeq[0]<br/>Philox RNG"]
            w2["Worker 2<br/>SeedSeq[1]<br/>Philox RNG"]
            w3["Worker 3<br/>SeedSeq[2]<br/>Philox RNG"]
        end
        
        collect["Collect Results"]
        stats["StatsEngine.compute()"]
        result["SimulationResult"]
    end

    run --> validate --> seed
    seed --> seq
    seed --> par
    par --> w1 & w2 & w3
    w1 & w2 & w3 --> collect
    seq --> collect
    collect --> stats --> result
```

---

## 3. Design Patterns

### 3.1 Template Method Pattern

**Location:** `MonteCarloSimulation.run()`

**Purpose:** Define the skeleton of the simulation algorithm, deferring specific steps to subclasses.

```mermaid
classDiagram
    class MonteCarloSimulation {
        <<abstract>>
        +run(n_simulations, ...) SimulationResult
        -_validate_run_params()
        -_run_sequential()
        -_run_parallel()
        -_compute_stats_with_engine()
        -_create_result()
        +single_simulation()* float
    }

    class PiEstimationSimulation {
        +single_simulation(n_points, antithetic)
    }

    class PortfolioSimulation {
        +single_simulation(initial_value, annual_return, ...)
    }

    class BlackScholesSimulation {
        +single_simulation(S0, K, T, r, sigma, ...)
        +calculate_greeks()
    }

    MonteCarloSimulation <|-- PiEstimationSimulation
    MonteCarloSimulation <|-- PortfolioSimulation
    MonteCarloSimulation <|-- BlackScholesSimulation

    note for MonteCarloSimulation "Template Method:\nrun() defines algorithm skeleton\nsingle_simulation() is the hook"
```

### 3.2 Strategy Pattern

**Location:** `StatsEngine` with `Metric` protocol

**Purpose:** Define a family of algorithms (metrics), encapsulate each one, and make them interchangeable.

```mermaid
classDiagram
    class StatsEngine {
        -_metrics: list~Metric~
        +compute(x, ctx) ComputeResult
    }

    class Metric {
        <<protocol>>
        +name: str
        +__call__(x, ctx) Any
    }

    class FnMetric~T~ {
        +name: str
        +fn: Callable
        +__call__(x, ctx) T
    }

    StatsEngine o-- Metric : contains
    Metric <|.. FnMetric : implements

    note for StatsEngine "Context: holds strategy list"
    note for Metric "Strategy Interface"
```

### 3.3 Registry Pattern

**Location:** `MonteCarloFramework`

**Purpose:** Maintain a collection of named simulations for lookup and comparison.

```mermaid
classDiagram
    class MonteCarloFramework {
        -simulations: dict~str, MonteCarloSimulation~
        -results: dict~str, SimulationResult~
        +register_simulation(sim, name)
        +run_simulation(name, n_simulations, ...)
        +compare_results(names, metric)
    }

    class MonteCarloSimulation {
        <<abstract>>
        +name: str
    }

    class SimulationResult {
        <<dataclass>>
        +mean: float
        +std: float
    }

    MonteCarloFramework o-- "0..*" MonteCarloSimulation : manages
    MonteCarloFramework o-- "0..*" SimulationResult : stores

    note for MonteCarloFramework "Registry Pattern:\nNamed lookup & comparison"
```

### 3.4 Adapter Pattern

**Location:** `FnMetric`

**Purpose:** Convert a plain function into an object implementing the `Metric` protocol.

```mermaid
flowchart LR
    subgraph Before["Plain Function"]
        fn["def mean(x, ctx) → float"]
    end

    subgraph Adapter["FnMetric (Adapter)"]
        wrap["FnMetric('mean', mean)"]
    end

    subgraph After["Protocol Object"]
        metric["Metric protocol<br/>.name: str<br/>.__call__(x, ctx)"]
    end

    fn --> wrap --> metric
```

### 3.5 Pattern Summary

| Pattern | Location | Benefit |
|---------|----------|---------|
| **Template Method** | `MonteCarloSimulation.run()` | Reuse execution logic, customize only simulation |
| **Strategy** | `StatsEngine` + `Metric` | Pluggable metrics without changing engine |
| **Registry** | `MonteCarloFramework` | Named lookup and comparison |
| **Builder** | `StatsContext.with_overrides()` | Fluent configuration |
| **Adapter** | `FnMetric` | Convert functions to protocol objects |

---

## 4. UML Diagrams

### 4.1 Class Diagram (Core Module)

```mermaid
classDiagram
    class MonteCarloSimulation {
        <<abstract>>
        +name: str
        +seed_seq: SeedSequence
        +rng: Generator
        +parallel_backend: str
        -_PCTS: tuple
        -_PARALLEL_THRESHOLD: int
        +__init__(name: str)
        +single_simulation(**kwargs)* float
        +set_seed(seed: int)
        +run(n_simulations, ...) SimulationResult
        -_run_sequential(...)
        -_run_parallel(...)
        -_create_result(...)
    }

    class SimulationResult {
        <<dataclass>>
        +results: ndarray
        +n_simulations: int
        +execution_time: float
        +mean: float
        +std: float
        +percentiles: dict
        +stats: dict
        +metadata: dict
        +result_to_string() str
    }

    class MonteCarloFramework {
        +simulations: dict
        +results: dict
        +register_simulation(simulation, name)
        +run_simulation(name, n_simulations, ...) SimulationResult
        +compare_results(names, metric) dict
    }

    class PiEstimationSimulation {
        +single_simulation(n_points, antithetic, _rng) float
    }

    class PortfolioSimulation {
        +single_simulation(initial_value, annual_return, volatility, years, _rng) float
    }

    class BlackScholesSimulation {
        +single_simulation(S0, K, T, r, sigma, option_type, exercise_type, _rng) float
        +calculate_greeks(n_simulations, ...) dict
    }

    MonteCarloSimulation <|-- PiEstimationSimulation
    MonteCarloSimulation <|-- PortfolioSimulation
    MonteCarloSimulation <|-- BlackScholesSimulation

    MonteCarloSimulation --> SimulationResult : creates
    MonteCarloFramework o-- MonteCarloSimulation : manages
    MonteCarloFramework o-- SimulationResult : stores
```

### 4.2 Class Diagram (Stats Engine Module)

```mermaid
classDiagram
    class StatsContext {
        <<dataclass>>
        +n: int
        +confidence: float
        +ci_method: CIMethod
        +percentiles: tuple
        +nan_policy: NanPolicy
        +target: float
        +eps: float
        +ddof: int
        +ess: int
        +n_bootstrap: int
        +bootstrap: BootstrapMethod
        +alpha() float
        +eff_n() int
        +with_overrides(**changes) StatsContext
    }

    class CIMethod {
        <<enumeration>>
        auto
        z
        t
        bootstrap
    }

    class NanPolicy {
        <<enumeration>>
        propagate
        omit
    }

    class BootstrapMethod {
        <<enumeration>>
        percentile
        bca
    }

    class StatsEngine {
        -_metrics: list~Metric~
        +compute(x, ctx, select) ComputeResult
    }

    class ComputeResult {
        <<dataclass>>
        +metrics: dict
        +skipped: list
        +errors: list
        +successful_metrics() set
    }

    class Metric {
        <<protocol>>
        +name: str
        +__call__(x, ctx) Any
    }

    class FnMetric~T~ {
        <<dataclass>>
        +name: str
        +fn: Callable
        +__call__(x, ctx) T
    }

    StatsContext --> CIMethod
    StatsContext --> NanPolicy
    StatsContext --> BootstrapMethod

    StatsEngine o-- Metric : contains
    StatsEngine --> ComputeResult : returns
    Metric <|.. FnMetric : implements
```

### 4.3 Sequence Diagram: Running a Simulation

```mermaid
sequenceDiagram
    participant User
    participant Sim as MonteCarloSimulation
    participant Pool as ThreadPool/ProcessPool
    participant Engine as StatsEngine
    participant Result as SimulationResult

    User->>Sim: run(n=10000, parallel=True)
    activate Sim

    Sim->>Sim: _validate_run_params()
    Sim->>Sim: spawn SeedSequences

    Note over Sim,Pool: Parallel Execution
    Sim->>Pool: submit chunks to workers
    activate Pool

    par Worker 1
        Pool->>Pool: single_simulation() × chunk_size
    and Worker 2
        Pool->>Pool: single_simulation() × chunk_size
    and Worker 3
        Pool->>Pool: single_simulation() × chunk_size
    end

    Pool-->>Sim: collected results[]
    deactivate Pool

    Sim->>Engine: compute(results, StatsContext)
    activate Engine
    Engine->>Engine: mean(), std(), ci_mean(), ...
    Engine-->>Sim: ComputeResult
    deactivate Engine

    Sim->>Result: _create_result()
    Result-->>Sim: SimulationResult
    Sim-->>User: SimulationResult
    deactivate Sim
```

### 4.4 Sequence Diagram: Bootstrap Confidence Interval

```mermaid
sequenceDiagram
    participant Engine as StatsEngine
    participant Bootstrap as ci_mean_bootstrap
    participant Means as _bootstrap_means
    participant BCa as _compute_bca_interval

    Engine->>Bootstrap: ci_mean_bootstrap(x, ctx)
    activate Bootstrap

    Bootstrap->>Bootstrap: _clean(x, ctx)
    
    Bootstrap->>Means: _bootstrap_means(arr, n_resamples, rng)
    activate Means
    
    loop n_bootstrap times
        Means->>Means: resample with replacement
        Means->>Means: compute mean
    end
    
    Means-->>Bootstrap: bootstrap_means[]
    deactivate Means

    alt method == "percentile"
        Bootstrap->>Bootstrap: np.percentile(means, [α/2, 1-α/2])
    else method == "bca"
        Bootstrap->>BCa: _compute_bca_interval(arr, means, confidence)
        activate BCa
        BCa->>BCa: bias_correction + acceleration
        BCa-->>Bootstrap: (low, high)
        deactivate BCa
    end

    Bootstrap-->>Engine: _CIResult.as_dict()
    deactivate Bootstrap
```

### 4.5 State Diagram: Simulation Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Initialized: __init__()

    Initialized --> Seeded: set_seed(seed)
    
    Seeded --> Running: run(n_simulations)
    Initialized --> Running: run(n_simulations)
    
    Running --> Computing: execution complete
    Running --> Error: exception raised
    
    Computing --> Completed: stats computed
    
    Error --> Initialized: create new instance
    
    Completed --> Running: run() again
    Completed --> [*]: return SimulationResult
```

---

## 5. Data Flow

### 5.1 Simulation Execution Flow

```mermaid
flowchart LR
    subgraph Input
        n["n_simulations"]
        seed["seed"]
        kwargs["**kwargs"]
    end

    subgraph Processing["MonteCarloSimulation.run()"]
        exec["Parallel/Sequential<br/>Execution"]
        single["single_simulation()<br/>× n_simulations"]
    end

    subgraph Stats["Statistics"]
        results["results: ndarray"]
        engine["StatsEngine.compute()"]
    end

    subgraph Output
        sr["SimulationResult"]
    end

    n & seed & kwargs --> exec
    exec --> single --> results --> engine --> sr
```

### 5.2 Statistics Computation Flow

```mermaid
flowchart TB
    Input["Input: ndarray x"] --> Clean["_clean(x, ctx)<br/>Handle NaN/Inf"]
    
    Clean --> Descriptive & CI & Bounds
    
    subgraph Descriptive["Descriptive Stats"]
        mean_f["mean()"]
        std_f["std()"]
        pct_f["percentiles()"]
        skew_f["skew()"]
        kurt_f["kurtosis()"]
    end
    
    subgraph CI["Confidence Intervals"]
        ci_mean_f["ci_mean()"]
        ci_boot_f["ci_mean_bootstrap()"]
        ci_cheb_f["ci_mean_chebyshev()"]
    end
    
    subgraph Bounds["Distribution-Free"]
        cheb_n["chebyshev_required_n()"]
        markov["markov_error_prob()"]
    end
    
    Descriptive & CI & Bounds --> Result["ComputeResult"]
```

---

## 6. Interface Design

### 6.1 Public API

```python
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
```

### 6.2 Usage Examples

**Minimal Custom Simulation:**

```python
from mcframework import MonteCarloSimulation

class DiceSimulation(MonteCarloSimulation):
    def single_simulation(self, _rng=None, n_dice=2):
        rng = self._rng(_rng, self.rng)
        return float(rng.integers(1, 7, size=n_dice).sum())

sim = DiceSimulation(name="2d6")
sim.set_seed(42)
result = sim.run(10_000, parallel=True)
print(result.mean)  # ~7.0
```

**Using the Framework:**

```python
from mcframework import MonteCarloFramework, PiEstimationSimulation

fw = MonteCarloFramework()
fw.register_simulation(PiEstimationSimulation())
result = fw.run_simulation("Pi Estimation", 100_000, n_points=10_000)
print(result.result_to_string())
```

---

## Appendix: Module Reference

| Module | Classes/Functions | Purpose |
|--------|-------------------|---------|
| `core.py` | `MonteCarloSimulation`, `SimulationResult`, `MonteCarloFramework` | Simulation execution |
| `stats_engine.py` | `StatsEngine`, `StatsContext`, `ComputeResult`, `FnMetric`, 12+ metric functions | Statistical analysis |
| `utils.py` | `z_crit`, `t_crit`, `autocrit` | Critical values |
| `sims/pi.py` | `PiEstimationSimulation` | π estimation |
| `sims/portfolio.py` | `PortfolioSimulation` | GBM wealth |
| `sims/black_scholes.py` | `BlackScholesSimulation`, `BlackScholesPathSimulation` | Option pricing |
