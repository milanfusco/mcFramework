# Project Plan

## Project Selection and Project Plan

### McFramework — Monte Carlo Simulation Framework

---

## 1) Team Members

| Name | Section | Email |
|:----:|:-------:|:-----:|
| Milan Fusco | 11:00 AM | mdfusco@student.ysu.edu |
| James Gabbert | 11:00 AM | jdgabbert@student.ysu.edu |

---

## 2) Project Description

McFramework is a Python library providing a robust, extensible foundation for building and running Monte Carlo simulations with rigorous statistical analysis.

**Key Capabilities:**

- Abstract base class pattern for defining custom simulations
- Deterministic parallel execution with reproducible RNG streams
- Comprehensive statistics engine with 12+ metric functions
- Multiple confidence interval methods (parametric, bootstrap, distribution-free)
- Built-in simulations: Pi estimation, Portfolio wealth, Black-Scholes options

> *For detailed architecture and UML diagrams, see [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md)*

---

## 3) Software System Type

**☑ System for Modeling and Simulation**

The framework is designed for computational experiments involving:

- Stochastic process simulation (random sampling, GBM)
- Statistical estimation (Monte Carlo integration)
- Financial modeling (option pricing, portfolio analysis)
- Uncertainty quantification (confidence intervals, error bounds)

**Architecture Classification:**

- **Library/Framework** — Provides reusable abstractions for simulation development
- **Batch Processing** — Executes thousands of independent simulation runs
- **Parallel System** — Distributes work across threads or processes

---

## 4) Project Plan

### Phase 1: Core Abstractions (Weeks 1-2)

- [x] Design `MonteCarloSimulation` abstract base class
- [x] Implement `SimulationResult` dataclass for outputs
- [x] Create `MonteCarloFramework` registry for managing multiple simulations
- [x] Define public API in `__init__.py`

### Phase 2: RNG & Parallel Execution (Weeks 3-4)

- [x] Implement reproducible seeding with `numpy.random.SeedSequence`
- [x] Build parallel execution with `ThreadPoolExecutor` (POSIX)
- [x] Build parallel execution with `ProcessPoolExecutor` (Windows)
- [x] Implement platform-aware backend auto-selection
- [x] Chunk-based work distribution for load balancing

### Phase 3: Statistics Engine (Weeks 5-7)

- [x] Design `StatsContext` configuration dataclass
- [x] Implement descriptive statistics: `mean`, `std`, `percentiles`, `skew`, `kurtosis`
- [x] Implement parametric CI: `ci_mean` with z/t critical values
- [x] Implement bootstrap CI: `ci_mean_bootstrap` with percentile and BCa methods
- [x] Implement distribution-free bounds: `ci_mean_chebyshev`, `markov_error_prob`
- [x] Build `StatsEngine` orchestrator with `ComputeResult` tracking

### Phase 4: Built-in Simulations (Weeks 8-9)

- [x] `PiEstimationSimulation` — geometric probability on unit disk
- [x] `PortfolioSimulation` — GBM wealth dynamics
- [x] `BlackScholesSimulation` — European/American option pricing
- [x] `BlackScholesPathSimulation` — path generation for visualization

### Phase 5: Testing & Quality (Weeks 10-11)

- [x] Unit tests for all modules (16 test files)
- [x] Edge case and error handling tests
- [x] Integration tests for full simulation runs
- [x] CI pipeline: lint, test, coverage, docs

### Phase 6: Documentation (Week 12)

- [x] NumPy-style docstrings with LaTeX math
- [x] Sphinx API documentation
- [x] UML class diagrams

---

## 5) Requirements

### User Requirements

| ID | Requirement |
|----|-------------|
| UR-1 | Users shall define custom simulations by subclassing `MonteCarloSimulation` and implementing `single_simulation()` |
| UR-2 | Users shall obtain reproducible results by setting a seed via `set_seed()` |
| UR-3 | Users shall run simulations in parallel by passing `parallel=True` to `run()` |
| UR-4 | Users shall receive statistical summaries including mean, std, percentiles, and confidence intervals |
| UR-5 | Users shall configure statistical behavior via `StatsContext` parameters |
| UR-6 | Users shall register and compare multiple simulations using `MonteCarloFramework` |
| UR-7 | Users shall extend the statistics engine by implementing the `Metric` protocol |

### System Requirements

| ID | Requirement |
|----|-------------|
| SR-1 | The system shall require Python ≥ 3.10, NumPy ≥ 1.24, SciPy ≥ 1.10 |
| SR-2 | The system shall use `SeedSequence.spawn()` to create independent RNG streams per worker |
| SR-3 | The system shall use `numpy.random.Philox` as the bit generator for parallel reproducibility |
| SR-4 | The system shall select `ThreadPoolExecutor` on POSIX and `ProcessPoolExecutor` on Windows by default |
| SR-5 | The system shall fall back to sequential execution for n < 20,000 simulations |
| SR-6 | The system shall compute confidence intervals using `scipy.stats.norm` and `scipy.stats.t` |
| SR-7 | The system shall support bootstrap resampling with configurable `n_bootstrap` (default 10,000) |
| SR-8 | The system shall implement BCa bootstrap using jackknife acceleration |
| SR-9 | The system shall handle NaN values according to `nan_policy` ("propagate" or "omit") |

---

## 6) Stakeholders

| Stakeholder Type | Role | Primary Concerns |
|------------------|------|------------------|
| **Simulation Developers** | Create custom `MonteCarloSimulation` subclasses | Clean API, extensibility, documentation |
| **Researchers** | Run experiments, analyze results | Reproducibility, statistical rigor, accuracy |
| **Students** | Learn Monte Carlo methods | Simple examples, clear explanations |
| **Library Maintainers** | Maintain codebase | Test coverage, code quality, CI/CD |
| **Framework Integrators** | Embed in larger systems | Modular design, minimal dependencies |

---

## 7) Development Methodology

**☑ Agile with XP (Extreme Programming) Practices**

| Practice | Evidence in Project |
|-------------|---------------------|
| **Test-Driven Development** | 16 test files covering unit, integration, edge cases, regression |
| **Git Workflow** | GitFlow with feature branches and pull requests |
| **Continuous Integration** | GitHub Actions: lint (ruff, pylint) → test (pytest) → build → docs → deploy |
| **Refactoring** | Numerical robustness improvements (NaN handling, edge cases) |
| **Small Releases** | Semantic versioning: v0.1.0 → v0.2.0 → v0.3.x → v0.4.0 → v0.5.0 |
| **Coding Standards** | PEP 8, PEP 585 type hints, NumPy docstring convention |
| **Collective Code Ownership** | Modular architecture (core, stats_engine, sims, utils) with clear separation of concerns |

---

## 8) Functional Requirements

| ID | Requirement | Module |
|----|-------------|--------|
| FR-1 | Provide abstract `MonteCarloSimulation` class with `single_simulation()` method | `core.py` |
| FR-2 | Execute simulations sequentially via `_run_sequential()` | `core.py` |
| FR-3 | Execute simulations in parallel via `_run_parallel()` with thread/process backends | `core.py` |
| FR-4 | Spawn independent RNG streams per worker chunk using `SeedSequence` | `core.py` |
| FR-5 | Return results in `SimulationResult` dataclass with mean, std, percentiles, stats, metadata | `core.py` |
| FR-6 | Register simulations by name in `MonteCarloFramework` | `core.py` |
| FR-7 | Compare metrics across simulations via `compare_results()` | `core.py` |
| FR-8 | Compute sample mean with NaN handling | `stats_engine.py` |
| FR-9 | Compute sample standard deviation with configurable ddof | `stats_engine.py` |
| FR-10 | Compute arbitrary percentiles via `numpy.percentile` | `stats_engine.py` |
| FR-11 | Compute skewness and kurtosis using `scipy.stats` | `stats_engine.py` |
| FR-12 | Compute parametric CI using z or t critical values | `stats_engine.py` |
| FR-13 | Compute bootstrap CI using percentile or BCa method | `stats_engine.py` |
| FR-14 | Compute Chebyshev distribution-free CI | `stats_engine.py` |
| FR-15 | Compute required n for target Chebyshev half-width | `stats_engine.py` |
| FR-16 | Compute Markov error probability bound | `stats_engine.py` |
| FR-17 | Track skipped and errored metrics in `ComputeResult` | `stats_engine.py` |
| FR-18 | Provide z-critical and t-critical value functions | `utils.py` |
| FR-19 | Auto-select z/t based on sample size threshold (n ≥ 30) | `utils.py` |
| FR-20 | Estimate π via geometric probability sampling | `sims/pi.py` |
| FR-21 | Simulate portfolio wealth under GBM dynamics | `sims/portfolio.py` |
| FR-22 | Price European options with discounted payoff | `sims/black_scholes.py` |
| FR-23 | Price American options using Longstaff-Schwartz LSM | `sims/black_scholes.py` |
| FR-24 | Calculate Greeks via finite difference bumping | `sims/black_scholes.py` |

---

## 9) Non-Functional Requirements

| ID | Category | Requirement |
|----|----------|-------------|
| NFR-1 | **Performance** | Parallel execution shall achieve near-linear speedup up to available CPU cores |
| NFR-2 | **Performance** | Sequential fallback for n < `_PARALLEL_THRESHOLD` (20,000) to avoid overhead |
| NFR-3 | **Reliability** | Identical results given the same seed, regardless of parallelism or platform |
| NFR-4 | **Reliability** | Graceful handling of empty arrays, NaN, and infinite values |
| NFR-5 | **Reliability** | `ComputeResult` tracks errors without crashing the engine |
| NFR-6 | **Portability** | Support Python 3.10, 3.11, 3.12 on Linux, macOS, Windows |
| NFR-7 | **Portability** | Auto-detect platform for thread vs. process backend |
| NFR-8 | **Maintainability** | Test coverage ≥ 90% |
| NFR-9 | **Maintainability** | All public functions have type hints and docstrings |
| NFR-10 | **Maintainability** | Modular design: core, stats_engine, sims, utils as separate concerns |
| NFR-11 | **Extensibility** | Custom simulations via subclassing `MonteCarloSimulation` |
| NFR-12 | **Extensibility** | Custom metrics via `Metric` protocol and `FnMetric` adapter |
| NFR-13 | **Documentation** | Sphinx-generated API docs with mathematical notation |

---

## 10) Usability Requirements

| ID | Requirement |
|----|-------------|
| USA-1 | A minimal simulation shall require implementing only `single_simulation()` (< 10 LOC) |
| USA-2 | The `run()` method shall use sensible defaults: `parallel=False`, `confidence=0.95`, `ci_method="auto"` |
| USA-3 | `SimulationResult.result_to_string()` shall produce human-readable summaries |
| USA-4 | Error messages shall specify invalid parameter values and valid ranges |
| USA-5 | `StatsContext.with_overrides()` shall allow easy configuration modification |
| USA-6 | The `Metric` protocol shall be simple: `name: str` + `__call__(x, ctx)` |
| USA-7 | Built-in simulations shall demonstrate framework capabilities with realistic parameters |
| USA-8 | Docstrings shall include Examples sections with executable code |
