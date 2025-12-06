# mcframework

![PyPI - Version](https://img.shields.io/pypi/v/mcframework)
[![Publish to PyPI](https://github.com/milanfusco/mcFramework/actions/workflows/publish.yml/badge.svg)](https://github.com/milanfusco/mcFramework/actions/workflows/publish.yml)
[![codecov](https://codecov.io/gh/milanfusco/mcframework/branch/main/graph/badge.svg)](https://codecov.io/gh/milanfusco/mcframework)
[![CI](https://github.com/milanfusco/mcframework/actions/workflows/ci.yml/badge.svg)](https://github.com/milanfusco/mcframework/actions/workflows/ci.yml)
[![Docs Deploy](https://github.com/milanfusco/mcFramework/actions/workflows/docs-deploy.yml/badge.svg)](https://github.com/milanfusco/mcFramework/actions/workflows/docs-deploy.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Lightweight, reproducible, and deterministic Monte Carlo simulation framework with statistically robust analytics and parallel execution.

## 📚 Documentation

**[View Full Documentation →](https://milanfusco.github.io/mcFramework/)**

The documentation includes:

- **Getting Started** — Installation and quick examples
- **API Reference** — Complete module documentation with type hints
- **System Design** — Architecture diagrams, UML, and design patterns
- **Project Plan** — Requirements, stakeholders, and methodology
---

## Installation

### From Source (Development)

```bash
git clone https://github.com/milanfusco/mcframework.git
cd mcframework
pip install -e .
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Python | ≥ 3.10 | Runtime |
| NumPy | ≥ 1.24 | Arrays, RNG |
| SciPy | ≥ 1.10 | Statistics |
| Matplotlib | ≥ 3.7 | Visualization |

### Optional Dependencies

```bash
# All extras
pip install -e ".[dev,test,docs,gui]"

# Individual extras
pip install -e ".[dev]"   # Linting (ruff, pylint)
pip install -e ".[test]"  # Testing (pytest, coverage)
pip install -e ".[docs]"  # Documentation (Sphinx, themes)
pip install -e ".[gui]"   # GUI application (PySide6)
```

---

## Features

### Core Framework
- **Abstract base class** (`MonteCarloSimulation`) — Define simulations by implementing `single_simulation()`
- **Deterministic parallelism** — Reproducible results via NumPy `SeedSequence` spawning
- **Cross-platform execution** — Threads on POSIX, processes on Windows
- **Structured results** — `SimulationResult` dataclass with metadata and formatting

### Statistics Engine
- **Descriptive statistics** — Mean, std, percentiles, skew, kurtosis
- **Parametric CI** — z/t critical values with auto-selection
- **Bootstrap CI** — Percentile and BCa methods
- **Distribution-free bounds** — Chebyshev intervals, Markov probability

### Built-in Simulations
- **Pi Estimation** — Geometric probability on unit disk
- **Portfolio Simulation** — GBM wealth dynamics
- **Black-Scholes** — European/American option pricing with Greeks

---

## Quick Start

```python
from mcframework import MonteCarloFramework, PiEstimationSimulation

sim = PiEstimationSimulation()
sim.set_seed(123)

fw = MonteCarloFramework()
fw.register_simulation(sim)

result = fw.run_simulation("Pi Estimation", 10_000, n_points=5000, parallel=True)
print(result.result_to_string())
```

### Defining a Custom Simulation

```python
from mcframework import MonteCarloSimulation

class DiceSumSimulation(MonteCarloSimulation):
    def __init__(self):
        super().__init__("Dice Sum")

    def single_simulation(self, _rng=None, n_dice: int = 5) -> float:
        rng = self._rng(_rng, self.rng)
        return float(rng.integers(1, 7, size=n_dice).sum())

sim = DiceSumSimulation()
sim.set_seed(42)
result = sim.run(10_000, parallel=True)
print(f"Mean: {result.mean:.2f}")  # ~17.5
```

### Extended Statistics

```python
result = sim.run(
    50_000,
    percentiles=(1, 5, 50, 95, 99),
    confidence=0.99,
    ci_method="auto",
)
print(result.stats["ci_mean"])  # 99% confidence interval
```

---

## Package Structure

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

## GUI Application

The framework includes a PySide6 GUI for Black-Scholes Monte Carlo simulations:

```bash
pip install -e ".[gui]"
python demos/gui/quant_black_scholes.py
```

**Features:**
- Live stock data from Yahoo Finance
- Monte Carlo path simulations
- Option pricing with Greeks (Δ, Γ, ν, Θ, ρ)
- Interactive what-if analysis
- 3D option price surfaces
- HTML report export
- 
**Scenario Presets:** High volatility (TSLA), Index ETFs (SPY), Crypto-adjacent (COIN), Dividend stocks (JNJ)

---

## Cross-Platform Parallel Execution

`MonteCarloSimulation.run(..., parallel=True)` automatically selects the optimal backend:

| Platform | Default Backend | Reason |
|----------|-----------------|--------|
| POSIX (Linux, macOS) | `ThreadPoolExecutor` | NumPy releases GIL |
| Windows | `ProcessPoolExecutor` | Avoids GIL serialization |

Override explicitly with `parallel_backend="thread"` or `parallel_backend="process"`.

---

## Development

### Testing

```bash
# Run tests with coverage
pytest --cov=mcframework -v

# Generate coverage reports
pytest --cov=mcframework --cov-report=xml:coverage.xml   # XML
pytest --cov=mcframework --cov-report=html               # HTML
```

### Linting

```bash
ruff check src/
pylint src/mcframework/
```

### Documentation

```bash
# Install docs dependencies
pip install -e ".[docs]"

# Build HTML documentation
sphinx-build -b html docs/source docs/_build/html

# Serve locally
python -m http.server 8000 -d docs/_build/html
```

The documentation uses:
- **Sphinx** with pydata-sphinx-theme
- **Mermaid** for interactive diagrams
- **NumPy-style docstrings** with LaTeX math
- **Light/dark theme toggle** with diagram re-rendering


## License

MIT License. See [LICENSE](LICENSE) file.

---

## Authors

- **Milan Fusco** — [mdfusco@student.ysu.edu](mailto:mdfusco@student.ysu.edu)
- **James Gabbert** — [jdgabbert@student.ysu.edu](mailto:jdgabbert@student.ysu.edu)
