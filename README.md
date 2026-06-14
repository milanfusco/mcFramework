# mcframework

![PyPI - Version](https://img.shields.io/pypi/v/mcframework)
[![Publish to PyPI](https://github.com/milanfusco/mcFramework/actions/workflows/publish.yml/badge.svg)](https://github.com/milanfusco/mcFramework/actions/workflows/publish.yml)
[![codecov](https://codecov.io/gh/milanfusco/mcframework/branch/main/graph/badge.svg)](https://codecov.io/gh/milanfusco/mcframework)
[![CI](https://github.com/milanfusco/mcframework/actions/workflows/ci.yml/badge.svg)](https://github.com/milanfusco/mcframework/actions/workflows/ci.yml)
[![Docs Deploy](https://github.com/milanfusco/mcFramework/actions/workflows/docs-deploy.yml/badge.svg)](https://github.com/milanfusco/mcFramework/actions/workflows/docs-deploy.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Lightweight, deterministic Monte Carlo simulation framework with robust statistical analytics, parallel execution, and GPU acceleration.

---

## Installation

```bash
pip install mcframework
```

### From Source (Development)

```bash
git clone https://github.com/milanfusco/mcframework.git
cd mcframework
pip install -e .
```

### Dependencies

| Package | Version | Purpose |
| ------- | ------- | ------- |
| Python | ≥ 3.10 | Runtime |
| NumPy | ≥ 1.26 | Arrays, RNG |
| SciPy | ≥ 1.10 | Statistics |
| Matplotlib | ≥ 3.7 | Visualization (optional) |
| PyTorch | ≥ 2.9.1 | GPU acceleration (optional) |
| CuPy | ≥ 12.0.0 | cuRAND backend (optional) |

### Optional Dependencies

```bash
# All extras
pip install -e ".[viz,dev,test,docs,gui,torch,cuda]"

# Individual extras
pip install -e ".[viz]"   # Visualization (Matplotlib)
pip install -e ".[dev]"   # Linting (ruff, pylint)
pip install -e ".[test]"  # Testing (pytest, coverage)
pip install -e ".[docs]"  # Documentation (Sphinx, themes)
pip install -e ".[gui]"   # GUI application (PySide6)
pip install -e ".[torch]"  # PyTorch backend (CPU, plus MPS/CUDA when available)
pip install -e ".[cuda]"  # PyTorch + CuPy for cuRAND
```

---

## Quick Start

```python
from mcframework import PiEstimationSimulation

sim = PiEstimationSimulation()
sim.set_seed(42)

result = sim.run(10_000, backend="thread")
print(result)
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
result = sim.run(10_000, backend="thread")
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

### Using the Framework Registry

`MonteCarloFramework` provides a registry for managing and comparing multiple simulations:

```python
from mcframework import MonteCarloFramework, PiEstimationSimulation

fw = MonteCarloFramework()
fw.register_simulation(PiEstimationSimulation())

result = fw.run_simulation("Pi Estimation", 10_000, n_points=5000, backend="thread")
print(result.result_to_string())
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
- **Bootstrap CI** — Percentile and BCa (bias-corrected and accelerated) methods
- **Distribution-free bounds** — Chebyshev intervals, Markov probability

### Torch Backends

- **CUDA** (NVIDIA) — Adaptive batch sizing, CUDA streams, dual RNG (torch.Generator + cuRAND), native float64, multi-GPU
- **MPS** (Apple Silicon) — Metal Performance Shaders on M1/M2/M3/M4, unified memory, best-effort determinism
- **Torch CPU** — Vectorized batch execution via PyTorch without GPU hardware
- **Pluggable architecture** — Protocol-based `ExecutionBackend` interface for custom backends

### Profiling

- **PyTorch profiler integration** — CPU and CUDA profiling with configurable schedules
- **Chrome trace export** — Visualize execution timelines in `chrome://tracing`
- **Memory and FLOPs** — Optional memory profiling and floating-point operation estimation

### Built-in Simulations

- **Pi Estimation** — Geometric probability on unit disk
- **Portfolio Simulation** — GBM (Geometric Brownian Motion) wealth dynamics
- **Black-Scholes** — European/American option pricing with Greeks

---

## Package Structure

```text
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
│   ├── torch_base.py    # Shared Torch/cuRAND utilities
│   ├── torch_cpu.py     # PyTorch CPU backend
│   ├── torch_mps.py     # Apple Silicon MPS backend
│   └── torch_cuda.py    # NVIDIA CUDA backend
└── sims/
    ├── __init__.py      # Simulation catalog
    ├── pi.py            # PiEstimationSimulation
    ├── portfolio.py     # PortfolioSimulation
    └── black_scholes.py # BlackScholesSimulation, BlackScholesPathSimulation
```

---

## Execution Backends

`MonteCarloSimulation.run()` supports multiple execution strategies via the `backend` parameter:

| Backend | Selection | Description |
| ------- | --------- | ----------- |
| `"sequential"` | `backend="sequential"` | Single-threaded execution |
| `"thread"` | `backend="thread"` (POSIX default) | `ThreadPoolExecutor` — NumPy releases GIL |
| `"process"` | `backend="process"` (Windows default) | `ProcessPoolExecutor` — avoids GIL serialization |
| `"torch"` (CPU) | `backend="torch", torch_device="cpu"` | Vectorized PyTorch batching on CPU |
| `"torch"` (MPS) | `backend="torch", torch_device="mps"` | Apple Silicon GPU via Metal |
| `"torch"` (CUDA) | `backend="torch", torch_device="cuda"` | NVIDIA GPU with adaptive batching |

---

## GPU Acceleration

Simulations that implement `torch_batch()` run on PyTorch — on CPU everywhere, and on NVIDIA CUDA or Apple Silicon MPS devices with significant speedups when available. Install PyTorch via the `torch` or `cuda` extras (see [Optional Dependencies](#optional-dependencies)).

### PyTorch Quick Start

```python
from mcframework import PiEstimationSimulation

sim = PiEstimationSimulation()
sim.set_seed(42)

# NVIDIA GPU
result = sim.run(1_000_000, backend="torch", torch_device="cuda")

# Apple Silicon GPU
result = sim.run(1_000_000, backend="torch", torch_device="mps")

# PyTorch CPU (vectorized batching, no GPU required)
result = sim.run(1_000_000, backend="torch", torch_device="cpu")
```

### Implementing GPU-Accelerated Simulations

Set `supports_batch = True` as a class attribute and implement `torch_batch()`:

```python
from mcframework import MonteCarloSimulation
import torch

class MySimulation(MonteCarloSimulation):
    supports_batch = True

    def single_simulation(self, _rng=None, **kwargs):
        rng = self._rng(_rng, self.rng)
        x, y = rng.random(), rng.random()
        return 4.0 if (x*x + y*y) <= 1.0 else 0.0

    def torch_batch(self, n, *, device, generator):
        x = torch.rand(n, device=device, generator=generator)
        y = torch.rand(n, device=device, generator=generator)
        return 4.0 * ((x*x + y*y) <= 1.0).float()
```

### Backend Comparison

| Feature | CUDA (NVIDIA) | MPS (Apple Silicon) | CPU |
| ------- | ------------- | ------------------- | --- |
| float64 support | Native | Emulated (float32 -> float64) | Native |
| Determinism | Bitwise | Best-effort (statistical) | Bitwise |
| Multi-GPU | Yes | Single device | Multi-core |
| CUDA streams | Yes | No | N/A |
| Adaptive batching | Yes | No | Sequential/Parallel |

For detailed usage, configuration, and troubleshooting, see the dedicated backend guides:

- **[CUDA Backend Guide](CUDA_README.md)** — adaptive batching, cuRAND, streams, multi-GPU
- **[MPS Backend Guide](MPS_README.md)** — Apple Silicon setup, float32 handling, determinism

---

## GUI Application

The framework includes a PySide6 GUI for Black-Scholes Monte Carlo simulations.

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

**Scenario Presets:** High volatility (TSLA), Index ETFs (SPY), Crypto-adjacent (COIN), Dividend stocks (JNJ)

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

---

## Links

- **[Documentation](https://milanfusco.github.io/mcFramework/)**
- **[PyPI Package](https://pypi.org/project/mcframework/)**
- **[GitHub Repository](https://github.com/milanfusco/mcframework)**
- **[QA (Coverage)](https://codecov.io/gh/milanfusco/mcframework)**

## License

MIT License. See [LICENSE](LICENSE) file.

## Author

- **Milan Fusco** — [mdfusco@student.ysu.edu](mailto:mdfusco@student.ysu.edu)
