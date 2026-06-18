# mcframework

![PyPI - Version](https://img.shields.io/pypi/v/mcframework)
[![Publish to PyPI](https://github.com/milanfusco/mcFramework/actions/workflows/publish.yml/badge.svg)](https://github.com/milanfusco/mcFramework/actions/workflows/publish.yml)
[![codecov](https://codecov.io/gh/milanfusco/mcframework/branch/main/graph/badge.svg)](https://codecov.io/gh/milanfusco/mcframework)
[![CI](https://github.com/milanfusco/mcframework/actions/workflows/ci.yml/badge.svg)](https://github.com/milanfusco/mcframework/actions/workflows/ci.yml)
[![Docs Deploy](https://github.com/milanfusco/mcFramework/actions/workflows/docs-deploy.yml/badge.svg)](https://github.com/milanfusco/mcFramework/actions/workflows/docs-deploy.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A lightweight, deterministic Monte Carlo simulation framework with robust statistical analytics, parallel execution, and optional GPU acceleration.

- **Documentation:** https://milanfusco.github.io/mcFramework/
- **Source:** https://github.com/milanfusco/mcframework
- **PyPI:** https://pypi.org/project/mcframework/

## Installation

```bash
pip install mcframework
```

The base install depends only on NumPy (>= 1.26) and SciPy (>= 1.10). Optional features are available as extras:

```bash
pip install "mcframework[torch]"   # PyTorch backends (CPU, plus MPS/CUDA when available)
pip install "mcframework[cuda]"    # PyTorch and CuPy for the cuRAND backend
pip install "mcframework[viz]"     # Matplotlib for visualization
pip install "mcframework[gui]"     # PySide6 desktop application
```

For development, install from source with the test and docs extras:

```bash
git clone https://github.com/milanfusco/mcframework.git
cd mcframework
pip install -e ".[dev,test,docs]"
```

## Quick Start

```python
from mcframework import PiEstimationSimulation

sim = PiEstimationSimulation()
sim.set_seed(42)

result = sim.run(10_000, backend="thread")
print(result)
```

Define a custom simulation by implementing a single method:

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

Request richer statistics on any run:

```python
result = sim.run(
    50_000,
    percentiles=(1, 5, 50, 95, 99),
    confidence=0.99,
    ci_method="auto",
)
print(result.stats["ci_mean"])  # 99% confidence interval
```

Register and compare multiple simulations with `MonteCarloFramework`:

```python
from mcframework import MonteCarloFramework, PiEstimationSimulation

fw = MonteCarloFramework()
fw.register_simulation(PiEstimationSimulation())

result = fw.run_simulation("Pi Estimation", 10_000, n_points=5000, backend="thread")
print(result.result_to_string())
```

## Features

**Core framework**

- `MonteCarloSimulation` base class: define a simulation by implementing `single_simulation()`.
- Deterministic parallelism: reproducible results via NumPy `SeedSequence` spawning.
- Cross-platform execution: threads on POSIX, processes on Windows.
- Structured `SimulationResult` with metadata and formatting.

**Statistics engine**

- Descriptive statistics: mean, standard deviation, percentiles, skew, kurtosis.
- Parametric confidence intervals: z and t critical values with auto-selection.
- Bootstrap confidence intervals: percentile and BCa (bias-corrected and accelerated).
- Distribution-free bounds: Chebyshev intervals and Markov probability.

**Torch backends**

- CUDA (NVIDIA): adaptive batch sizing, CUDA streams, dual RNG (`torch.Generator` and cuRAND), native float64, multi-GPU.
- MPS (Apple Silicon): Metal Performance Shaders on M1/M2/M3/M4, unified memory, best-effort determinism.
- Torch CPU: vectorized batch execution without GPU hardware.
- Pluggable `ExecutionBackend` protocol for custom backends.

**Profiling**

- PyTorch profiler integration for CPU and CUDA, with Chrome trace export and optional memory and FLOPs reporting.

**Built-in simulations**

- Pi estimation (geometric probability on the unit disk).
- Portfolio simulation (geometric Brownian motion wealth dynamics).
- Black-Scholes European and American option pricing with Greeks.

## Execution Backends

`MonteCarloSimulation.run()` selects an execution strategy via the `backend` parameter:

| Backend | Selection | Description |
| ------- | --------- | ----------- |
| `"sequential"` | `backend="sequential"` | Single-threaded execution |
| `"thread"` | `backend="thread"` (POSIX default) | `ThreadPoolExecutor`, effective when NumPy releases the GIL |
| `"process"` | `backend="process"` (Windows default) | `ProcessPoolExecutor`, avoids GIL serialization |
| `"torch"` | `backend="torch", torch_device="cpu"|"mps"|"cuda"` | Vectorized batching on CPU, Apple Silicon, or NVIDIA GPU |

### Reproducibility note

Determinism is **per-(backend, block-layout)**, not absolute. The sequential
backend draws all samples from a single spawned stream, while the thread/process
backends spawn one stream per work block (block count depends on `n_workers`,
`n_simulations`, and the chunk factor). Since `backend="auto"` switches from
sequential to parallel once `n_simulations` crosses `20_000`, the same `seed`
can yield different draws above vs. below that threshold, and sequential and
parallel runs are not bitwise identical. Pin `backend` and `n_workers` for
run-to-run reproducible numbers; statistical properties (mean, variance, CI
coverage) hold regardless.

## Performance

![Backend performance comparison on Apple Silicon](https://raw.githubusercontent.com/milanfusco/mcFramework/main/demos/apple_silicon_benchmark.png)

Execution time, throughput, and speedup across backends on Apple Silicon. Speedup is measured against single-sample sequential execution; the vectorized Torch and GPU backends benefit most at large simulation counts, where batching amortizes per-call overhead. Run `python demos/demo_apple_silicon_benchmark.py` to reproduce the figure on your own hardware.

## GPU Acceleration

A simulation runs on PyTorch when it sets `supports_batch = True` and implements `torch_batch()`. The same implementation runs on CPU, Apple Silicon MPS, and NVIDIA CUDA.

```python
from mcframework import MonteCarloSimulation
import torch

class MySimulation(MonteCarloSimulation):
    supports_batch = True

    def single_simulation(self, _rng=None, **kwargs):
        rng = self._rng(_rng, self.rng)
        x, y = rng.random(), rng.random()
        return 4.0 if (x * x + y * y) <= 1.0 else 0.0

    def torch_batch(self, n, *, device, generator):
        x = torch.rand(n, device=device, generator=generator)
        y = torch.rand(n, device=device, generator=generator)
        return 4.0 * ((x * x + y * y) <= 1.0).float()

result = sim.run(1_000_000, backend="torch", torch_device="cuda")  # or "mps", "cpu"
```

See the backend guides for configuration, determinism notes, and troubleshooting:

- [CUDA Backend Guide](https://milanfusco.github.io/mcFramework/guides/cuda.html)
- [MPS Backend Guide](https://milanfusco.github.io/mcFramework/guides/mps.html)

## GUI Application

The `gui` extra installs a PySide6 desktop application for Black-Scholes Monte Carlo analysis: live market data, path simulations, option pricing with Greeks, what-if analysis, and 3D price surfaces.

```bash
pip install "mcframework[gui]"
python demos/gui/quant_black_scholes.py
```

## Development

```bash
pytest --cov=mcframework -v   # tests with coverage
ruff check src/               # lint
pylint src/mcframework/       # lint
sphinx-build -b html docs/source docs/_build/html   # build docs
```

The [Architecture guide](https://milanfusco.github.io/mcFramework/architecture.html) documents the package layout, design patterns, and UML diagrams.

## License

MIT License. See [LICENSE](LICENSE).

## Author

Milan Fusco ([mdfusco@student.ysu.edu](mailto:mdfusco@student.ysu.edu))
