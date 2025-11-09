# mcframework

Lightweight, reproducible Monte Carlo simulation framework.

## Installation

### From Source (Development)

Clone the repository and install in editable mode:

```bash
git clone https://github.com/yourusername/mcframework.git
cd mcframework
pip install -e .
```

### Dependencies

The framework requires:
- Python >= 3.10
- numpy >= 1.24
- scipy >= 1.10
- matplotlib >= 3.7

### Optional Dependencies

For development:
```bash
pip install -e ".[dev,test,docs]"
```

## Features

- Simple base class (`MonteCarloSimulation`) for defining simulations by implementing `single_simulation`.
- Deterministic & reproducible parallel execution using `numpy` `SeedSequence` spawning.
- Flexible process-based parallelism with stable results independent of worker scheduling.
- Extensible statistics engine (`StatsEngine`) with built-in metrics: mean, std, percentiles, skew, kurtosis, confidence interval for mean.
- Custom percentile selection and confidence interval control (`auto` / `z` / `t`).
- Structured `SimulationResult` with metadata and pretty formatting helper.
- Registry / runner (`MonteCarloFramework`) for managing multiple simulations and comparing metrics.

## Quick Start

```python
from mcframework import MonteCarloFramework, PiEstimationSimulation

sim = PiEstimationSimulation()
sim.set_seed(123)
fw = MonteCarloFramework()
fw.register_simulation(sim)
res = fw.run_simulation("Pi Estimation", 10000, n_points=5000, parallel=True)
print(res.result_to_string())
```

For a comprehensive example with visualizations, see [`demo.py`](demo.py) which demonstrates Pi estimation and portfolio simulations with detailed plots.

## Defining a Custom Simulation

```python
from mcframework import MonteCarloSimulation
import numpy as np

class DiceSumSimulation(MonteCarloSimulation):
    def __init__(self):
        super().__init__("Dice Sum")

    def single_simulation(self, rng=None, n_dice: int = 5) -> float:
        r = self._rng(rng, self.rng)
        return float(r.integers(1, 7, size=n_dice).sum())
```

## Extended Statistics

You can override percentiles and control CI computation:

```python
res = sim.run(
    50_000,
    percentiles=(1, 5, 50, 95, 99),
    confidence=0.99,
    ci_method="auto",
)
print(res.stats["ci_mean"])  # Engine-computed CI
```

Disable stats engine:

```python
res = sim.run(10_000, compute_stats=False)
```

## Best Practices Followed

- Deterministic RNG management (seed sequence spawning) for parallel reproducibility.
- Separation of concerns: core simulation orchestration vs statistical utilities (`utils`, `stats_engine`).
- Defensive logging and structured metadata capture.
- Extensible metrics via protocol + injectable engine.
- Avoid hidden state: all configurable aspects exposed as parameters.

## Roadmap / Ideas

- Bootstrap confidence intervals (more robust for non-normal data).
- Streaming / incremental statistics for huge simulation counts.
- Variance reduction techniques (control variates, importance sampling).
- Progress bar integration (e.g., `tqdm`).

## Development

Dev dependencies:

```
pip install numpy scipy matplotlib \
            pytest pylint mypy ruff sphinx \
            sphinx-rtd-theme pytest-cov
```


Run tests with coverage:

```
pytest --cov=mcframework -v 
```

Generate coverage report (XML):

```
pytest --cov=mcframework --cov-report=xml:coverage.xml
```

Generate coverage report (HTML):

```
pytest --cov=mcframework --cov-report=html
```

Build HTML docs:

```
sphinx-build -b html docs/source docs/_build/html
```

## License

MIT License. See `LICENSE` file.
