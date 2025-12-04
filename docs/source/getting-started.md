# Getting Started

This guide walks you through the fundamentals of `mcframework`, from running your first simulation to building custom ones with advanced statistical analysis.

## Prerequisites

- Python 3.10+
- NumPy, SciPy, Matplotlib (installed automatically with the package)

```bash
git clone https://github.com/milanfusco/mcframework.git
cd mcframework
pip install -e .
```

---

## Your First Simulation

Let's estimate π using Monte Carlo integration. The idea: throw random darts at a square and count how many land inside an inscribed circle.

```python
from mcframework import MonteCarloFramework, PiEstimationSimulation

# Create and seed the simulation
sim = PiEstimationSimulation()
sim.set_seed(42)  # For reproducibility

# Register with the framework
fw = MonteCarloFramework()
fw.register_simulation(sim)

# Run 10,000 simulations, each throwing 5,000 darts
result = fw.run_simulation(
    "Pi Estimation",
    n_simulations=10_000,
    n_points=5000,
    parallel=True
)

# View results
print(result.result_to_string())
```

**What happened?**

 1. Each of the 10,000 simulations threw 5,000 random points
 2. The framework ran them in parallel using all CPU cores
 3. Statistics were automatically computed on the resulting estimates

---

## Understanding `SimulationResult`

Every simulation returns a `SimulationResult` object containing:

```python
result.results        # np.ndarray of raw simulation values
result.n_simulations  # Number of simulations run
result.execution_time # Wall-clock time in seconds
result.mean           # Sample mean
result.std            # Sample standard deviation
result.percentiles    # Dict of requested percentiles {5: ..., 50: ..., 95: ...}
result.stats          # Additional stats from the engine (ci_mean, skew, etc.)
result.metadata       # Timestamps, seed info, etc.
```

### Confidence Intervals

The stats engine computes confidence intervals automatically:

```python
ci = result.stats.get("ci_mean")
print(f"95% CI: [{ci[0]:.6f}, {ci[1]:.6f}]")

# For more detail, access the full CI dict:
ci_detail = result.stats.get("ci_mean")  # Contains: low, high, method, se, crit
```

---

## Customizing Your Run

### Control Statistics Computation

```python
# Custom percentiles
result = sim.run(
    10_000,
    percentiles=(1, 5, 25, 50, 75, 95, 99),
    confidence=0.99,      # 99% CI instead of 95%
    ci_method="t",        # Force t-distribution critical values
)

# Disable stats engine for raw speed
result = sim.run(10_000, compute_stats=False)
```

### Parallel Execution Control

```python
# Control worker count
result = sim.run(50_000, parallel=True, n_workers=4)

# Force specific backend
sim.parallel_backend = "thread"   # or "process"
result = sim.run(50_000, parallel=True)

# Progress callback
def on_progress(completed, total):
    print(f"Progress: {completed}/{total} ({100*completed/total:.1f}%)")

result = sim.run(50_000, parallel=True, progress_callback=on_progress)
```

---

## Building Custom Simulations

Create your own simulation by subclassing `MonteCarloSimulation`:

```python
from mcframework import MonteCarloSimulation
import numpy as np

class CoinFlipStreakSimulation(MonteCarloSimulation):
    """
    Simulate flipping coins and find the longest streak of heads.
    """
    
    def __init__(self):
        super().__init__("Coin Flip Streak")
    
    def single_simulation(
        self,
        n_flips: int = 100,
        _rng=None,
        **kwargs
    ) -> float:
        """
        Flip n_flips coins and return the longest consecutive heads streak.
        
        Parameters
        ----------
        n_flips : int
            Number of coin flips per simulation
        _rng : Generator, optional
            Thread-safe RNG provided by the framework
        
        Returns
        -------
        float
            Length of the longest heads streak
        """
        # Always use the framework's RNG for reproducibility
        rng = self._rng(_rng, self.rng)
        
        # Simulate flips (1 = heads, 0 = tails)
        flips = rng.integers(0, 2, size=n_flips)
        
        # Find longest streak of heads
        max_streak = current_streak = 0
        for flip in flips:
            if flip == 1:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return float(max_streak)


# Use your custom simulation
sim = CoinFlipStreakSimulation()
sim.set_seed(42)

result = sim.run(
    n_simulations=50_000,
    n_flips=1000,
    parallel=True,
    percentiles=(50, 90, 99)
)

print(f"Expected longest streak in 1000 flips: {result.mean:.1f}")
print(f"Median: {result.percentiles[50]:.0f}")
print(f"90th percentile: {result.percentiles[90]:.0f}")
print(f"99th percentile: {result.percentiles[99]:.0f}")
```

---

## Portfolio Simulation Example

Model investment growth under Geometric Brownian Motion:

```python
from mcframework import PortfolioSimulation, MonteCarloFramework

sim = PortfolioSimulation()
sim.set_seed(2024)

fw = MonteCarloFramework()
fw.register_simulation(sim)

# Simulate $100k invested for 30 years
result = fw.run_simulation(
    "Portfolio Simulation",
    n_simulations=100_000,
    initial_value=100_000,
    annual_return=0.07,    # 7% expected return
    volatility=0.18,       # 18% annual volatility
    years=30,
    parallel=True,
    percentiles=(5, 25, 50, 75, 95)
)

print(f"After 30 years:")
print(f"  Expected value: ${result.mean:,.0f}")
print(f"  Median value: ${result.percentiles[50]:,.0f}")
print(f"  5th percentile (bad case): ${result.percentiles[5]:,.0f}")
print(f"  95th percentile (good case): ${result.percentiles[95]:,.0f}")
```

---

## Black-Scholes Option Pricing

Price European and American options with Greeks:

```python
from mcframework import BlackScholesSimulation

bs = BlackScholesSimulation()
bs.set_seed(42)

# Price a European call option
result = bs.run(
    n_simulations=100_000,
    S0=100.0,         # Current stock price
    K=105.0,          # Strike price
    T=0.5,            # Time to maturity (years)
    r=0.05,           # Risk-free rate
    sigma=0.25,       # Volatility
    option_type="call",
    exercise_type="european",
    parallel=True
)

print(f"European Call Price: ${result.mean:.4f}")
print(f"95% CI: [${result.stats['ci_mean'][0]:.4f}, ${result.stats['ci_mean'][1]:.4f}]")

# Calculate Greeks
greeks = bs.calculate_greeks(
    n_simulations=50_000,
    S0=100.0, K=105.0, T=0.5, r=0.05, sigma=0.25,
    option_type="call",
    parallel=True
)

print(f"\nGreeks:")
print(f"  Delta: {greeks['delta']:.4f}")
print(f"  Gamma: {greeks['gamma']:.6f}")
print(f"  Vega:  {greeks['vega']:.4f}")
print(f"  Theta: {greeks['theta']:.4f}")
print(f"  Rho:   {greeks['rho']:.4f}")
```

---

## Comparing Multiple Simulations

Use `MonteCarloFramework` to manage and compare simulations:

```python
from mcframework import MonteCarloFramework, PiEstimationSimulation

fw = MonteCarloFramework()

# Create variations with different parameters
sim_small = PiEstimationSimulation()
sim_small.name = "Pi (1k points)"
sim_small.set_seed(42)

sim_large = PiEstimationSimulation()
sim_large.name = "Pi (100k points)"
sim_large.set_seed(42)

fw.register_simulation(sim_small)
fw.register_simulation(sim_large)

# Run both
fw.run_simulation("Pi (1k points)", 1000, n_points=1000, parallel=True)
fw.run_simulation("Pi (100k points)", 1000, n_points=100_000, parallel=True)

# Compare metrics
comparison = fw.compare_results(
    ["Pi (1k points)", "Pi (100k points)"],
    metric="std"  # Compare standard deviations
)

print("Standard deviation comparison:")
for name, std in comparison.items():
    print(f"  {name}: {std:.6f}")
```

---

## Advanced: Custom Statistics Engine

Create a custom stats engine with specific metrics:

```python
from mcframework.stats_engine import (
    StatsEngine, FnMetric, StatsContext,
    mean, std, ci_mean, percentiles
)
import numpy as np

# Define custom metrics
def coefficient_of_variation(x, ctx):
    """CV = std / mean, measuring relative variability."""
    m = float(np.mean(x))
    s = float(np.std(x, ddof=1))
    return s / m if m != 0 else float('nan')

def interquartile_range(x, ctx):
    """IQR = Q3 - Q1"""
    q1, q3 = np.percentile(x, [25, 75])
    return float(q3 - q1)

# Build custom engine
custom_engine = StatsEngine([
    FnMetric("mean", mean),
    FnMetric("std", std),
    FnMetric("ci_mean", ci_mean),
    FnMetric("cv", coefficient_of_variation),
    FnMetric("iqr", interquartile_range),
])

# Use with a simulation
from mcframework import PiEstimationSimulation

sim = PiEstimationSimulation()
sim.set_seed(42)

result = sim.run(
    10_000,
    stats_engine=custom_engine,
    parallel=True
)

print(f"Mean: {result.stats.get('mean', result.mean):.6f}")
print(f"CV: {result.stats.get('cv'):.4f}")
print(f"IQR: {result.stats.get('iqr'):.6f}")
```

---

## Bootstrap Confidence Intervals

For non-normal distributions, use bootstrap CIs:

```python
from mcframework import PiEstimationSimulation
from mcframework.stats_engine import StatsContext

sim = PiEstimationSimulation()
sim.set_seed(42)

result = sim.run(
    5000,
    parallel=True,
    extra_context={
        "n_bootstrap": 10_000,
        "bootstrap": "bca",  # Bias-corrected and accelerated
    }
)

# Bootstrap CI is in stats
bootstrap_ci = result.stats.get("ci_mean_bootstrap")
if bootstrap_ci:
    print(f"Bootstrap 95% CI: [{bootstrap_ci['low']:.6f}, {bootstrap_ci['high']:.6f}]")
    print(f"Method: {bootstrap_ci['method']}")
```

---

## Reproducibility Guide

For fully reproducible results:

1. **Always set a seed** before running:

   ```python
   sim.set_seed(42)
   ```

2. **Use the same worker count** for parallel runs:

   ```python
   result = sim.run(10_000, parallel=True, n_workers=8)
   ```

3. **Check seed entropy** in results:

   ```python
   print(f"Seed entropy: {result.metadata['seed_entropy']}")
   ```

The framework uses NumPy's `SeedSequence.spawn()` to create independent, deterministic random streams for each parallel worker.

---

## Next Steps

- Explore the [API Reference](api/_modules/core.rst) for detailed class documentation
- Check out the [demos/](https://github.com/milanfusco/mcframework/tree/main/demos) folder for visualization examples
- Try the GUI application for interactive Black-Scholes analysis:

  ```bash
  pip install -e ".[gui]"
  python demos/gui/quant_black_scholes.py
  ```
  