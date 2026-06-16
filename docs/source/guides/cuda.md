# CUDA Backend

NVIDIA GPU acceleration for batch Monte Carlo simulations, with adaptive memory management and native float64 support.

## Overview

The CUDA backend provides GPU-accelerated batch execution using PyTorch's CUDA interface. Unlike the MPS backend (Apple Silicon), CUDA supports native float64 precision and adds several performance features:

- **Adaptive batch sizing**: estimates an optimal batch size from available GPU memory.
- **Dual RNG modes**: PyTorch's generator (default) or cuRAND via CuPy for higher throughput.
- **CUDA streams**: overlapped execution for better GPU utilization.
- **Native float64**: no conversion overhead, unlike the MPS backend.
- **Memory safety**: a probe run estimates per-sample memory to avoid out-of-memory errors.

## Requirements

- NVIDIA GPU with CUDA support ([compatibility list](https://developer.nvidia.com/cuda-gpus))
- PyTorch with CUDA enabled ([installation guide](https://pytorch.org/get-started/locally/))
- Optional: CuPy for cuRAND mode ([installation](https://docs.cupy.dev/en/stable/install.html))

Check CUDA availability:

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"
```

## Installation

Basic GPU support installs PyTorch with CUDA. The default `torch.Generator` RNG mode works out of the box:

```bash
pip install -e ".[torch]"
```

For cuRAND mode (native GPU random number generation), add CuPy:

```bash
pip install -e ".[cuda]"
```

CuPy requires CUDA toolkit development files. If you hit build errors, install prebuilt wheels:

```bash
pip install cupy-cuda12x  # CUDA 12.x
pip install cupy-cuda11x  # CUDA 11.x
```

See the [CuPy installation docs](https://docs.cupy.dev/en/stable/install.html) for troubleshooting.

## Quick Start

```python
from mcframework.sims import PiEstimationSimulation

sim = PiEstimationSimulation()
sim.set_seed(42)

# Run 1M simulations on GPU (adaptive batching is automatic)
result = sim.run(1_000_000, backend="torch", torch_device="cuda")
print(f"pi ~= {result.mean:.6f} +/- {result.std / 1000:.6f}")
```

The backend automatically estimates an optimal batch size, splits the workload into batches when needed, uses CUDA streams for overlapped execution, and returns float64 results for statistical precision.

### Advanced Configuration

For fine-grained control, construct the backend directly:

```python
from mcframework.backends import TorchCUDABackend

backend = TorchCUDABackend(
    device_id=0,          # first GPU
    batch_size=100_000,   # fixed batch size
    use_streams=True,     # enable CUDA streams
)

results = backend.run(sim, n_simulations=10_000_000, seed_seq=sim.seed_seq)
```

## Memory Management

The backend uses a two-phase approach to prevent out-of-memory errors:

1. **Probe run**: executes 1,000 samples to estimate per-sample memory usage.
2. **Batch calculation**: allocates roughly 75% of available GPU memory per batch.

For very large workloads, it splits the run across batches with progress tracking. To set the batch size yourself:

```python
backend = TorchCUDABackend(device_id=0, batch_size=50_000)
```

Or let adaptive batching handle it:

```python
backend = TorchCUDABackend(device_id=0)  # batch_size=None (default)
```

## RNG Modes

### torch.Generator (default)

Uses PyTorch's Philox counter-based RNG. Fully deterministic and available everywhere:

```python
sim.set_seed(42)
result1 = sim.run(100_000, backend="torch", torch_device="cuda")
result2 = sim.run(100_000, backend="torch", torch_device="cuda")
# Bitwise-identical results
```

See [PyTorch Random Sampling](https://pytorch.org/docs/stable/notes/randomness.html).

### cuRAND (optional)

Native GPU random number generation via CuPy, which can be faster for RNG-heavy simulations:

```python
backend = TorchCUDABackend(device_id=0, use_curand=True)
results = backend.run(sim, n_simulations=1_000_000, seed_seq=sim.seed_seq)
```

cuRAND mode requires CuPy and a simulation that implements `curand_batch()` (see below). References: [cuRAND Library](https://docs.nvidia.com/cuda/curand/index.html), [CuPy Random](https://docs.cupy.dev/en/stable/reference/random.html).

## Implementing GPU-Accelerated Simulations

A GPU-capable simulation needs two things: set `supports_batch = True`, and implement `torch_batch()`.

```python
from mcframework import MonteCarloSimulation
import torch

class PiEstimationSimulation(MonteCarloSimulation):
    supports_batch = True

    def single_simulation(self, _rng=None, **kwargs):
        # NumPy implementation for CPU and parallel backends
        rng = self._rng(_rng, self.rng)
        x, y = rng.random(), rng.random()
        return 4.0 if (x * x + y * y) <= 1.0 else 0.0

    def torch_batch(self, n, *, device, generator):
        # GPU-accelerated batch implementation
        x = torch.rand(n, device=device, generator=generator)
        y = torch.rand(n, device=device, generator=generator)
        inside = (x * x + y * y) <= 1.0
        return 4.0 * inside.float()
```

Key points:

- Use the provided `generator` for all random sampling (never `torch.manual_seed()`).
- Return float32 or float64 tensors (float64 is preferred on CUDA).
- The framework handles device placement and dtype conversion.

For cuRAND, also implement `curand_batch()`:

```python
def curand_batch(self, n, device_id, rng):
    import cupy as cp
    x = rng.uniform(0, 1, size=n, dtype=cp.float32)
    y = rng.uniform(0, 1, size=n, dtype=cp.float32)
    inside = (x * x + y * y) <= 1.0
    return 4.0 * inside.astype(cp.float32)
```

## Performance

CUDA has several advantages over the MPS backend:

| Feature | CUDA (NVIDIA) | MPS (Apple Silicon) | CPU |
|---------|---------------|---------------------|-----|
| float64 support | Native | Emulated | Native |
| Determinism | Bitwise | Statistical | Bitwise |
| Multi-GPU | Yes | Single device | Multi-core |
| Streams | Yes | No | N/A |

To compare backends on your hardware:

```python
import time

sim = PiEstimationSimulation()
sim.set_seed(42)

start = time.time()
sim.run(1_000_000, backend="sequential", compute_stats=False)
cpu_time = time.time() - start

start = time.time()
sim.run(1_000_000, backend="torch", torch_device="cuda", compute_stats=False)
cuda_time = time.time() - start

print(f"Speedup: {cpu_time / cuda_time:.2f}x")
```

The speedup depends on simulation complexity and grows with the amount of computational work per sample.

## Multi-GPU

Select a device with `device_id`:

```python
backend = TorchCUDABackend(device_id=1)  # second GPU

import torch
print(f"Available GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  [{i}] {torch.cuda.get_device_name(i)}")
```

For data parallelism across multiple GPUs, run separate backend instances concurrently.

## Architecture

The implementation keeps a clean separation of concerns:

```
sim.run(n, backend="torch", torch_device="cuda")
    |
simulation.py        delegates to the TorchBackend factory
    |
backends/torch.py    creates TorchCUDABackend with config
    |
backends/torch_cuda.py   all CUDA-specific implementation
```

This keeps `simulation.py` backend-agnostic; CUDA-specific complexity is isolated in the backends module.

## Error Handling

The backend validates configuration and reports actionable errors. A simulation that declares `supports_batch = True` but does not implement `torch_batch()` raises a `NotImplementedError` naming the class and explaining how to fix it:

```python
class BadSimulation(MonteCarloSimulation):
    supports_batch = True  # claims GPU support

    def single_simulation(self, _rng=None):
        return 1.0

sim = BadSimulation()
sim.run(100, backend="torch", torch_device="cuda")
# NotImplementedError: Simulation 'BadSimulation' has supports_batch = True
# but does not implement torch_batch().
```

## Testing

CUDA tests skip automatically when no GPU is available:

```bash
pytest tests/test_torch_cuda.py -v       # CUDA backend tests
pytest tests/test_torch_cuda.py -v -rs   # show skip reasons
```

Coverage includes validation and error handling, determinism, adaptive batch sizing, memory management, CUDA streams, and multi-device support.

## Troubleshooting

**"CUDA device requested but not available"**

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If this prints `False`, reinstall PyTorch with CUDA support from the [PyTorch install guide](https://pytorch.org/get-started/locally/).

**"CUDA out of memory"**

Reduce the batch size, or let adaptive batching handle it:

```python
backend = TorchCUDABackend(device_id=0, batch_size=10_000)
```

**CuPy build errors**

Install a prebuilt wheel matching your CUDA version:

```bash
pip install cupy-cuda12x
```

See the [CuPy install docs](https://docs.cupy.dev/en/stable/install.html).

**Non-deterministic results**

Set the seed before running:

```python
sim.set_seed(42)
result = sim.run(...)
```

## References

- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [PyTorch CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html)
- [CuPy Documentation](https://docs.cupy.dev/en/stable/)
- [cuRAND Library](https://docs.nvidia.com/cuda/curand/index.html)
- [PyTorch Random Sampling](https://pytorch.org/docs/stable/notes/randomness.html)
