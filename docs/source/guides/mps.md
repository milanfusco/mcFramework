# MPS Backend

Apple Metal Performance Shaders (MPS) GPU acceleration for batch Monte Carlo simulations on Apple Silicon Macs.

## Overview

The MPS backend provides GPU-accelerated batch execution on Apple Silicon (M1/M2/M3/M4) using PyTorch's Metal Performance Shaders interface. Its key characteristics:

- **Apple Silicon only**: M1, M2, M3, and M4 series chips.
- **Single-batch execution**: no adaptive batching needed.
- **Best-effort determinism**: preserves RNG structure but is not bitwise reproducible.
- **Float32 only**: Metal does not support float64; the framework handles conversion.
- **Unified memory**: GPU and CPU share physical memory.

## Requirements

- macOS 12.3 (Monterey) or later
- Apple Silicon Mac (M1/M2/M3/M4 series)
- PyTorch with MPS support ([installation guide](https://pytorch.org/get-started/locally/))

Check MPS availability:

```bash
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available() and torch.backends.mps.is_built()}')"
```

## Installation

```bash
pip install -e ".[torch]"
```

This installs PyTorch with MPS support. The backend is available automatically on compatible systems.

## Quick Start

```python
from mcframework.sims import PiEstimationSimulation

sim = PiEstimationSimulation()
sim.set_seed(42)

# Run 1M simulations on the Apple GPU
result = sim.run(1_000_000, backend="torch", torch_device="mps")
print(f"pi ~= {result.mean:.6f} +/- {result.std / 1000:.6f}")
```

The backend executes on the Apple GPU, handles the float32 to float64 conversion for statistical precision, and uses unified memory for efficient transfers. To construct it directly:

```python
from mcframework.backends import TorchMPSBackend

backend = TorchMPSBackend()
results = backend.run(sim, n_simulations=1_000_000, seed_seq=sim.seed_seq)
```

## Memory Management

MPS uses Apple's unified memory architecture, so the GPU and CPU share the same physical memory. In practice this means no explicit memory management, efficient (often zero-copy) transfers, and lower out-of-memory risk than discrete GPUs with fixed memory. The backend processes simulations in a single batch.

## Determinism

MPS provides best-effort determinism, not bitwise reproducibility:

```python
sim.set_seed(42)
result1 = sim.run(100_000, backend="torch", torch_device="mps")
result2 = sim.run(100_000, backend="torch", torch_device="mps")

# Statistical properties are preserved
assert abs(result1.mean - result2.mean) < 0.01
assert abs(result1.std - result2.std) < 0.01

# But results are not bitwise identical
assert not np.array_equal(result1.results, result2.results)
```

This stems from Metal scheduling variation, float32 rounding differences, and GPU kernel execution order. For Monte Carlo work this is usually acceptable: means and variances are statistically equivalent and confidence intervals retain correct coverage. For exact bitwise reproducibility, use the CPU or CUDA backends. See the [PyTorch MPS notes](https://pytorch.org/docs/stable/notes/mps.html).

## Implementing MPS-Compatible Simulations

A simulation needs to set `supports_batch = True`, implement `torch_batch()`, and return float32 tensors.

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
        # MPS requires float32
        x = torch.rand(n, device=device, generator=generator)
        y = torch.rand(n, device=device, generator=generator)
        inside = (x * x + y * y) <= 1.0
        return 4.0 * inside.float()
```

Key points:

- Use the provided `generator` for all random sampling.
- Return float32 tensors; Metal does not support float64.
- The framework converts float32 to float64 on CPU for statistical precision.
- The same `torch_batch()` works across CPU, MPS, and CUDA.

## Performance

| Feature | MPS (Apple Silicon) | CUDA (NVIDIA) | CPU |
|---------|---------------------|---------------|-----|
| float64 support | Emulated | Native | Native |
| Determinism | Statistical | Bitwise | Bitwise |
| Multi-GPU | Single device | Yes | Multi-core |
| Streams | No | Yes | N/A |
| Unified memory | Yes | No | N/A |
| Batch processing | Single batch | Adaptive | Sequential/Parallel |

MPS is a good fit for Apple Silicon Macs running medium to large workloads, and for development and prototyping where statistical reproducibility is sufficient. It is not suited to exact bitwise reproducibility, float64-heavy numerical work, or multi-GPU scaling.

To compare backends on your Mac:

```python
import time

sim = PiEstimationSimulation()
sim.set_seed(42)

start = time.time()
sim.run(1_000_000, backend="sequential", compute_stats=False)
cpu_time = time.time() - start

start = time.time()
sim.run(1_000_000, backend="torch", torch_device="mps", compute_stats=False)
mps_time = time.time() - start

print(f"MPS speedup: {cpu_time / mps_time:.2f}x")
```

Speedup grows with the amount of computational work per sample and varies by chip generation. Benchmark on your own hardware rather than relying on fixed numbers.

## Architecture

```
sim.run(n, backend="torch", torch_device="mps")
    |
simulation.py        delegates to the TorchBackend factory
    |
backends/torch.py    creates TorchMPSBackend
    |
backends/torch_mps.py    all MPS-specific implementation
```

This keeps `simulation.py` backend-agnostic; MPS-specific complexity is isolated in the backends module.

## Dtype Handling

MPS does not support float64 natively. The framework handles this transparently: the simulation returns a float32 tensor on the MPS device, the backend moves it to CPU and promotes it to float64, and the stats engine receives float64 data.

```python
samples = ...                            # float32 tensor on the MPS device
samples = samples.detach().cpu()         # move to CPU
samples = samples.to(torch.float64)      # promote to float64
return samples.numpy()
```

## Testing

MPS tests skip automatically when no Apple GPU is available:

```bash
pytest tests/test_torch_mps.py -v       # MPS backend tests
pytest tests/test_torch_mps.py -v -rs   # show skip reasons
```

Coverage includes basic functionality and statistical reproducibility, float32 to float64 conversion, stats correctness, confidence-interval coverage, and absence of NaN/Inf values.

## Troubleshooting

**"MPS device requested but not available"**

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
python -c "import torch; print(torch.backends.mps.is_built())"
```

Common causes are running on non-Apple-Silicon hardware, macOS older than 12.3, or a PyTorch build without MPS support. Reinstall PyTorch with:

```bash
pip install --upgrade torch torchvision torchaudio
```

**Results differ between runs**

This is expected with MPS best-effort determinism. Results are statistically equivalent but not bitwise identical. Use the CPU or CUDA backend for exact reproducibility.

**"RuntimeError: Cannot copy out of meta tensor"**

This can occur with complex tensor operations. Simplify the `torch_batch()` implementation, call `.contiguous()` before the operation, or move tensors to CPU for the affected step.

**Slower than expected**

Close memory-intensive apps, check for thermal throttling, and confirm the machine is on AC power, since battery mode may throttle the GPU.

## Float64 Limitations

MPS does not support float64 operations. If a simulation needs float64 precision, use the CPU backend for high-precision work, use CUDA if an NVIDIA GPU is available, or accept the float32 to float64 conversion path (sufficient for most Monte Carlo applications):

```python
result = sim.run(n, backend="sequential")                      # float64 throughout
result = sim.run(n, backend="torch", torch_device="cuda")      # native float64
result = sim.run(n, backend="torch", torch_device="mps")       # float32 to float64
```

## Known Limitations

- No bitwise determinism (statistical reproducibility only).
- No native float64 (the framework handles conversion).
- Single device (no multi-GPU).
- No streams (sequential execution).
- macOS only.

## References

- [Apple Metal Documentation](https://developer.apple.com/metal/)
- [PyTorch MPS Backend](https://pytorch.org/docs/stable/notes/mps.html)
- [Metal Performance Shaders](https://developer.apple.com/documentation/metalperformanceshaders)
- [Apple Silicon Guide](https://developer.apple.com/documentation/apple-silicon)
