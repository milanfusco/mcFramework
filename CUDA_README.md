# CUDA Backend for Monte Carlo Simulations

NVIDIA GPU acceleration for batch Monte Carlo simulations with adaptive memory management and native float64 support.

## Overview

The CUDA backend provides GPU-accelerated batch execution for Monte Carlo simulations using PyTorch's CUDA interface. Unlike the MPS backend (Apple Silicon), CUDA supports native float64 precision and includes several performance optimizations:

- **Adaptive batch sizing** - automatically estimates optimal batch size based on available GPU memory
- **Dual RNG modes** - use PyTorch's generator (default) or cuRAND via CuPy for maximum throughput  
- **CUDA streams** - overlapped execution for better GPU utilization
- **Native float64** - zero conversion overhead compared to MPS backend
- **Memory safety** - probe runs prevent OOM errors on large workloads

## Requirements

- NVIDIA GPU with CUDA support ([compatibility list](https://developer.nvidia.com/cuda-gpus))
- PyTorch with CUDA enabled ([installation guide](https://pytorch.org/get-started/locally/))
- Optional: CuPy for cuRAND mode ([installation](https://docs.cupy.dev/en/stable/install.html))

Check your CUDA availability:
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"
```

## Installation

### Basic GPU Support (Recommended)
```bash
pip install -e ".[gpu]"
```

This installs PyTorch with CUDA support. The default `torch.Generator` RNG mode works out of the box.

### Full CUDA Support (Optional)
For cuRAND mode (native GPU random number generation):
```bash
pip install -e ".[cuda]"
```

Note: CuPy requires CUDA toolkit development files. If you get build errors, use prebuilt wheels:
```bash
pip install cupy-cuda12x  # For CUDA 12.x
pip install cupy-cuda11x  # For CUDA 11.x
```

See [CuPy installation docs](https://docs.cupy.dev/en/stable/install.html) for troubleshooting.

## Quick Start

### Simple Usage

```python
from mcframework.sims import PiEstimationSimulation

sim = PiEstimationSimulation()
sim.set_seed(42)

# Run 1M simulations on GPU (uses adaptive batching)
result = sim.run(1_000_000, backend="torch", torch_device="cuda")
print(f"π ≈ {result.mean:.6f} ± {result.std/1000:.6f}")
```

That's it. The backend automatically:
- Estimates optimal batch size based on your GPU memory
- Splits the workload into batches if needed
- Uses CUDA streams for overlapped execution
- Returns float64 results for stats precision

### Advanced Configuration

For fine-grained control, construct the backend directly:

```python
from mcframework.backends import TorchCUDABackend

# Fixed batch size (useful for benchmarking or memory-constrained GPUs)
backend = TorchCUDABackend(
    device_id=0,           # Use first GPU
    batch_size=100_000,    # Fixed batch size
    use_streams=True       # Enable CUDA streams
)

results = backend.run(sim, n_simulations=10_000_000, seed_seq=sim.seed_seq)
```

## Memory Management

The backend uses a two-phase approach to prevent OOM errors:

1. **Probe run** - executes 1,000 samples to estimate per-sample memory usage
2. **Batch calculation** - allocates ~75% of available GPU memory per batch

For very large workloads, the backend automatically splits across multiple batches with progress tracking.

### Manual Batch Size

If you know your memory constraints:
```python
backend = TorchCUDABackend(device_id=0, batch_size=50_000)
```

Or let adaptive batching handle it:
```python
backend = TorchCUDABackend(device_id=0)  # batch_size=None (default)
```

## RNG Modes

### torch.Generator (Default)

Uses PyTorch's Philox counter-based RNG. Fully deterministic and works everywhere:

```python
sim.set_seed(42)
result1 = sim.run(100_000, backend="torch", torch_device="cuda")
result2 = sim.run(100_000, backend="torch", torch_device="cuda")
# Bitwise identical results
```

Docs: [PyTorch Random Sampling](https://pytorch.org/docs/stable/notes/randomness.html)

### cuRAND (Optional)

Native GPU random number generation via CuPy. Potentially faster for RNG-heavy simulations:

```python
backend = TorchCUDABackend(device_id=0, use_curand=True)
results = backend.run(sim, n_simulations=1_000_000, seed_seq=sim.seed_seq)
```

**Requirements:**
- CuPy installed
- Simulation implements `curand_batch()` method (see examples below)

Docs: [cuRAND Library](https://docs.nvidia.com/cuda/curand/index.html), [CuPy Random](https://docs.cupy.dev/en/stable/reference/random.html)

## Implementing GPU-Accelerated Simulations

Your simulation needs two things:

1. Set `supports_batch = True`
2. Implement `torch_batch()` method

### Example: Pi Estimation

```python
from mcframework import MonteCarloSimulation
import torch

class PiEstimationSimulation(MonteCarloSimulation):
    supports_batch = True
    
    def single_simulation(self, _rng=None, **kwargs):
        # NumPy implementation for CPU/parallel backends
        rng = self._rng(_rng, self.rng)
        x, y = rng.random(), rng.random()
        return 4.0 if (x*x + y*y) <= 1.0 else 0.0
    
    def torch_batch(self, n, *, device, generator):
        # GPU-accelerated batch implementation
        x = torch.rand(n, device=device, generator=generator)
        y = torch.rand(n, device=device, generator=generator)
        inside = (x*x + y*y) <= 1.0
        return 4.0 * inside.float()  # float32 is fine, framework handles conversion
```

**Key points:**
- Use the provided `generator` for all random sampling (never `torch.manual_seed()`)
- Return float32 or float64 tensors (float64 preferred for CUDA)
- Framework handles device placement and dtype conversion

### Optional: cuRAND Implementation

```python
def curand_batch(self, n, device_id, rng):
    # CuPy implementation using cuRAND
    import cupy as cp
    x = rng.uniform(0, 1, size=n, dtype=cp.float32)
    y = rng.uniform(0, 1, size=n, dtype=cp.float32)
    inside = (x*x + y*y) <= 1.0
    return 4.0 * inside.astype(cp.float32)
```

## Performance Tips

### CUDA vs MPS vs CPU

CUDA has several advantages over the MPS backend:

| Feature | CUDA (NVIDIA) | MPS (Apple Silicon) | CPU |
|---------|---------------|---------------------|-----|
| float64 support | Native | Emulated | Native |
| Determinism | Bitwise | Statistical | Bitwise |
| Multi-GPU | Yes | Single device | Multi-core |
| Streams | Yes | No | N/A |

### Benchmarking

Compare backends on your hardware:

```python
import time

sim = PiEstimationSimulation()
sim.set_seed(42)

# CPU baseline
start = time.time()
sim.run(1_000_000, backend="sequential", compute_stats=False)
cpu_time = time.time() - start

# CUDA
start = time.time()
sim.run(1_000_000, backend="torch", torch_device="cuda", compute_stats=False)
cuda_time = time.time() - start

print(f"Speedup: {cpu_time/cuda_time:.2f}x")
```

Expected speedup depends on simulation complexity (GPU benefits increase with computational work per sample).

## Multi-GPU Support

Select device via `device_id`:

```python
# Use second GPU
backend = TorchCUDABackend(device_id=1)

# Check available devices
import torch
print(f"Available GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  [{i}] {torch.cuda.get_device_name(i)}")
```

For data parallelism across multiple GPUs, run separate backend instances in parallel (future enhancement).

## Architecture

The implementation maintains clean separation of concerns:

```
User code:
  sim.run(n, backend="torch", torch_device="cuda")
    ↓
simulation.py (unchanged):
  Delegates to TorchBackend factory
    ↓
backends/torch.py:
  Creates TorchCUDABackend with config
    ↓
backends/torch_cuda.py:
  All CUDA-specific implementation
```

This design keeps `simulation.py` backend-agnostic - CUDA complexity is isolated in the backends module.

## Error Handling

The backend includes defensive validation to catch common mistakes:

```python
class BadSimulation(MonteCarloSimulation):
    supports_batch = True  # Claims GPU support
    # But doesn't implement torch_batch()
    
    def single_simulation(self, _rng=None):
        return 1.0

sim = BadSimulation()
sim.run(100, backend="torch", torch_device="cuda")
# NotImplementedError: Simulation 'BadSimulation' has supports_batch = True 
# but does not implement torch_batch() method.
# Either implement torch_batch(n, device, generator) or set 
# use_curand=True with curand_batch() implementation.
```

All error messages include:
- What went wrong
- Which class is affected  
- How to fix it
- Why it's required

## Testing

Run the test suite to verify your CUDA setup:

```bash
# All Torch backend tests (CUDA tests auto-skip if unavailable)
pytest tests/test_torch_backend.py -v

# Only CUDA tests
pytest tests/test_torch_backend.py -k "CUDA" -v

# Show skipped tests with reasons
pytest tests/test_torch_backend.py -v -rs
```

Test coverage includes:
- Validation and error handling (defensive programming)
- Basic functionality and determinism
- Adaptive batch sizing
- Memory management and leak detection
- CUDA streams
- Multi-device support
- Performance benchmarks

## Troubleshooting

### "CUDA device requested but not available"

Check PyTorch CUDA installation:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If False, reinstall PyTorch with CUDA support: https://pytorch.org/get-started/locally/

### "CUDA out of memory"

Reduce batch size:
```python
backend = TorchCUDABackend(device_id=0, batch_size=10_000)
```

Or let adaptive batching handle it (should work automatically).

### CuPy build errors

Install prebuilt wheels instead of building from source:
```bash
pip install cupy-cuda12x  # Match your CUDA version
```

See https://docs.cupy.dev/en/stable/install.html

### Determinism not working

Verify you're setting the seed:
```python
sim.set_seed(42)  # Required for reproducibility
result = sim.run(...)
```

## Implementation Details

Files modified:
- `src/mcframework/backends/torch_cuda.py` - Complete CUDA backend (~640 lines)
- `src/mcframework/backends/torch_base.py` - cuRAND generator utilities
- `src/mcframework/backends/torch.py` - Factory accepts CUDA kwargs
- `src/mcframework/simulation.py` - Updated docstring (device-specific dtype policy)
- `tests/test_torch_backend.py` - 30+ CUDA tests with skipif guards
- `pyproject.toml` - Optional `cuda` dependency group

No changes to core simulation logic - clean architectural separation maintained.

## References

- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [PyTorch CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html)
- [CuPy Documentation](https://docs.cupy.dev/en/stable/)
- [cuRAND Library](https://docs.nvidia.com/cuda/curand/index.html)
- [PyTorch Random Sampling](https://pytorch.org/docs/stable/notes/randomness.html)

## License

Same as parent project (MIT).
