# MPS Backend for Monte Carlo Simulations

Apple Metal Performance Shaders (MPS) GPU acceleration for batch Monte Carlo simulations on Apple Silicon Macs.

## Overview

The MPS backend provides GPU-accelerated batch execution for Monte Carlo simulations on Apple Silicon (M1/M2/M3/M4) using PyTorch's Metal Performance Shaders interface. This enables significant speedups over CPU execution on modern Macs.

Key characteristics:
- **Apple Silicon only** - M1, M2, M3, M4 series chips
- **Automatic batching** - single-batch execution (no adaptive batching needed)
- **Best-effort determinism** - preserves RNG structure but not bitwise reproducible
- **Float32 requirement** - Metal doesn't support float64 (framework handles conversion)
- **Efficient memory** - unified memory architecture shared between CPU and GPU

## Requirements

- macOS 12.3 (Monterey) or later
- Apple Silicon Mac (M1/M2/M3/M4 series)
- PyTorch with MPS support ([installation guide](https://pytorch.org/get-started/locally/))

Check your MPS availability:
```bash
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available() and torch.backends.mps.is_built()}')"
```

## Installation

```bash
pip install -e ".[gpu]"
```

This installs PyTorch with MPS support. The backend is automatically available on compatible systems.

## Quick Start

### Simple Usage

```python
from mcframework.sims import PiEstimationSimulation

sim = PiEstimationSimulation()
sim.set_seed(42)

# Run 1M simulations on Apple GPU
result = sim.run(1_000_000, backend="torch", torch_device="mps")
print(f"π ≈ {result.mean:.6f} ± {result.std/1000:.6f}")
```

The backend automatically:
- Executes simulation on Apple GPU
- Handles float32 → float64 conversion for stats precision
- Uses unified memory for efficient CPU ↔ GPU transfers

### Direct Backend Construction

```python
from mcframework.backends import TorchMPSBackend

backend = TorchMPSBackend()
results = backend.run(sim, n_simulations=1_000_000, seed_seq=sim.seed_seq)
```

## Memory Management

MPS uses Apple's unified memory architecture - GPU and CPU share the same physical memory. This means:

- **No explicit memory management needed** - automatic overflow to system RAM
- **Efficient transfers** - zero-copy between CPU and GPU in many cases
- **Less OOM risk** - compared to discrete GPUs with fixed memory

The backend processes simulations in a single batch, leveraging Metal's memory management.

## Determinism Considerations

**Important**: MPS provides **best-effort** determinism, not bitwise reproducibility.

### What This Means

```python
sim.set_seed(42)
result1 = sim.run(100_000, backend="torch", torch_device="mps")
result2 = sim.run(100_000, backend="torch", torch_device="mps")

# Statistical properties preserved:
assert abs(result1.mean - result2.mean) < 0.01  # Very close
assert abs(result1.std - result2.std) < 0.01    # Very close

# But NOT bitwise identical:
assert not np.array_equal(result1.results, result2.results)  # May differ slightly
```

### Why?

- Metal backend scheduling variations
- float32 arithmetic rounding differences  
- GPU kernel execution order non-determinism

### Practical Impact

For Monte Carlo simulations, this is usually fine:
- Means and variances are statistically equivalent
- Confidence intervals have correct coverage
- Results are scientifically reproducible

For exact bitwise reproducibility, use CPU or CUDA backends.

Docs: [PyTorch MPS Notes](https://pytorch.org/docs/stable/notes/mps.html)

## Implementing MPS-Compatible Simulations

Your simulation needs:

1. Set `supports_batch = True`
2. Implement `torch_batch()` method
3. **Return float32 tensors** (MPS requirement)

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
        # GPU-accelerated implementation
        # IMPORTANT: MPS requires float32
        x = torch.rand(n, device=device, generator=generator)
        y = torch.rand(n, device=device, generator=generator)
        inside = (x*x + y*y) <= 1.0
        return 4.0 * inside.float()  # float32 - framework converts to float64
```

**Key points:**
- Use the provided `generator` for all random sampling
- Return float32 tensors (Metal limitation - no float64 support)
- Framework automatically converts float32 → float64 on CPU for stats precision
- Works across CPU, MPS, and CUDA with same code

## Performance Comparison

### MPS vs CUDA vs CPU

| Feature | MPS (Apple Silicon) | CUDA (NVIDIA) | CPU |
|---------|---------------------|---------------|-----|
| float64 support | Emulated | Native | Native |
| Determinism | Statistical | Bitwise | Bitwise |
| Multi-GPU | Single device | Yes | Multi-core |
| Streams | No | Yes | N/A |
| Unified memory | Yes | No | N/A |
| Batch processing | Single batch | Adaptive | Sequential/Parallel |

### When to Use MPS

**Good for:**
- Apple Silicon Macs (M1/M2/M3/M4)
- Medium to large workloads (10k+ simulations)
- Development and prototyping
- When statistical reproducibility is sufficient

**Not ideal for:**
- Exact bitwise reproducibility requirements
- float64-heavy numerical computations
- Multi-GPU scaling needs

### Benchmarking

Compare backends on your Mac:

```python
import time

sim = PiEstimationSimulation()
sim.set_seed(42)

# CPU baseline
start = time.time()
sim.run(1_000_000, backend="sequential", compute_stats=False)
cpu_time = time.time() - start

# MPS
start = time.time()
sim.run(1_000_000, backend="torch", torch_device="mps", compute_stats=False)
mps_time = time.time() - start

print(f"MPS Speedup: {cpu_time/mps_time:.2f}x")
```

Expected speedup varies by chip generation:
- M1: 5-15x vs CPU
- M2: 8-20x vs CPU
- M3/M4: 10-25x vs CPU

Speedup increases with simulation complexity.

## Architecture

MPS backend follows the same clean architecture as CUDA:

```
User code:
  sim.run(n, backend="torch", torch_device="mps")
    ↓
simulation.py (unchanged):
  Delegates to TorchBackend factory
    ↓
backends/torch.py:
  Creates TorchMPSBackend
    ↓
backends/torch_mps.py:
  All MPS-specific implementation
```

Zero changes to `simulation.py` - MPS complexity is isolated in the backends module.

## Dtype Handling

MPS doesn't support float64 natively. The framework handles this transparently:

```python
# In simulation's torch_batch():
samples = ...  # float32 tensor on MPS device

# Framework's MPS backend:
samples = samples.detach().cpu()        # Move to CPU
samples = samples.to(torch.float64)     # Promote to float64
return samples.numpy()                  # Return as NumPy array
```

Stats engine receives float64 data, ensuring precision for statistical computations.

## Error Handling

The backend includes defensive validation:

```python
class BadSimulation(MonteCarloSimulation):
    supports_batch = True
    # Missing torch_batch implementation
    
    def single_simulation(self, _rng=None):
        return 1.0

sim = BadSimulation()
sim.run(100, backend="torch", torch_device="mps")
# NotImplementedError: Simulation 'BadSimulation' has supports_batch = True
# but does not implement torch_batch() method.
```

All errors include clear remediation steps.

## Testing

Run the test suite:

```bash
# All Torch backend tests (MPS tests auto-skip if unavailable)
pytest tests/test_torch_backend.py -v

# Only MPS tests
pytest tests/test_torch_backend.py -k "MPS" -v

# Show skipped tests
pytest tests/test_torch_backend.py -v -rs
```

Test coverage includes:
- Basic functionality and statistical reproducibility
- Float32 → float64 conversion
- Stats computation correctness
- Confidence interval coverage
- No NaN/Inf generation
- Execution time bounds

## Troubleshooting

### "MPS device requested but not available"

Check requirements:
```bash
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
python -c "import torch; print(f'MPS built: {torch.backends.mps.is_built()}')"
```

**Common causes:**
- Not running on Apple Silicon Mac
- macOS < 12.3
- PyTorch not built with MPS support

**Fix**: Reinstall PyTorch with MPS support:
```bash
pip install --upgrade torch torchvision torchaudio
```

See: https://pytorch.org/get-started/locally/

### Results differ between runs

This is expected with MPS (best-effort determinism). Results are statistically equivalent but not bitwise identical.

For exact reproducibility, use CPU or CUDA backends.

### "RuntimeError: Cannot copy out of meta tensor"

This can happen with complex tensor operations. Workarounds:
1. Simplify `torch_batch()` implementation
2. Use explicit `.contiguous()` before operations
3. Move tensors to CPU before complex operations

### Performance slower than expected

Check system:
- Close memory-intensive apps
- Ensure thermal throttling isn't occurring
- Try smaller batch sizes if swapping to disk
- Verify you're on AC power (battery mode may throttle GPU)

## Float64 Limitations

MPS doesn't support float64 operations. If your simulation needs float64 precision:

**Option 1**: Use CPU backend for high-precision work
```python
result = sim.run(n, backend="sequential")  # float64 throughout
```

**Option 2**: Use CUDA if you have access to NVIDIA GPU
```python
result = sim.run(n, backend="torch", torch_device="cuda")  # Native float64
```

**Option 3**: Accept float32 → float64 conversion (usually fine)
```python
result = sim.run(n, backend="torch", torch_device="mps")  # float32 → float64
```

For most Monte Carlo applications, the float32 → float64 conversion path is sufficient.

## Implementation Details

Files implementing MPS backend:
- `src/mcframework/backends/torch_mps.py` - MPS backend (~256 lines)
- `src/mcframework/backends/torch_base.py` - Shared Torch utilities
- `src/mcframework/backends/torch.py` - Factory that selects backend
- `src/mcframework/simulation.py` - Device-specific dtype docs
- `tests/test_torch_backend.py` - 10+ MPS tests with skipif guards

No changes to core simulation logic.

## Apple Silicon Performance Tips

### M1/M2 (8-core GPU)
- Optimal for workloads: 10k - 1M simulations
- Speedup: 5-15x over CPU

### M3/M4 (10+ core GPU)  
- Optimal for workloads: 10k - 10M simulations
- Speedup: 10-25x over CPU

### Memory Considerations
- Unified memory is shared with system
- Close background apps for maximum available memory
- 8GB Macs: aim for < 5M simulations per batch
- 16GB+ Macs: can handle 10M+ simulations

### Power Efficiency
MPS backend is significantly more power-efficient than CPU for large workloads - ideal for laptop use.

## References

- [Apple Metal Documentation](https://developer.apple.com/metal/)
- [PyTorch MPS Backend](https://pytorch.org/docs/stable/notes/mps.html)
- [Metal Performance Shaders](https://developer.apple.com/documentation/metalperformanceshaders)
- [Apple Silicon Guide](https://developer.apple.com/documentation/apple-silicon)

## Known Limitations

1. **No bitwise determinism** - statistical reproducibility only
2. **float64 not supported** - Metal limitation, framework handles conversion
3. **Single device** - can't split across multiple GPUs
4. **No streams** - executes sequentially
5. **macOS only** - platform limitation

For applications requiring exact determinism or float64 precision, use CPU or CUDA backends.

## License

Same as parent project (MIT).

