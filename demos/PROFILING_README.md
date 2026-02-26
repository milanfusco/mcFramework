# PyTorch Profiler Integration

This document explains how to use the PyTorch profiler with McFramework to analyze performance.

## Quick Start

Run the demo:

```bash
cd demos
python demo_profiling.py
```

## What Gets Profiled

The profiler captures:

- **CPU/GPU time**: Time spent on each operation
- **Memory usage**: Allocations and deallocations
- **FLOPs**: Floating-point operations count
- **Tensor shapes**: Input/output dimensions
- **Call stacks**: Function call hierarchy (optional)

## Usage Patterns

### Pattern 1: Wrap a Backend (Recommended)

```python
from mcframework.backends import TorchCPUBackend
from mcframework.profiling import ProfiledTorchBackend, TorchProfilerConfig
from mcframework.sims import PiEstimationSimulation

# Create simulation
sim = PiEstimationSimulation()
sim.set_seed(42)

# Create and wrap backend
base_backend = TorchCPUBackend()
config = TorchProfilerConfig(
    profile_memory=True,
    output_dir="./my_profiling"
)
profiled_backend = ProfiledTorchBackend(base_backend, config)

# Run normally - profiling happens automatically
results = profiled_backend.run(
    sim,
    n_simulations=100_000,
    seed_seq=np.random.SeedSequence(42)
)
```

### Pattern 2: Context Manager

```python
from mcframework.profiling import profile_simulation, TorchProfilerConfig

config = TorchProfilerConfig(output_dir="./profiling")
with profile_simulation(config, name="my_simulation"):
    # Your code here
    results = backend.run(sim, 100_000, seed_seq)
```

## Configuration Options

```python
config = TorchProfilerConfig(
    # Activities to profile
    activities=["cpu"],           # Or ["cpu", "cuda"] for GPU
    
    # Profiling schedule
    schedule_wait=0,              # Wait steps before profiling
    schedule_warmup=1,            # Warmup steps
    schedule_active=3,            # Active profiling steps per cycle
    schedule_repeat=2,            # Number of cycles (= number of traces)
    
    # Data collection
    record_shapes=True,           # Record tensor shapes
    profile_memory=True,          # Track memory
    with_stack=False,             # Stack traces (slow)
    with_flops=True,              # Calculate FLOPs
    
    # Output
    output_dir="./profiler_results",
    export_chrome_trace=True,     # Chrome trace JSON
    export_stacks=False,          # Stack trace files
)
```

## Understanding the Profiler Schedule

The PyTorch profiler works in cycles defined by the schedule:

- **wait**: Steps to skip (no profiling)
- **warmup**: Steps to warm up (profiling but no trace saved)
- **active**: Steps to actively profile (data collected)
- **repeat**: Number of cycles to run

Each `repeat` cycle generates **one trace file**. So:

- `repeat=1` → 1 trace file
- `repeat=2` → 2 trace files (default)
- `repeat=3` → 3 trace files

**Example**: With `wait=0, warmup=1, active=3, repeat=2`:

- Cycle 1: Steps 0-3 (warmup step 0, active steps 1-3) → `trace_4.json`
- Cycle 2: Steps 4-7 (warmup step 4, active steps 5-7) → `trace_8.json`

By default, simulations are **chunked** to ensure enough steps for all cycles.

## Viewing Results

### Console Output

The profiler prints a table to the console:

```
-----------------  ------------  ------------  ------------  
Name               Self CPU %    Self CPU      CPU total    
-----------------  ------------  ------------  ------------  
aten::normal_      45.2%         150.000us     150.000us    
aten::randn        32.1%         120.000us     120.000us    
...
```

### Chrome Trace Viewer

1. Open Chrome browser
2. Navigate to: `chrome://tracing`
3. Click **Load** button
4. Select a `trace_*.json` file from your output directory
5. Explore the interactive timeline!

The Chrome trace shows:

- Timeline of all operations
- GPU/CPU utilization
- Memory allocations over time
- Operation dependencies

### MPS Metal Traces

To view MPS Metal traces, set `enable_mps_profiler=True` in `TorchProfilerConfig` and run the simulation.
Then, use Xcode Instruments to view the traces.

- Open XCode, then select XCode -> Developer Tools -> Instruments
- Select the "Metal System Trace" instrument template
- Target the process to profile (`<path to venv>/bin/python`)
- Set the argument to the script to profile (e.g. `demos/demo_profiling.py`)
- Ensure that `MTL_CAPTURE_ENABLED=1` is set in the environment variables.
- Set the working directory to the directory containing the script to profile (e.g. `demos`).
- Click the "Record" button to start profiling.

## Important Notes

### Current Limitations

- **Batch Execution Required**: Only simulations with `supports_batch = True` work with Torch backends

### Performance Tips

1. **Start with defaults**: Use `TorchProfilerConfig()` first
2. **Disable stack traces**: Keep `with_stack=False` unless debugging
3. **Limit active steps**: Use `schedule_active=3-5` to reduce overhead
4. **Memory profiling**: Enable `profile_memory=True` to find memory bottlenecks
5. **Compare devices**: Profile cpu vs mps/cuda to see acceleration gains

## Examples

### Example 1: Basic Profiling

```python
import numpy as np
from mcframework.backends import TorchCPUBackend
from mcframework.profiling import ProfiledTorchBackend
from mcframework.sims import PiEstimationSimulation

sim = PiEstimationSimulation()
backend = ProfiledTorchBackend(TorchCPUBackend())

results = backend.run(
    sim,
    n_simulations=100_000,
    seed_seq=np.random.SeedSequence(42)
)
# Check ./profiler_results/ for output
```

### Example 2: Compare CPU vs MPS

```python
from mcframework.backends import TorchBackend
from mcframework.profiling import ProfiledTorchBackend, TorchProfilerConfig

sim = PiEstimationSimulation()

# Profile CPU
cpu_config = TorchProfilerConfig(output_dir="./profile_cpu")
cpu_backend = ProfiledTorchBackend(
    TorchBackend(device="cpu"),
    cpu_config
)
cpu_results = cpu_backend.run(sim, 500_000, seed_seq)

# Profile MPS
mps_config = TorchProfilerConfig(output_dir="./profile_mps")
mps_backend = ProfiledTorchBackend(
    TorchBackend(device="mps"),
    mps_config
)
mps_results = mps_backend.run(sim, 500_000, seed_seq)

# Compare the traces in Chrome!
```

### Example 3: Detailed Memory Profiling

```python
config = TorchProfilerConfig(
    schedule_active=10,        # Profile more steps
    profile_memory=True,       # Memory tracking
    with_stack=True,           # Stack traces for memory leaks
    output_dir="./memory_profile"
)

backend = ProfiledTorchBackend(TorchCPUBackend(), config)
results = backend.run(sim, 1_000_000, seed_seq)
```

## Troubleshooting

### Issue: "Simulation does not support Torch batch execution"

**Solution**: Only `PiEstimationSimulation` currently supports `torch_batch()`. Use this simulation, or add `supports_batch = True` and implement `torch_batch()` for your custom simulation.

### Issue: Profiler output is empty

**Solution**: Increase `schedule_active` to capture more steps:

```python
config = TorchProfilerConfig(schedule_active=5)
```

### Issue: Profiling is very slow

**Solution**: Disable stack traces and reduce active steps:

```python
config = TorchProfilerConfig(
    with_stack=False,
    schedule_active=3
)
```

## Architecture

The profiler is implemented as a **wrapper pattern** that:

1. Wraps any existing Torch backend (CPU, MPS, CUDA)
2. Adds profiling instrumentation without modifying core logic
3. Outputs results to configurable directories
4. Supports multiple output formats (console, JSON, stacks)

This keeps profiling code completely separate from simulation logic.

## Further Reading

- [PyTorch Profiler Documentation](https://pytorch.org/docs/stable/profiler.html)
- [Chrome Tracing Documentation](https://www.chromium.org/developers/how-tos/trace-event-profiling-tool/)
- [XCode Instruments Documentation](https://developer.apple.com/library/archive/documentation/AnalysisTools/Conceptual/instruments_help-collection/Chapter/Chapter.html)
- [McFramework Backends](../src/mcframework/backends/)
