"""
Demo of PyTorch profiler integration with McFramework.

Note: Currently only PiEstimationSimulation supports torch_batch execution.
"""

import numpy as np
from mcframework.backends import TorchCPUBackend, TorchBackend
from mcframework.profiling import ProfiledTorchBackend, TorchProfilerConfig
from mcframework.sims import PiEstimationSimulation


def demo_basic_profiling():
    """Basic profiling with default settings."""
    print("=" * 60)
    print("Demo 1: Basic Profiling (Pi Estimation)")
    print("=" * 60)
    
    sim = PiEstimationSimulation()
    sim.set_seed(42)
    base_backend = TorchCPUBackend()
    
    # Use default config
    profiled_backend = ProfiledTorchBackend(base_backend)
    
    results = profiled_backend.run(
        sim,
        n_simulations=100_000,
        seed_seq=np.random.SeedSequence(42)
    )
    
    print(f"\nPi Estimate: {results.mean():.6f} (True value: 3.141593)")
    print(f"Std: {results.std():.4f}")
    print("Check ./profiler_results/ for trace files\n")


def demo_custom_config():
    """Profiling with custom configuration for multiple traces."""
    print("=" * 60)
    print("Demo 2: Custom Configuration (Multiple Traces)")
    print("=" * 60)
    
    sim = PiEstimationSimulation()
    sim.set_seed(123)
    base_backend = TorchCPUBackend()
    
    # Custom config to generate multiple traces
    config = TorchProfilerConfig(
        schedule_wait=0,        # Start immediately
        schedule_warmup=1,      # 1 warmup step
        schedule_active=3,      # Profile 3 steps per cycle
        schedule_repeat=3,      # Repeat 3 times = 3 traces
        profile_memory=True,    # Track memory usage
        with_flops=True,        # Calculate FLOPs
        with_stack=False,       # Disable stack traces (faster)
        output_dir="./profiling_detailed"
    )
    
    profiled_backend = ProfiledTorchBackend(base_backend, config=config)
    
    results = profiled_backend.run(
        sim,
        n_simulations=500_000_000,  # Larger simulation
        seed_seq=np.random.SeedSequence(123)
    )
    
    print(f"\nPi estimate: {results.mean():.6f}")
    print(f"Error: {abs(results.mean() - np.pi):.6f}")
    
    # Count trace files
    import os
    trace_files = [f for f in os.listdir("./profiling_detailed") if f.startswith("trace_")]
    print(f"Generated {len(trace_files)} trace files in ./profiling_detailed/")
    print(f"Files: {sorted(trace_files)}\n")


def demo_compare_devices():
    """Profile and compare different device backends."""
    print("=" * 60)
    print("Demo 3: Compare CPU vs GPU Backends")
    print("=" * 60)
    
    sim = PiEstimationSimulation()
    sim.set_seed(42)
    n_sims = 500_000_000
    
    # Profile CPU
    print("\nProfiling CPU backend...")
    cpu_backend = TorchBackend(device="cpu")
    cpu_config = TorchProfilerConfig(
        output_dir="./profiling_cpu",
        schedule_active=3
    )
    cpu_profiled = ProfiledTorchBackend(cpu_backend, config=cpu_config)
    
    cpu_results = cpu_profiled.run(
        sim, n_sims, np.random.SeedSequence(42)
    )
    print(f"CPU Result: {cpu_results.mean():.6f}")
    
    # Try MPS if available (Apple Silicon)
    try:
        from mcframework.backends import is_mps_available
        if is_mps_available():
            print("\nProfiling MPS backend (Apple Silicon GPU)...")
            mps_backend = TorchBackend(device="mps")
            mps_config = TorchProfilerConfig(
                output_dir="./profiling_mps",
                schedule_active=3,
                activities=["cpu"],
                enable_mps_profiler=True,
            )
            mps_profiled = ProfiledTorchBackend(mps_backend, config=mps_config)
            
            mps_results = mps_profiled.run(
                sim, n_sims, seed_seq=np.random.SeedSequence(42)
            )
            print(f"MPS Result: {mps_results.mean():.6f}")
            print("\nCompare profiling results:")
            print("  - CPU: ./profiling_cpu/")
            print("  - MPS: ./profiling_mps/")
        else:
            print("\nMPS not available on this system (requires Apple Silicon)")
    except ImportError:
        print("\nMPS backend not available")


def main():
    """Run all profiling demos."""
    print("\n" + "=" * 60)
    print("McFramework PyTorch Profiler Demo")
    print("=" * 60 + "\n")
    
    try:
        demo_basic_profiling()
        demo_custom_config()
        demo_compare_devices()
        
        print("\n" + "=" * 60)
        print("Demos Complete!")
        print("=" * 60)
        print("\nTo view profiling results:")
        print("1. Open Chrome browser")
        print("2. Navigate to: chrome://tracing")
        print("3. Click 'Load' and select a trace_*.json file")
        print("4. Explore the timeline visualization!")

        from mcframework.backends import is_mps_available
        if is_mps_available():
            print ("\n\n" + "=" * 60)
            print("MPS Metal traces are available!")
            print ("=" * 60)
            print("\nTo view MPS Metal traces:")
            print("1. Open XCode, then select XCode -> Developer Tools -> Instruments")
            print("2. Select the 'Metal System Trace' instrument template")
            print("3. Target the process to profile (`<path to venv>/bin/python`)")
            print("4. Set the argument to the script to profile (e.g. `demos/demo_profiling.py`)")
            print("5. Ensure that `MTL_CAPTURE_ENABLED=1` is set in the environment variables.")
            print("6. Set the working directory to the directory containing the script to profile (e.g. `demos`).")
            print("7. Click the 'Record' button to start profiling.")
        
    except Exception as e:
        print(f"\nError running demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()