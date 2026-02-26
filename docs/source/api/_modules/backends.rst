Backends Module
===============

.. currentmodule:: mcframework.backends

The ``backends`` module provides pluggable execution strategies for Monte Carlo
simulations, from single-threaded CPU to GPU-accelerated batch processing.

.. contents:: On This Page
   :local:
   :depth: 2

----

Overview
--------

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Backend
     - Class
     - Use Case
   * - **Sequential**
     - :class:`~mcframework.backends.SequentialBackend`
     - Single-threaded, debugging, small jobs
   * - **Thread**
     - :class:`~mcframework.backends.ThreadBackend`
     - NumPy-heavy code (releases GIL)
   * - **Process**
     - :class:`~mcframework.backends.ProcessBackend`
     - Python-bound code, Windows
   * - **Torch CPU**
     - :class:`~mcframework.backends.TorchCPUBackend`
     - Vectorized CPU batch execution
   * - **Torch MPS**
     - :class:`~mcframework.backends.TorchMPSBackend`
     - Apple Silicon GPU (M1/M2/M3/M4)
   * - **Torch CUDA**
     - :class:`~mcframework.backends.TorchCUDABackend`
     - NVIDIA GPU acceleration

----

Quick Start
-----------

**CPU Backends:**

.. code-block:: python

   from mcframework import PiEstimationSimulation

   sim = PiEstimationSimulation()
   sim.set_seed(42)

   # Sequential (single-threaded)
   result = sim.run(10_000, backend="sequential")

   # Thread-based parallelism (default on POSIX)
   result = sim.run(100_000, backend="thread", n_workers=8)

   # Process-based parallelism (default on Windows)
   result = sim.run(100_000, backend="process", n_workers=4)

   # Auto-selection based on platform and job size
   result = sim.run(100_000, backend="auto")

**GPU Backends (requires PyTorch):**

.. code-block:: python

   # Torch CPU (vectorized, no GPU required)
   result = sim.run(1_000_000, backend="torch", torch_device="cpu")

   # Apple Silicon GPU (M1/M2/M3/M4 Macs)
   result = sim.run(1_000_000, backend="torch", torch_device="mps")

   # NVIDIA CUDA GPU
   result = sim.run(1_000_000, backend="torch", torch_device="cuda")

----

CPU Backends
------------

SequentialBackend
~~~~~~~~~~~~~~~~~

Single-threaded execution for debugging and small jobs.

.. autosummary::
   :toctree: generated
   :nosignatures:

   SequentialBackend

**When to use:**

- Debugging and testing
- Jobs with < 20,000 simulations
- When reproducibility debugging is needed

.. code-block:: python

   result = sim.run(1000, backend="sequential")

ThreadBackend
~~~~~~~~~~~~~

Thread-based parallelism using :class:`~concurrent.futures.ThreadPoolExecutor`.

.. autosummary::
   :toctree: generated
   :nosignatures:

   ThreadBackend

**When to use:**

- NumPy-heavy code that releases the GIL
- POSIX systems (macOS, Linux)
- When process spawn overhead is significant

.. code-block:: python

   result = sim.run(100_000, backend="thread", n_workers=8)

ProcessBackend
~~~~~~~~~~~~~~

Process-based parallelism using :class:`~concurrent.futures.ProcessPoolExecutor`.

.. autosummary::
   :toctree: generated
   :nosignatures:

   ProcessBackend

**When to use:**

- Python-bound code that doesn't release the GIL
- Windows (threads serialize under GIL)
- CPU-intensive pure Python calculations

.. code-block:: python

   result = sim.run(100_000, backend="process", n_workers=4)

----

Torch GPU Backends
------------------

The Torch backends enable GPU-accelerated batch execution for simulations that
implement the :meth:`~mcframework.core.MonteCarloSimulation.torch_batch` method.

.. note::

   **Installation:** GPU backends require PyTorch. Install with:

   .. code-block:: bash

      pip install mcframework[gpu]

TorchBackend (Unified)
~~~~~~~~~~~~~~~~~~~~~~

Factory class that auto-selects the appropriate device-specific backend.

.. autosummary::
   :toctree: generated
   :nosignatures:

   TorchBackend

**Usage:**

.. code-block:: python

   from mcframework.backends import TorchBackend

   # Auto-creates TorchCPUBackend
   backend = TorchBackend(device="cpu")

   # Auto-creates TorchMPSBackend (Apple Silicon)
   backend = TorchBackend(device="mps")

   # Auto-creates TorchCUDABackend (NVIDIA)
   backend = TorchBackend(device="cuda")

   # Run simulation
   results = backend.run(sim, n_simulations=1_000_000, seed_seq=sim.seed_seq)

TorchCPUBackend
~~~~~~~~~~~~~~~

Vectorized batch execution on CPU using PyTorch :class:`~torch.Tensor`.

.. autosummary::
   :toctree: generated
   :nosignatures:

   TorchCPUBackend

**When to use:**

- Baseline testing before GPU deployment
- Systems without GPU acceleration
- Debugging vectorized code
- Small to medium simulation sizes

.. code-block:: python

   from mcframework.backends import TorchCPUBackend

   backend = TorchCPUBackend()
   results = backend.run(sim, 100_000, sim.seed_seq, progress_callback=None)

TorchMPSBackend
~~~~~~~~~~~~~~~

Apple Silicon GPU acceleration via Metal Performance Shaders (MPS).

.. autosummary::
   :toctree: generated
   :nosignatures:

   TorchMPSBackend

**Requirements:**

- macOS 12.3+ with Apple Silicon (M1/M2/M3/M4)
- PyTorch with MPS support

**Dtype Policy:**

Metal Performance Shaders only supports up to `float32 <https://developer.apple.com/documentation/metalperformanceshaders/mpsdatatype/float32>`_ on GPU. 
Therefore, the framework promotes the results to `float64 <https://docs.pytorch.org/docs/stable/generated/torch.Tensor.double.html#torch.Tensor.double>`_ on CPU (see :meth:`~torch.Tensor.to`) 
for stats engine precision.

.. warning::

   **MPS Determinism Caveat**

   Apple's documentation confirms the lack of float64 support: `MPSDataType <https://developer.apple.com/documentation/metalperformanceshaders/mpsdatatype>`_. 
   
   Also, other issues on other projects have reported a similar problem:

   - `Apple Forums thread <https://developer.apple.com/forums/thread/797778>`_
   - `PyTorch Discuss thread <https://discuss.pytorch.org/t/apple-m1max-pytorch-error-typeerror-cannot-convert-a-mps-tensor-to-float64-dtype-as-the-mps-framework-doesnt-support-float64-please-use-float32-instead/196669>`_ 
   - `PyTorch Lightning GitHub issue <https://github.com/Lightning-AI/pytorch-lightning/issues/21261>`_

   Torch MPS preserves RNG stream structure but does not guarantee bitwise
   reproducibility due to Metal backend scheduling and float32 arithmetic.
   Statistical properties (mean, variance, CI coverage) remain correct. 
   (see ``TestMPSDeterminism`` in ``tests/test_torch_backend.py`` for actual tests)

.. code-block:: python

   from mcframework.backends import TorchMPSBackend, is_mps_available

   if is_mps_available():
       backend = TorchMPSBackend()
       results = backend.run(sim, 1_000_000, sim.seed_seq, None)

TorchCUDABackend
~~~~~~~~~~~~~~~~

NVIDIA GPU acceleration with adaptive batching and CUDA streams.

.. autosummary::
   :toctree: generated
   :nosignatures:

   TorchCUDABackend

**Features:**

- Adaptive batch sizing based on GPU memory
- CUDA stream support for async execution
- Native float64 support (no precision loss)
- Optional cuRAND integration for maximum performance

**Configuration Options:**

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``device_id``
     - 0
     - CUDA device index for multi-GPU systems
   * - ``use_curand``
     - False
     - Use cuRAND instead of torch.Generator
   * - ``batch_size``
     - None
     - Fixed batch size (None = adaptive)
   * - ``use_streams``
     - True
     - Enable CUDA streams for async execution

.. code-block:: python

   from mcframework.backends import TorchCUDABackend, is_cuda_available

   if is_cuda_available():
       # Basic usage
       backend = TorchCUDABackend()

       # Advanced configuration
       backend = TorchCUDABackend(
           device_id=0,
           use_curand=False,
           batch_size=None,  # Adaptive
           use_streams=True,
       )

       results = backend.run(sim, 10_000_000, sim.seed_seq, progress_callback)

----

Implementing Torch Support
--------------------------

To enable GPU acceleration for your simulation, implement :meth:`~mcframework.core.MonteCarloSimulation.torch_batch`:

.. code-block:: python

   from mcframework import MonteCarloSimulation

   class MySimulation(MonteCarloSimulation):
       supports_batch = True  # Required flag

       def single_simulation(self, _rng=None, **kwargs):
           rng = self._rng(_rng, self.rng)
           return float(rng.normal())

       def torch_batch(self, n, *, device, generator):
           """Vectorized Torch implementation."""
           import torch

           # Use explicit generator for reproducibility
           samples = torch.randn(n, device=device, generator=generator)

           # Return float32 for MPS compatibility
           # Framework promotes to float64 on CPU
           return samples.float()

**Key Requirements:**

1. Set ``supports_batch = True`` as a class attribute
2. All random sampling must use the provided ``generator``
3. Never use global RNG (``torch.manual_seed()``)
4. Return float32 for MPS compatibility

----

RNG Architecture
----------------

The framework uses explicit PyTorch :class:`~torch.Generator` objects seeded from NumPy's
:class:`~numpy.random.SeedSequence` to maintain reproducible parallel streams:

.. code-block:: python

   from mcframework.backends import make_torch_generator
   import numpy as np

   # Create seed sequence
   seed_seq = np.random.SeedSequence(42)

   # Create explicit generator (spawns child seed)
   generator = make_torch_generator(torch.device("cpu"), seed_seq)

   # Use in sampling
   samples = torch.rand(1000, generator=generator)

**Why explicit generators?**

- :func:`~torch.manual_seed` is global state that breaks parallel composition
- Explicit generators enable deterministic multi-stream MC
- Mirrors NumPy's :meth:`~numpy.random.SeedSequence.spawn` semantics

----

Utility Functions
-----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   make_blocks
   worker_run_chunk
   is_windows_platform
   validate_torch_device
   make_torch_generator
   is_mps_available
   is_cuda_available

**Availability Checks:**

.. code-block:: python

   from mcframework.backends import is_mps_available, is_cuda_available

   print(f"MPS available: {is_mps_available()}")
   print(f"CUDA available: {is_cuda_available()}")

**Device Validation:**

.. code-block:: python

   from mcframework.backends import validate_torch_device

   validate_torch_device("cpu")   # Always passes
   validate_torch_device("mps")   # Raises RuntimeError if unavailable
   validate_torch_device("cuda")  # Raises RuntimeError if unavailable

----

Backend Protocol
----------------

All backends implement the :class:`~mcframework.backends.ExecutionBackend` protocol:

.. autosummary::
   :toctree: generated
   :nosignatures:

   ExecutionBackend

**Protocol Definition:**

.. code-block:: python

   from typing import Protocol, Callable
   import numpy as np

   class ExecutionBackend(Protocol):
       def run(
           self,
           sim: "MonteCarloSimulation",
           n_simulations: int,
           seed_seq: np.random.SeedSequence | None,
           progress_callback: Callable[[int, int], None] | None = None,
           **simulation_kwargs,
       ) -> np.ndarray:
           """Execute simulations and return results array."""
           ...

----

.. note::

   Torch backends achieve massive speedups through vectorization, not just
   parallelization. The entire batch executes as tensor operations.

----

Module Reference
----------------

Base Classes and Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: mcframework.backends.base
   :members:
   :undoc-members:
   :show-inheritance:

Sequential Backend
~~~~~~~~~~~~~~~~~~

.. automodule:: mcframework.backends.sequential
   :members:
   :undoc-members:
   :show-inheritance:

Parallel Backends
~~~~~~~~~~~~~~~~~

.. automodule:: mcframework.backends.parallel
   :members:
   :undoc-members:
   :show-inheritance:

Torch Backend (Unified)
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: mcframework.backends.torch
   :members:
   :undoc-members:
   :show-inheritance:

Torch CPU Backend
~~~~~~~~~~~~~~~~~

.. automodule:: mcframework.backends.torch_cpu
   :members:
   :undoc-members:
   :show-inheritance:

Torch MPS Backend (Apple Silicon)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: mcframework.backends.torch_mps
   :members:
   :undoc-members:
   :show-inheritance:

Torch CUDA Backend (NVIDIA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: mcframework.backends.torch_cuda
   :members:
   :undoc-members:
   :show-inheritance:

----

See Also
--------

- :doc:`core` — Base simulation class and framework
- :doc:`stats_engine` — Statistical analysis of results
- ``demos/demo_apple_silicon_benchmark.py`` — Benchmark script for Apple Silicon
