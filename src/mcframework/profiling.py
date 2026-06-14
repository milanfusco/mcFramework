"""
PyTorch profiler integration for performance analysis.

This module provides profiling capabilities for Torch backends without
modifying the core simulation logic.
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .simulation import MonteCarloSimulation

logger = logging.getLogger(__name__)

__all__ = [
    "TorchProfilerConfig",
    "ProfiledTorchBackend",
    "profile_simulation",
]


class TorchProfilerConfig:  # pragma: no cover
    """
    Configuration for PyTorch profiler.

    Parameters
    ----------
    activities : list[str], optional
        List of activities to profile. Options: ["cpu", "cuda"].
        Defaults to ["cpu"].
    schedule_wait : int, default 0
        Number of steps to skip before profiling starts.
    schedule_warmup : int, default 1
        Number of warmup steps (profiling but no trace saved).
    schedule_active : int, default 3
        Number of active profiling steps per cycle.
    schedule_repeat : int, default 2
        Number of times to repeat the schedule. Each repeat generates one trace.
        With repeat=2, you get 2 trace files total.
    record_shapes : bool, default True
        Record tensor shapes.
    profile_memory : bool, default True
        Profile memory usage.
    with_stack : bool, default False
        Record stack traces (slower but more detailed).
    with_flops : bool, default True
        Calculate FLOPs estimates.
    output_dir : str, default "./profiler_results"
        Directory to save profiler output.
    export_chrome_trace : bool, default True
        Export Chrome trace format.
    export_stacks : bool, default False
        Export stack traces.
    """

    def __init__(
        self,
        activities: list[str] | None = None,
        schedule_wait: int = 0,
        schedule_warmup: int = 1,
        schedule_active: int = 3,
        schedule_repeat: int = 2,
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = False,
        with_flops: bool = True,
        output_dir: str = "./profiler_results",
        export_chrome_trace: bool = True,
        export_stacks: bool = False,
        acc_events: bool = True,
        enable_mps_profiler: bool = False,
    ):
        self.activities = activities or ["cpu"]
        self.schedule_wait = schedule_wait
        self.schedule_warmup = schedule_warmup
        self.schedule_active = schedule_active
        self.schedule_repeat = schedule_repeat
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_flops = with_flops
        self.output_dir = Path(output_dir)
        self.export_chrome_trace = export_chrome_trace
        self.export_stacks = export_stacks
        self.acc_events = acc_events
        self.enable_mps_profiler = enable_mps_profiler


class ProfiledTorchBackend:  # pragma: no cover
    """
    Wrapper that adds PyTorch profiling to any Torch backend.

    This class wraps an existing Torch backend (CPU, MPS, CUDA) and adds
    profiling capabilities without modifying the underlying implementation.

    By default, the profiler splits simulations into chunks to generate
    multiple traces, allowing the profiler schedule (wait/warmup/active/repeat)
    to work properly. Each chunk generates a separate trace file.

    Parameters
    ----------
    backend : TorchBackend or device-specific backend
        The torch backend to wrap (TorchCPUBackend, TorchMPSBackend, etc.).
    config : TorchProfilerConfig, optional
        Profiler configuration. If None, uses defaults.
    enable_chunking : bool, default True
        If True, splits simulations into chunks to generate multiple traces.
        Set to False to run all simulations at once (generates only one trace).
    chunk_size : int, optional
        Size of each chunk. If None, automatically calculated based on
        total simulations and profiler schedule to ensure sufficient traces.

    Notes
    -----
    **Chunking behavior**: With chunking enabled (default), a simulation of
    N samples with ``schedule_active=3`` will be split into at least 3+
    chunks, generating 3+ trace files. Without chunking, only one trace
    is generated regardless of the schedule.

    Examples
    --------
    >>> from mcframework.backends import TorchCPUBackend
    >>> from mcframework.profiling import ProfiledTorchBackend, TorchProfilerConfig
    >>>
    >>> # Create base backend
    >>> base_backend = TorchCPUBackend()
    >>>
    >>> # Wrap with profiler (chunking enabled by default)
    >>> config = TorchProfilerConfig(
    ...     schedule_active=5,
    ...     output_dir="./my_profiling",
    ...     profile_memory=True
    ... )
    >>> backend = ProfiledTorchBackend(base_backend, config)
    >>>
    >>> # Run - generates multiple trace files
    >>> results = backend.run(sim, n_simulations=100000, seed_seq=seed_seq)  # doctest: +SKIP
    >>>
    >>> # Disable chunking for single trace
    >>> backend_no_chunk = ProfiledTorchBackend(
    ...     base_backend, config, enable_chunking=False
    ... )  # doctest: +SKIP
    """

    def __init__(
        self,
        backend: Any,
        config: TorchProfilerConfig | None = None,
        enable_chunking: bool = True,
        chunk_size: int | None = None,
    ):
        """
        Initialize profiled backend.

        Parameters
        ----------
        backend : Any
            The torch backend to wrap.
        config : TorchProfilerConfig, optional
            Profiler configuration.
        enable_chunking : bool, default True
            If True, split simulations into chunks to generate multiple traces.
            This allows the profiler schedule to work properly.
        chunk_size : int, optional
            Size of each chunk. If None, automatically determined based on
            total simulations and profiler schedule.
        """
        try:
            import torch  # pylint: disable=import-outside-toplevel
            import torch.profiler  # pylint: disable=import-outside-toplevel
        except ImportError as err:
            raise ImportError(
                "PyTorch profiler requires PyTorch. Install with: pip install mcframework[torch]"
            ) from err

        self._backend = backend
        self._config = config or TorchProfilerConfig()
        self._torch = torch
        self._enable_chunking = enable_chunking
        self._chunk_size = chunk_size

        # Create output directory
        self._config.output_dir.mkdir(parents=True, exist_ok=True)

    def _create_profiler(self) -> Any:
        """Create and configure PyTorch profiler."""
        activities = []

        for a in self._config.activities:
            if a == "cpu":
                activities.append(self._torch.profiler.ProfilerActivity.CPU)

            elif a == "cuda":
                if self._torch.cuda.is_available():
                    activities.append(self._torch.profiler.ProfilerActivity.CUDA)
                else:
                    logger.warning("CUDA requested but not available with %s", self._backend.device)

            elif a == "mps":
                # MPS is NOT supported by torch.profiler activities
                if not self._config.enable_mps_profiler:
                    raise ValueError(
                        "MPS activity requested, but torch.profiler has no MPS backend. "
                        "Use enable_mps_profiler=True to capture Metal traces."
                    ) from None

            else:
                logger.warning("Unknown activity: %s", a)

        if not activities:
            activities = [self._torch.profiler.ProfilerActivity.CPU]

        schedule = self._torch.profiler.schedule(
            wait=self._config.schedule_wait,
            warmup=self._config.schedule_warmup,
            active=self._config.schedule_active,
            repeat=self._config.schedule_repeat,
        )

        return self._torch.profiler.profile(
            activities=activities,
            schedule=schedule,
            on_trace_ready=self._trace_handler,
            record_shapes=self._config.record_shapes,
            profile_memory=self._config.profile_memory,
            with_stack=self._config.with_stack,
            with_flops=self._config.with_flops,
            acc_events=self._config.acc_events,
        )

    def _trace_handler(self, prof: Any) -> None:
        """Handle profiler trace output."""
        timestamp = prof.step_num

        # Print summary to console
        logger.info("Profiler trace ready for step %d", timestamp)

        # Export Chrome trace
        if self._config.export_chrome_trace:
            trace_path = self._config.output_dir / f"trace_{timestamp}.json"
            prof.export_chrome_trace(str(trace_path))
            logger.info("Chrome trace exported to: %s", trace_path)

        sort_key = "cuda_time_total" if "cuda" in self._config.activities else "cpu_time_total"

        # Export stacks
        if self._config.export_stacks:
            stacks_path = self._config.output_dir / f"stacks_{timestamp}.txt"
            prof.export_stacks(str(stacks_path))
            logger.info("Stacks exported to: %s", stacks_path)

        # Print summary
        print(prof.key_averages().table(sort_by=sort_key, row_limit=10))

    def _calculate_chunk_size(self, n_simulations: int) -> int:
        """Calculate optimal chunk size based on profiler schedule."""
        # Total steps in one schedule cycle
        total_steps = self._config.schedule_wait + self._config.schedule_warmup + self._config.schedule_active

        # We want at least total_steps chunks to capture the full schedule
        min_chunks = total_steps * self._config.schedule_repeat

        # But also don't make chunks too small
        chunk_size = max(n_simulations // min_chunks, 10_000)

        return chunk_size

    def run(
        self,
        sim: MonteCarloSimulation,
        n_simulations: int,
        seed_seq: np.random.SeedSequence | None,
        progress_callback: Callable[[int, int], None] | None = None,
        **simulation_kwargs: Any,
    ) -> np.ndarray:
        """
        Run simulations with profiling enabled.

        This method wraps the underlying backend's run() method with
        PyTorch profiler instrumentation. If chunking is enabled, it splits
        the simulations into multiple batches to generate multiple profiler
        traces based on the schedule configuration.

        Parameters match the underlying backend's run() method.
        """
        logger.info("Starting profiled run on %s backend", self._backend.device)

        use_mps_profiler = (
            str(self._backend.device) == "mps"
            and self._config.enable_mps_profiler
            and hasattr(self._torch, "mps")
            and hasattr(self._torch.mps, "profiler")
        )

        mps_ctx = self._torch.mps.profiler.profile() if use_mps_profiler else nullcontext()

        if str(self._backend.device) == "mps":
            if use_mps_profiler and os.getenv("MTL_CAPTURE_ENABLED") == "1":
                logger.info(
                    "MPS Metal tracing enabled (MTL_CAPTURE_ENABLED=1). "
                    "Use Xcode Instruments or gputrace to view GPU kernel timelines."
                )
            elif use_mps_profiler:
                logger.info(
                    "MPS Metal capture requested, but MTL_CAPTURE_ENABLED=0. "
                    "Set MTL_CAPTURE_ENABLED=1 to capture GPU kernel timelines."
                )

        if self._config.output_dir.exists():
            for f in self._config.output_dir.glob("trace_*.json"):
                with contextlib.suppress(OSError):
                    f.unlink()
            for f in self._config.output_dir.glob("trace_*.json.tmp"):
                with contextlib.suppress(OSError):
                    f.unlink()

        if not self._enable_chunking:
            # Simple mode: run once, step once
            with mps_ctx, self._create_profiler() as prof:
                results = self._backend.run(
                    sim, n_simulations, seed_seq, progress_callback, **simulation_kwargs
                )
                prof.step()

            logger.info("Profiling complete. Results saved to: %s", self._config.output_dir)
            return results

        # Chunked mode: split into multiple runs for better profiling
        chunk_size = self._chunk_size or self._calculate_chunk_size(n_simulations)
        n_chunks = (n_simulations + chunk_size - 1) // chunk_size

        logger.info(
            "Running %d simulations in %d chunks of ~%d for profiling", n_simulations, n_chunks, chunk_size
        )

        # Spawn seed sequences for each chunk
        if seed_seq is None:
            seed_seq = np.random.SeedSequence()
        chunk_seeds = seed_seq.spawn(n_chunks)

        results_list = []
        completed = 0
        with mps_ctx, self._create_profiler() as prof:
            for i, chunk_seed in enumerate(chunk_seeds):
                # Calculate chunk size (last chunk may be smaller)
                current_chunk_size = min(chunk_size, n_simulations - completed)

                # Run this chunk
                chunk_results = self._backend.run(
                    sim,
                    current_chunk_size,
                    chunk_seed,
                    None,  # Don't pass progress callback to chunks
                    **simulation_kwargs,
                )

                results_list.append(chunk_results)
                completed += current_chunk_size

                # Report progress
                if progress_callback:
                    progress_callback(completed, n_simulations)

                # Step the profiler after each chunk
                prof.step()

                logger.debug(
                    "Completed chunk %d/%d (%d/%d simulations)", i + 1, n_chunks, completed, n_simulations
                )

        # Combine all chunks
        results = np.concatenate(results_list)

        logger.info("Profiling complete. Results saved to: %s", self._config.output_dir)
        return results


@contextmanager
def profile_simulation(
    config: TorchProfilerConfig | None = None,
    name: str = "simulation",
):  # pragma: no cover
    """
    Context manager for profiling code blocks.

    Use this for more fine-grained profiling control.

    Parameters
    ----------
    config : TorchProfilerConfig, optional
        Profiler configuration.
    name : str, default "simulation"
        Name prefix for output files.

    Examples
    --------
    >>> from mcframework.profiling import profile_simulation, TorchProfilerConfig
    >>>
    >>> config = TorchProfilerConfig(output_dir="./my_profiles")
    >>> with profile_simulation(config, name="my_sim"):
    ...     # Your simulation code here
    ...     results = sim.run(10000)  # doctest: +SKIP
    """
    try:
        import torch  # pylint: disable=import-outside-toplevel
        import torch.profiler  # pylint: disable=import-outside-toplevel
    except ImportError:
        logger.warning("PyTorch not available, profiling disabled")
        yield
        return

    config = config or TorchProfilerConfig()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Create profiler with simpler schedule for context manager
    profiler = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=config.record_shapes,
        profile_memory=config.profile_memory,
        with_stack=config.with_stack,
        with_flops=config.with_flops,
        acc_events=config.acc_events,
    )

    with profiler:
        yield profiler

    # Export results
    if config.export_chrome_trace:
        trace_path = config.output_dir / f"{name}_trace.json"
        profiler.export_chrome_trace(str(trace_path))
        logger.info("Chrome trace exported to: %s", trace_path)

    # Determine sort key based on activities
    sort_key = "cuda_time_total" if "cuda" in config.activities else "cpu_time_total"

    # Print summary
    print(profiler.key_averages().table(sort_by=sort_key, row_limit=15))
