r"""
Base Torch utilities for GPU-accelerated Monte Carlo simulations.

This module provides shared utilities for all Torch-based backends:

Functions
    :func:`make_torch_generator` — Create explicit Torch RNG generators
    :func:`validate_torch_available` — Check if PyTorch is installed

Constants
    :data:`VALID_TORCH_DEVICES` — Supported Torch device types

Notes
-----
**RNG discipline.** All random sampling uses explicit ``torch.Generator``
objects seeded from :class:`numpy.random.SeedSequence`. Never uses global
Torch RNG (``torch.manual_seed``).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

__all__ = [
    "VALID_TORCH_DEVICES",
    "make_torch_generator",
    "make_curand_generator",
    "validate_torch_available",
    "import_torch",
]

# Valid Torch device types
VALID_TORCH_DEVICES = ("cpu", "mps", "cuda")


def import_torch():
    """
    Import and return the torch module.

    Returns
    -------
    module
        The torch module.

    Raises
    ------
    ImportError
        If PyTorch is not installed.
    """
    try:
        import torch as th  # pylint: disable=import-outside-toplevel
        return th
    except ImportError as e:
        raise ImportError(
            "Torch backend requires PyTorch. Install with: pip install mcframework[gpu]"
        ) from e


def validate_torch_available() -> None:
    """
    Check if PyTorch is installed and available.

    Raises
    ------
    ImportError
        If PyTorch is not installed.
    """
    import_torch()


def make_torch_generator(
    device: "torch.device",
    seed_seq: np.random.SeedSequence | None,
) -> "torch.Generator":
    r"""
    Create an explicit Torch generator seeded from a SeedSequence.

    This function spawns a child seed from the provided SeedSequence and
    uses it to initialize a Torch Generator. This preserves the hierarchical
    spawning model used by the NumPy backend.

    Parameters
    ----------
    device : torch.device
        Device for the generator (``"cpu"``, ``"mps"``, or ``"cuda"``).
    seed_seq : SeedSequence or None
        NumPy seed sequence to derive the Torch seed from.

    Returns
    -------
    torch.Generator
        Explicitly seeded generator for reproducible sampling.

    Notes
    -----
    **Why explicit generators?**

    - ``torch.manual_seed()`` is global state that breaks parallel composition
    - Explicit generators enable deterministic multi-stream MC
    - This mirrors NumPy's ``SeedSequence.spawn()`` semantics

    **Seed derivation:**

    .. code-block:: python

        child_seed = seed_seq.spawn(1)[0]
        seed_int = child_seed.generate_state(1, dtype="uint64")[0]
        generator.manual_seed(seed_int)

    This ensures each call with the same ``seed_seq`` produces identical results.

    Examples
    --------
    >>> import numpy as np
    >>> import torch
    >>> seed_seq = np.random.SeedSequence(42)
    >>> gen = make_torch_generator(torch.device("cpu"), seed_seq)  # doctest: +SKIP
    """
    th = import_torch()

    generator = th.Generator(device=device)

    if seed_seq is not None:
        # Spawn a child seed to preserve hierarchical RNG structure
        child_seed = seed_seq.spawn(1)[0]
        # Convert to 64-bit integer for Torch's Philox counter
        seed_int = int(child_seed.generate_state(1, dtype=np.uint64)[0])
        generator.manual_seed(seed_int)
    else:
        logger.warning(
            "No seed set for Torch backend; results will not be reproducible. "
            "Call set_seed() before run() for deterministic simulations."
        )

    return generator


def make_curand_generator(
    device_id: int,
    seed_seq: np.random.SeedSequence | None,
):
    r"""
    Create an explicit cuRAND generator seeded from a SeedSequence.

    This function spawns a child seed from the provided SeedSequence and
    uses it to initialize a CuPy RandomState. This preserves the hierarchical
    spawning model used by the NumPy backend.

    Parameters
    ----------
    device_id : int
        CUDA device index for the generator.
    seed_seq : SeedSequence or None
        NumPy seed sequence to derive the cuRAND seed from.

    Returns
    -------
    cupy.random.RandomState
        Explicitly seeded cuRAND generator for reproducible sampling.

    Raises
    ------
    ImportError
        If CuPy is not installed.

    Notes
    -----
    **Why explicit generators?**

    - ``cupy.random.seed()`` is global state that breaks parallel composition
    - Explicit generators enable deterministic multi-stream MC
    - This mirrors NumPy's ``SeedSequence.spawn()`` semantics

    **Seed derivation:**

    .. code-block:: python

        child_seed = seed_seq.spawn(1)[0]
        seed_int = child_seed.generate_state(1, dtype="uint64")[0]
        rng = cupy.random.RandomState(seed=seed_int)

    This ensures each call with the same ``seed_seq`` produces identical results.

    Examples
    --------
    >>> import numpy as np
    >>> seed_seq = np.random.SeedSequence(42)
    >>> # Requires CuPy installation
    >>> # gen = make_curand_generator(0, seed_seq)  # doctest: +SKIP
    """
    try:
        import cupy as cp  # pylint: disable=import-outside-toplevel
    except ImportError as e:
        raise ImportError(
            "cuRAND backend requires CuPy. Install with: pip install mcframework[cuda]"
        ) from e

    # Set device context
    cp.cuda.Device(device_id).use()

    if seed_seq is not None:
        # Spawn a child seed to preserve hierarchical RNG structure
        child_seed = seed_seq.spawn(1)[0]
        # Convert to 64-bit integer for cuRAND
        seed_int = int(child_seed.generate_state(1, dtype=np.uint64)[0])
        rng = cp.random.RandomState(seed=seed_int)
    else:
        logger.warning(
            "No seed set for cuRAND backend; results will not be reproducible. "
            "Call set_seed() before run() for deterministic simulations."
        )
        rng = cp.random.RandomState()

    return rng
