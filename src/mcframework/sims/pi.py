"""Pi estimation simulation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy.random import Generator

from ..core import MonteCarloSimulation

if TYPE_CHECKING:
    import torch

__all__ = ["PiEstimationSimulation"]


class PiEstimationSimulation(MonteCarloSimulation):
    r"""
    Estimate :math:`\pi` by geometric probability on the unit disk.

    The simulation throws :math:`n` i.i.d. points :math:`(X_i, Y_i)` uniformly on
    :math:`[-1, 1]^2` and uses the identity

    .. math::
       \pi = 4 \,\Pr\!\left(X^2 + Y^2 \le 1\right),

    to form the Monte Carlo estimator

    .. math::
       \widehat{\pi}_n = \frac{4}{n} \sum_{i=1}^n \mathbf{1}\{X_i^2 + Y_i^2 \le 1\}.

    Attributes
    ----------
    name : str
        Human-readable label registered with :class:`~mcframework.core.MonteCarloFramework`.
    supports_batch : bool
        Whether this simulation supports Torch batch execution (``True``).

    Notes
    -----
    This simulation supports both scalar (NumPy) and vectorized (Torch) execution:

    - **NumPy path**: Uses :meth:`single_simulation` with optional antithetic sampling.
    - **Torch path**: Uses :meth:`torch_batch` for GPU-accelerated batch execution.

    Example
    -------
    >>> sim = PiEstimationSimulation()
    >>> sim.set_seed(42)
    >>> result = sim.run(100_000, backend="torch")  # GPU-ready  # doctest: +SKIP
    """

    supports_batch: bool = True

    def __init__(self):
        super().__init__("Pi Estimation")

    def single_simulation(  # pylint: disable=arguments-differ
        self,
        n_points: int = 10_000,
        antithetic: bool = False,
        _rng: Optional[Generator] = None,
        **kwargs,
    ) -> float:
        r"""
        Throw :math:`n_{\text{points}}` darts at :math:`[-1, 1]^2` and return the
        single-run estimator :math:`\widehat{\pi}`.

        Parameters
        ----------
        n_points : int, default ``10_000``
            Number of uniformly distributed points to simulate. The Monte Carlo
            variance decays as :math:`\mathcal{O}(n_{\text{points}}^{-1})`.
        antithetic : bool, default ``False``
            Whether to pair each point :math:`(x, y)` with its reflection
            :math:`(-x, -y)` to achieve first-order variance cancellation.
        **kwargs : Any
            Ignored. Reserved for framework compatibility.

        Returns
        -------
        float
            Estimate of :math:`\pi` computed via
            :math:`\widehat{\pi} = 4 \,\widehat{p}`, where
            :math:`\widehat{p}` is the observed fraction of darts that land inside
            the unit disk.
        """
        rng = self._rng(_rng, self.rng)
        if not antithetic:  # pragma: no cover
            pts = rng.uniform(-1.0, 1.0, (n_points, 2))
            inside = np.sum(np.sum(pts * pts, axis=1) <= 1.0)
            return float(4.0 * inside / n_points)
        # Antithetic sampling mirrors each draw (x, y) with (-x, -y)
        m = n_points // 2
        u = rng.uniform(-1.0, 1.0, (m, 2))
        ua = -u
        pts = np.vstack([u, ua])
        if pts.shape[0] < n_points:
            pts = np.vstack([pts, rng.uniform(-1.0, 1.0, (1, 2))])
        inside = np.sum(np.sum(pts * pts, axis=1) <= 1.0)
        return float(4.0 * inside / n_points)

    def torch_batch(
        self,
        n: int,
        *,
        device: "torch.device",
        generator: "torch.Generator",
    ) -> "torch.Tensor":
        r"""
        Vectorized Torch implementation for GPU-accelerated Pi estimation.

        Each element of the returned tensor is an independent estimate of :math:`\pi`
        using the standard Monte Carlo disk-in-square method. This is equivalent to
        calling :meth:`single_simulation` with ``n_points=1`` for each draw.

        Parameters
        ----------
        n : int
            Number of :math:`\pi` estimates to generate.
        device : torch.device
            Device for computation (``"cpu"``, ``"mps"``, or ``"cuda"``).
        generator : torch.Generator
            Explicit Torch generator for reproducible random sampling. All random
            operations must use this generator—never rely on global Torch RNG.

        Returns
        -------
        torch.Tensor
            A 1D tensor of length ``n`` where each element is ``4.0`` (inside disk)
            or ``0.0`` (outside disk). Returns float32 for MPS compatibility;
            the framework promotes to float64 after moving to CPU.

        Notes
        -----
        Unlike :meth:`single_simulation`, this method does not support the
        ``n_points`` parameter—each simulation is a single point evaluation.
        For high-precision estimates, use many simulations and let the framework
        compute the mean.

        The expected value of each element is :math:`\pi`, so the sample mean
        converges to :math:`\pi` as ``n → ∞``.

        **MPS compatibility.** This method returns float32 tensors to support
        Apple MPS backend (which doesn't support float64). The framework handles
        promotion to float64 after moving results to CPU.
        """
        import torch as th  # pylint: disable=import-outside-toplevel

        # Sample points uniformly in [-1, 1]^2 using explicit generator
        x = th.rand(n, device=device, generator=generator) * 2.0 - 1.0
        y = th.rand(n, device=device, generator=generator) * 2.0 - 1.0

        # Check if inside unit disk
        inside = (x * x + y * y) <= 1.0

        # Return 4.0 * indicator (expected value = pi)
        # Note: Use float32 on device (MPS doesn't support float64)
        # The framework promotes to float64 after moving to CPU
        return 4.0 * inside.float()
