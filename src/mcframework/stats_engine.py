r"""
mcframework.stats_engine
========================
Statistical metrics and the engine used by the Monte Carlo framework.

This module defines:

- :class:`StatsContext`: a typed, explicit configuration object shared by all metrics.
- :class:`FnMetric`: a frozen adapter that names a metric function.
- :class:`StatsEngine`: an orchestrator that evaluates one or more metrics.

Common metrics include :func:`mean`, :func:`std`, :func:`percentiles`,
:func:`skew`, :func:`kurtosis`, and confidence intervals such as
:func:`ci_mean`, :func:`ci_mean_bootstrap`, and :func:`ci_mean_chebyshev`.

See Also
--------
mcframework.utils.autocrit
    Selects a z/t critical value for a target confidence level and effective sample size.
"""


# DEV NOTE:
# ===========================================================================
# The type checker throws a fit since x is ndarray and the checker can't verify
# that numpy/scipy functions accept that. So, we're suppressing the error
# with type: ignore[arg-type] where needed.
# ===========================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Callable, Generic, Iterable, Optional, Protocol, Sequence, TypeVar, Union

import numpy as np
from scipy.special import erfinv
from scipy.stats import kurtosis as sp_kurtosis
from scipy.stats import norm
from scipy.stats import skew as sp_skew

from .utils import autocrit

# Create local logger to avoid circular import with core
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


_PCTS = (5, 25, 50, 75, 95)  # default percentiles


class NanPolicy(str, Enum):
    r"""
    Strategies for handling the propagation of non-finite values.

    Attributes
    ----------
    propagate : str
        Propagate any NaNs or infinities encountered in the sample.
    omit : str
        Drop non-finite observations before computing a metric.
    """

    propagate = "propagate"
    omit = "omit"


class CIMethod(str, Enum):
    r"""
    Parametric strategies for selecting confidence-interval critical values.

    Attributes
    ----------
    auto : str
        Choose Student-t when :math:`n_\text{eff} < 30`, otherwise z.
    z : str
        Always use the normal :math:`z` critical value.
    t : str
        Always use the Student-:math:`t` critical value.
    bootstrap : str
        Defer to resampling-based bootstrap intervals.
    """

    auto = "auto"
    z = "z"
    t = "t"
    bootstrap = "bootstrap"


class BootstrapMethod(str, Enum):
    r"""
    Supported bootstrap confidence-interval flavors.

    Attributes
    ----------
    percentile : str
        Use the percentile method on the bootstrap distribution.
    bca : str
        Use the bias-corrected and accelerated (BCa) adjustment.
    """

    percentile = "percentile"
    bca = "bca"


@dataclass(slots=True)
class StatsContext:
    r"""
    Shared, explicit configuration for statistic and CI computations.

    Attributes
    ----------
    n : int
        Declared sample size (fallback when NaNs are not omitted).
    confidence : float, default 0.95
        Confidence level in :math:`(0, 1)`.
    ci_method : {"auto", "z", "t", "bootstrap"}, default "auto"
        Strategy for :func:`ci_mean`.
        If ``"auto"``, use Student-t when :math:`n_\text{eff} < 30` else normal z.
    percentiles : tuple of int, default ``(5, 25, 50, 75, 95)``
        Percentiles to compute in :func:`percentiles`.
    nan_policy : {"propagate", "omit"}, default "propagate"
        If ``"omit"``, drop non-finite values before all computations.
    target : float, optional
        Optional target value (e.g., true mean) for bias/MSE/Markov metrics.
    eps : float, optional
        Tolerance used by Chebyshev sizing and Markov bounds, when required.
    ddof : int, default 1
        Degrees of freedom for :func:`std` (1 => Bessel correction).
    ess : int, optional
        Effective sample size override (e.g., from MCMC diagnostics).
    rng : int or numpy.random.Generator, optional
        Seed or Generator used by bootstrap methods for reproducibility.
    n_bootstrap : int, default 10000
        Number of bootstrap resamples for :func:`ci_mean_bootstrap`.
    bootstrap : {"percentile", "bca"}, default "percentile"
        Bootstrap flavor for :func:`ci_mean_bootstrap`.
    block_size : int, optional
        Reserved for future block bootstrap support.

    Notes
    -----
    The context is immutable by convention at runtime; prefer :meth:`with_overrides`
    to construct a modified copy with a small set of changed fields.

    Examples
    --------
    >>> ctx = StatsContext(n=5000, confidence=0.95, ci_method=CIMethod.auto, nan_policy=NanPolicy.omit)
    >>> round(ctx.alpha, 2)
    0.05
    """

    n: int
    confidence: float = 0.95
    ci_method: CIMethod = "auto"
    percentiles: tuple[int, ...] = (5, 25, 50, 75, 95)
    nan_policy: NanPolicy = "propagate"
    target: Optional[float] = None
    eps: Optional[float] = None
    ddof: int = 1
    ess: Optional[int] = None
    rng: Optional[Union[int, np.random.Generator]] = None
    n_bootstrap: int = 10_000
    bootstrap: BootstrapMethod = "percentile"
    block_size: Optional[int] = None  # future: block bootstrap

    # ergonomics
    def with_overrides(self, **changes) -> "StatsContext":
        r"""
        Return a shallow copy with selected fields replaced.

        Parameters
        ----------
        **changes :
            Field overrides passed to :func:`dataclasses.replace`.

        Returns
        -------
        StatsContext
            Modified copy.

        Examples
        --------
        >>> ctx = StatsContext(n=1000)
        >>> ctx2 = ctx.with_overrides(confidence=0.9, n_bootstrap=2000)
        """
        return replace(self, **changes)

    @property
    def alpha(self) -> float:
        r"""
        One-sided tail probability :math:`\alpha = 1 - \text{confidence}`.

        Returns
        -------
        float
        """
        return 1.0 - self.confidence

    def q_bound(self) -> tuple[float, float]:
        r"""
        Percentile bounds corresponding to the current confidence.

        For :math:`\alpha = 1 - \text{confidence}`, returns
        :math:`(100\alpha/2,\; 100(1-\alpha/2))`.

        Returns
        -------
        tuple of float
            (lower_percentile, upper_percentile)
        """
        alpha = self.alpha
        return 100.0 * (alpha / 2), 100.0 * (1 - alpha / 2)

    def eff_n(self, observed_len: int, finite_count: Optional[int] = None) -> int:
        r"""
        Effective sample size :math:`n_\text{eff}` used by CI calculations.

        Priority is:
        1) explicit :attr:`ess`; 2) count of finite values if ``nan_policy="omit"``;
        3) declared :attr:`n` (fallback); else ``observed_len``.

        Parameters
        ----------
        observed_len : int
            Raw length of the input array.
        finite_count : int, optional
            Count of finite values (used when ``nan_policy="omit"``).

        Returns
        -------
        int
        """
        if self.ess is not None:
            return int(self.ess)
        if self.nan_policy == "omit" and finite_count is not None:
            return int(finite_count)
        return int(self.n or observed_len)

    def get_generators(self) -> np.random.Generator:
        r"""
        Return a NumPy :class:`~numpy.random.Generator` initialized from :attr:`rng`.

        Returns
        -------
        numpy.random.Generator
        """
        if isinstance(self.rng, np.random.Generator):
            return self.rng
        if isinstance(self.rng, (int, np.integer)):
            return np.random.default_rng(int(self.rng))
        return np.random.default_rng()

    def __post_init__(self) -> None:
        r"""
        Validate field ranges (confidence, percentiles, n_bootstrap, ddof).

        Raises
        ------
        ValueError
            If any field is outside its allowed range.
        """
        if not (0.0 < self.confidence < 1.0):
            raise ValueError("confidence must be in (0,1)")
        if any(p < 0 or p > 100 for p in self.percentiles):
            raise ValueError("percentiles must be in [0,100]")
        if self.n_bootstrap <= 0:
            raise ValueError("n_bootstrap must be > 0")
        if self.ddof < 0:
            raise ValueError("ddof must be >= 0")
        if self.eps is not None and self.eps <= 0:
            raise ValueError("eps must be positive")


class Metric(Protocol):
    r"""
    Protocol for metric callables used by :class:`StatsEngine`.

    A metric exposes a ``name`` attribute and is callable as:

    ``metric(x: numpy.ndarray, ctx: StatsContext) -> Any``

    Attributes
    ----------
    name : str
        Human-readable key under which the metric's value is returned.
    """

    name: str

    def __call__(self, x: np.ndarray, ctx: StatsContext, /) -> Any: ...


T = TypeVar("T")  # abc


@dataclass(frozen=True)
class FnMetric(Generic[T]):
    r"""
    Lightweight adapter that binds a human-readable ``name`` to a metric function.

    Parameters
    ----------
    name : str
        Key under which the metric result is stored in :meth:`StatsEngine.compute`.
    fn : callable
        Function with signature ``fn(x: ndarray, ctx: StatsContext) -> T``.
    doc : str, optional
        Short description displayed by UIs or docs.

    Examples
    --------
    >>> import numpy as np
    >>> m = FnMetric("mean", lambda a, ctx: float(np.mean(a)))
    >>> m(np.array([1, 2, 3]), StatsContext(n=3))
    2.0
    """

    name: str
    fn: Callable[[np.ndarray, StatsContext], T]
    doc: str = ""

    def __call__(self, x: np.ndarray, ctx: StatsContext) -> T:
        r"""
        Compute the metric.

        Parameters
        ----------
        x : ndarray
            Input sample.
        ctx : StatsContext
            Context parameters (see individual metric docs).

        Returns
        -------
        Any
            Metric value.

        Examples
        --------
        >>> m = FnMetric("mean", lambda a, _: float(np.mean(a)))
        >>> m(np.array([1, 2, 3]), {})
        2.0
        """
        return self.fn(x, ctx)


def _validate_ctx(ctx: dict[str, Any], required: set[str], optional: set[str]):
    missing = required - ctx.keys()
    if missing:
        raise ValueError(
            f"Missing required context keys: {missing} \n "
            f"Optional keys are: {optional} \n "
            f"Provided keys are: {set(ctx.keys())}"
        )


class StatsEngine:
    r"""
    Orchestrator that evaluates a set of metrics over an input array.

    Parameters
    ----------
    metrics : iterable of Metric
        Callables with a ``name`` and signature ``metric(x, ctx)``.

    Notes
    -----
    All metrics receive the *same* :class:`StatsContext`. Prefer field names that
    read well across multiple metrics and avoid collisions.

    Examples
    --------
    >>> eng = StatsEngine([FnMetric("mean", mean), FnMetric("std", std)])
    >>> x = np.array([1., 2., 3.])
    >>> eng.compute(x, StatsContext(n=len(x)))
    {'mean': 2.0, 'std': 1.0}
    """

    def __init__(self, metrics: Iterable[Metric]):
        self._metrics = list(metrics)

    def available(self) -> tuple[str, ...]:
        return tuple(m.name for m in self._metrics)

    def compute(
        self,
        x: np.ndarray,
        ctx: Optional[StatsContext] = None,
        select: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        r"""
        Evaluate all registered metrics on ``x``.

        Parameters
        ----------
        x : ndarray
            Sample values.
        ctx : StatsContext, optional
            Context parameters. If None, one is built from **kwargs.
        select : sequence of str, optional
            If given, compute only the metrics with these names.
        **kwargs :
            Used to build a StatsContext if ctx is None.
            Required: 'n' (int)
            Optional: 'confidence', 'ci_method', 'percentiles', etc.

        Returns
        -------
        dict
            Mapping from metric name to computed value.
        """
        if ctx is not None:
            ctx = _ensure_ctx(ctx, x)

        # build context from kwargs
        else:
            base = dict(kwargs)
            base.setdefault("n", int(np.asarray(x).size))
            ctx = StatsContext(**base)

        metrics_to_compute = (
            self._metrics if select is None else [m for m in self._metrics if m.name in set(select)]
        )

        out: dict[str, Any] = {}
        for m in metrics_to_compute:
            try:
                result = m(x, ctx)

                # Filter out empty dicts (metrics that can't compute)
                if isinstance(result, dict) and len(result) == 0:
                    logger.debug(f"Metric '{m.name}' returned empty dict, skipping")
                    continue

                out[m.name] = result

            except ValueError as e:
                msg = str(e)
                # Check for eps requirement
                if (
                    "requires ctx.target" in msg
                    or "requires ctx.eps" in msg  # ← Add this
                    or "Missing required context keys" in msg
                ):
                    logger.debug(f"Skipping metric {m.name}: {msg}")
                    continue
                raise
            except Exception:
                logger.exception(f"Error computing metric {m.name}")
                continue

        return out


def _ensure_ctx(ctx: Any, x: np.ndarray) -> StatsContext:
    r"""
    Normalize arbitrary context inputs into a :class:`StatsContext`.

    Parameters
    ----------
    ctx : Any
        A :class:`StatsContext`, mapping, object with attributes, or ``None``.
    x : ndarray
        Sample used to infer the fallback ``n`` when missing.

    Returns
    -------
    StatsContext
        Context instance with all required fields populated.

    Raises
    ------
    TypeError
        If ``ctx`` cannot be interpreted as configuration data.
    """
    # ctx is a StatsContext
    if isinstance(ctx, StatsContext):
        return ctx

    arr_len = int(np.asarray(x).size)

    # ctx is None
    if ctx is None:
        return StatsContext(n=arr_len)

    # ctx is a dict
    if isinstance(ctx, dict):
        data = dict(ctx)
        data.setdefault("n", arr_len)
        return StatsContext(**data)

    # Fallback: try to read attributes
    try:
        data = dict(vars(ctx))
    except TypeError:
        raise TypeError("ctx must be a StatsContext, dict, None, or an object with attributes")
    data.setdefault("n", arr_len)
    return StatsContext(**data)


def _clean(x: np.ndarray, ctx: Any) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Sanitize the input sample and return the finite-value mask.

    Parameters
    ----------
    x : ndarray
        Raw input sample.
    ctx : StatsContext
        Context whose :attr:`~StatsContext.nan_policy` guides filtering.

    Returns
    -------
    tuple of ndarray
        ``(arr, finite_mask)`` where ``arr`` is the possibly filtered sample and
        ``finite_mask`` marks finite elements in the original sample.

    Raises
    ------
    ValueError
        If an unknown :attr:`~StatsContext.nan_policy` is supplied.
    """
    arr = np.asarray(x, dtype=float)
    finite = np.isfinite(arr)

    if ctx.nan_policy == "omit":
        arr = arr[finite]
    elif ctx.nan_policy not in ("omit", "propagate", "raise"):
        raise ValueError(f"Unknown nan_policy: {ctx.nan_policy}")

    return arr, finite


def _effective_sample_size(x: np.ndarray, ctx: StatsContext) -> int:
    r"""
    Compute :math:`n_\text{eff}` used by confidence-interval calculations.

    Parameters
    ----------
    x : ndarray
        Input sample.
    ctx : StatsContext
        Context governing NaN handling and declared sample size.

    Returns
    -------
    int
        Effective sample size according to :meth:`StatsContext.eff_n`.
    """
    arr, finite = _clean(x, ctx)
    return ctx.eff_n(observed_len=arr.size, finite_count=finite)


def mean(x: np.ndarray, ctx: StatsContext):
    r"""
    Sample mean.

    Parameters
    ----------
    x : ndarray
        Input sample.
    ctx : StatsContext
        If ``nan_policy="omit"``, non-finite values are excluded.

    Returns
    -------
    float
        :math:`\bar X = \frac{1}{n}\sum_i x_i`.

    Examples
    --------
    >>> mean(np.array([1, 2, 3]))
    2.0
    """
    ctx = _ensure_ctx(ctx, x)
    arr, _ = _clean(x, ctx)
    return float(np.mean(arr)) if arr.size else float("nan")


def std(x: np.ndarray, ctx: StatsContext):
    r"""
    Sample standard deviation with Bessel correction.

    Parameters
    ----------
    x : ndarray
        Input sample.
    ctx : StatsContext
        Uses :attr:`StatsContext.ddof` (default 1).
        If ``nan_policy="omit"``, non-finite values are excluded.
    Returns
    -------
    float
        :math:`s = \sqrt{\frac{1}{n-1}\sum_i (x_i-\bar X)^2}` (returns ``0.0`` if :math:`n_\text{eff} \le 1`).

    Examples
    --------
    >>> std(np.array([1, 2, 3]), {})
    1.0
    """
    ctx = _ensure_ctx(ctx, x)
    arr, finite = _clean(x, ctx)
    n_eff = ctx.eff_n(observed_len=arr.size, finite_count=finite)
    if n_eff <= 1:
        return 0.0
    return float(np.std(arr, ddof=ctx.ddof))


def percentiles(x: np.ndarray, ctx: StatsContext) -> dict[int, float]:
    r"""
    Empirical percentiles evaluated on the cleaned sample.

    Parameters
    ----------
    x : ndarray
        Input sample.
    ctx : StatsContext
        Uses :attr:`StatsContext.percentiles` and :attr:`StatsContext.nan_policy`.

    Returns
    -------
    dict[int, float]
        Mapping :math:`p \mapsto Q_p(x)` where :math:`Q_p` is the
        :math:`p`-th percentile.

    Examples
    --------
    >>> percentiles(np.array([0., 1., 2., 3.]), {"percentiles": (50, 75)})
    {50: 1.5, 75: 2.25}
    """
    ctx = _ensure_ctx(ctx, x)
    arr, _ = _clean(x, ctx)
    if arr.size == 0:
        return {p: float("nan") for p in ctx.percentiles}
    pct_values = np.percentile(arr, ctx.percentiles)
    return dict(zip(ctx.percentiles, map(float, pct_values)))


def skew(x: np.ndarray, ctx: StatsContext) -> float:
    r"""
    Unbiased sample skewness (Fisher–Pearson standardized third central moment).

    Parameters
    ----------
    x : ndarray
        Input sample.
    ctx : StatsContext
        Uses :attr:`StatsContext.nan_policy`.

    Returns
    -------
    float
        Fisher–Pearson standardized third central moment, returning ``0.0`` if
        :math:`n_\text{eff} \le 2`.

    Notes
    -----
    Uses :func:`scipy.stats.skew` with ``bias=False``.

    Examples
    --------
    >>> round(skew(np.array([1, 2, 3, 10.0]), {}), 3) > 0
    True
    """
    ctx = _ensure_ctx(ctx, x)
    arr, _ = _clean(x, ctx)
    return float(sp_skew(arr, bias=False)) if arr.size > 2 else 0.0  # type: ignore[arg-type]


def kurtosis(x: np.ndarray, ctx: StatsContext) -> float:
    r"""
    Unbiased sample **excess** kurtosis (Fisher definition).

    Parameters
    ----------
    x : ndarray
        Input sample.
    ctx : StatsContext
        Uses :attr:`StatsContext.nan_policy`.

    Returns
    -------
    float
        Excess kurtosis (``0.0`` if :math:`n_\text{eff} \le 3`).

    Notes
    -----
    Uses :func:`scipy.stats.kurtosis` with ``fisher=True, bias=False``.

    Examples
    --------
    >>> round(kurtosis(np.array([1, 2, 3, 4.0]), {}), 6)
    -1.200000
    """
    ctx = _ensure_ctx(ctx, x)
    arr, _ = _clean(x, ctx)
    return float(sp_kurtosis(arr, fisher=True, bias=False)) if arr.size > 3 else 0.0  # type: ignore[arg-type]


def ci_mean(x: np.ndarray, ctx) -> dict[str, float | str]:
    r"""
    Parametric CI for :math:`\mathbb{E}[X]` using z/t critical values.

    Let :math:`\bar X` be the sample mean and :math:`SE = s/\sqrt{n_\text{eff}}`.
    The interval is

    .. math::
       \bar X \pm c \cdot SE,

    where :math:`c` is selected by :func:`mcframework.utils.autocrit` according
    to :attr:`StatsContext.ci_method` and :math:`n_\text{eff}`.

    Parameters
    ----------
    x : ndarray
        Input sample.
    ctx : StatsContext or Mapping
        Configuration supplying at least :attr:`StatsContext.n`. Additional
        fields such as :attr:`StatsContext.confidence`, :attr:`StatsContext.ddof`,
        and :attr:`StatsContext.ci_method` refine the interval.

    Returns
    -------
    dict[str, float | str]
        Mapping with keys

        ``confidence``
            Requested confidence level.
        ``method``
            Resolved CI method (``"z"``, ``"t"``, or ``"auto"``).
        ``low`` / ``high``
            Lower and upper endpoints.
        ``se`` / ``crit``
            Standard error and critical value when :math:`n_\text{eff} \ge 2`.
    """
    ctx = _ensure_ctx(ctx, x)
    arr, _ = _clean(x, ctx)

    # if everything got dropped or x was empty
    if arr.size == 0:
        return {
            "confidence": ctx.confidence,
            "method": ctx.ci_method,
            "low": float("nan"),
            "high": float("nan"),
        }

    n_eff = _effective_sample_size(arr, ctx)  # counts after cleaning if 'omit'
    if n_eff < 2:
        return {
            "confidence": ctx.confidence,
            "method": ctx.ci_method,
            "low": float("nan"),
            "high": float("nan"),
            "se": float("nan"),
            "crit": float("nan"),
        }

    mu = float(np.mean(arr))

    s = float(np.std(arr, ddof=getattr(ctx, "ddof", 1)))
    if s == 0.0:
        # degenerate data -> zero SE -> CI collapses to point
        se = 0.0
    else:
        se = s / np.sqrt(n_eff)

    crit, method = autocrit(ctx.confidence, n_eff, ctx.ci_method)

    return {
        "confidence": ctx.confidence,
        "method": method,
        "se": float(se),
        "crit": float(crit),
        "low": float(mu - crit * se),
        "high": float(mu + crit * se),
    }


def _bootstrap_means(arr: np.ndarray, B: int, rng: np.random.Generator) -> np.ndarray:
    r"""
    Generate bootstrap replicates of the sample mean.

    Parameters
    ----------
    arr : ndarray
        Cleaned sample values.
    B : int
        Number of bootstrap resamples.
    rng : numpy.random.Generator
        Random generator used to draw resamples.

    Returns
    -------
    ndarray
        Array of length ``B`` containing :math:`\{\bar X_b^*\}`.
    """
    n = arr.size
    idx = rng.integers(0, n, size=(B, n), endpoint=False)
    return arr[idx].mean(axis=1)


def ci_mean_bootstrap(x: np.ndarray, ctx: StatsContext) -> dict[str, float | str]:
    r"""
    Bootstrap confidence interval for :math:`\mathbb{E}[X]` via resampling.

    Generates ``n_bootstrap`` bootstrap samples by drawing with replacement
    from the input data, computes the mean of each sample, and returns
    the percentile-based confidence interval from the resulting bootstrap
    distribution.

    The CI is constructed as

    .. math::
       \left[\,Q_{\alpha/2}(\bar X^*),\; Q_{1-\alpha/2}(\bar X^*)\,\right],

    where :math:`\bar X^*` denotes the bootstrap means and
    :math:`\alpha = 1 - \text{confidence}`.

    Parameters
    ----------
    x : ndarray
        Input sample.
    ctx : StatsContext or Mapping
        Configuration that may specify:

        - :attr:`StatsContext.confidence`
            Confidence level :math:`\in (0, 1)`.
        - :attr:`StatsContext.n_bootstrap`
            Number of bootstrap resamples (default ``10_000``).
        - :attr:`StatsContext.nan_policy`
            Whether to omit non-finite values before bootstrapping.
        - :attr:`StatsContext.bootstrap`
            Flavor (``"percentile"`` or ``"bca"``).
        - :attr:`StatsContext.rng`
            Seed or generator for reproducibility.

    Returns
    -------
    dict
        Mapping with keys:

        - ``confidence`` : float
          The requested confidence level.
        - ``method`` : str
          ``"bootstrap-percentile"`` or ``"bootstrap-bca"``.
        - ``low`` : float
          Lower bound of the CI (NaN if sample is empty).
        - ``high`` : float
          Upper bound of the CI (NaN if sample is empty).

    See Also
    --------
    ci_mean : Parametric CI using z/t critical values.
    ci_mean_chebyshev : Distribution-free CI via Chebyshev's inequality.

    Notes
    -----
    The bootstrap percentile method is distribution-free and asymptotically
    valid under mild regularity conditions. For small samples, it may have
    lower coverage than the nominal confidence level.

    Examples
    --------
    >>> x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> result = ci_mean_bootstrap(x, {"confidence": 0.9, "n_bootstrap": 5000, "rng": 42})
    >>> result["method"]
    'bootstrap'
    >>> 1.5 < result["low"] < result["high"] < 4.5
    True
    """
    ctx = _ensure_ctx(ctx, x)
    arr, _ = _clean(x, ctx)
    if arr.size == 0:
        return {}
    B = int(ctx.n_bootstrap)
    g = ctx.get_generators()
    means = _bootstrap_means(arr, B, g)
    loq, hiq = ctx.q_bound()
    method = ctx.bootstrap
    if method == "percentile" or arr.size < 3:
        low, high = np.percentile(means, [loq, hiq])
        return {
            "confidence": ctx.confidence,
            "method": "bootstrap-percentile",
            "low": float(low),
            "high": float(high),
        }

    # BCa
    m_hat = float(np.mean(arr))
    prop = float(np.sum(means < m_hat)) / B
    prop = np.clip(prop, 1e-12, 1 - 1e-12)
    z0 = float(np.sqrt(2) * erfinv(2 * prop - 1))

    s = np.sum(arr, dtype=float)
    jack = (s - arr) / (arr.size - 1)
    d = jack - float(np.mean(jack))
    a = float(np.sum(d**3)) / (6.0 * (np.sum(d**2) ** 1.5) + 1e-30)

    zlo = float(norm.ppf((1 - ctx.confidence) / 2))
    zhi = float(norm.ppf(1 - (1 - ctx.confidence) / 2))

    def _adj(z: float) -> float:
        num = z0 + z
        den = 1.0 - a * num
        return float(norm.cdf(z0 + num / den)) * 100.0

    p_lo, p_hi = np.clip(_adj(zlo), 0, 100), np.clip(_adj(zhi), 0, 100)
    low, high = np.percentile(means, [p_lo, p_hi])
    return {
        "confidence": ctx.confidence,
        "method": "bootstrap-bca",
        "low": float(low),
        "high": float(high),
    }


def ci_mean_chebyshev(x: np.ndarray, ctx: StatsContext) -> dict[str, float | str]:
    r"""
    Distribution-free CI for :math:`\mathbb{E}[X]` via Chebyshev's inequality.

    For :math:`\delta = 1 - \text{confidence}`, choose :math:`z=1/\sqrt{\delta}`
    so that

    .. math::
       \Pr\!\left(\,|\bar X - \mu| \ge z\,SE\,\right) \le \delta,
       \qquad SE = \frac{s}{\sqrt{n_\text{eff}}}.

    Parameters
    ----------
    x : ndarray
        Input sample.
    ctx : StatsContext or Mapping
        Requires :attr:`StatsContext.n` and may specify
        :attr:`StatsContext.confidence`.

    Returns
    -------
    dict[str, float | str]
        Mapping with keys ``confidence``, ``method``, ``low``, and ``high``.
    """
    ctx = _ensure_ctx(ctx, x)
    n_eff = _effective_sample_size(x, ctx)
    if n_eff < 2:
        return {}
    mu = mean(x, ctx)
    s = std(x, ctx)
    k = 1.0 / np.sqrt(max(1e-30, 1.0 - ctx.confidence))  # 1/sqrt(alpha)
    half = k * s / np.sqrt(n_eff)
    return {
        "confidence": ctx.confidence,
        "method": "chebyshev",
        "low": float(mu - half),
        "high": float(mu + half),
    }


def chebyshev_required_n(x: np.ndarray, ctx: StatsContext) -> int:
    r"""
    Required :math:`n` to achieve Chebyshev CI half-width :math:`\le \varepsilon`.

    With :math:`\delta = 1 - \text{confidence}`, the half-width is
    :math:`z\,SE = \dfrac{s}{\sqrt{n_\text{eff}\,\delta}}` where :math:`z=1/\sqrt{\delta}`.
    Solve :math:`n_\text{eff} \ge \dfrac{s^2}{\varepsilon^2\,\delta}`.

    Parameters
    ----------
    x : ndarray
        Input sample.
    ctx : StatsContext or Mapping
        Must supply :attr:`StatsContext.eps` (target half-width) and may
        override :attr:`StatsContext.confidence`.

    Returns
    -------
    int
        Minimum integer :math:`n_\text{eff}`.

    Examples
    --------
    >>> chebyshev_required_n(np.array([1., 2., 3.]), {"eps": 0.5, "confidence": 0.9})
    8
    """
    ctx = _ensure_ctx(ctx, x)
    arr, _ = _clean(x, ctx)
    if ctx.eps is None:
        raise ValueError("chebyshev_required_n requires ctx.eps")
    if ctx.eps <= 0:
        raise ValueError("ctx.eps must be positive")
    s = std(x, ctx)
    k = 1.0 / np.sqrt(max(1e-30, 1.0 - ctx.confidence))
    return int(np.ceil(((k * s) / float(ctx.eps)) ** 2))


def markov_error_prob(x: np.ndarray, ctx: StatsContext) -> float:
    r"""
    Markov bound on error probability for target :math:`\theta`.

    Using the squared error of the sample mean,
    :math:`\mathrm{MSE}(\bar X) \approx \frac{s^2}{n} + \text{Bias}^2`, Markov gives

    .. math::
       \Pr\!\left(\,|\bar X - \theta| \ge \varepsilon\,\right)
       \;\le\; \frac{\mathrm{MSE}}{\varepsilon^2}.

    Parameters
    ----------
    x : ndarray
        Input sample.
    ctx : StatsContext or Mapping
        Requires :attr:`StatsContext.target` and :attr:`StatsContext.eps`. The
        declared :attr:`StatsContext.n` influences the context but does not enter
        directly into the bound.

    Returns
    -------
    float
        Upper bound in :math:`[0, 1]`.
    """
    ctx = _ensure_ctx(ctx, x)
    if ctx.target is None:
        raise ValueError("markov_error_prob requires ctx.target")
    if ctx.eps is None:
        raise ValueError("markov_error_prob requires ctx.eps")
    if ctx.eps <= 0:
        raise ValueError("ctx.eps must be positive")
    arr, _ = _clean(x, ctx)
    mse = float(np.mean((arr - ctx.target) ** 2))
    return mse / (ctx.eps**2)


def bias_to_target(x: np.ndarray, ctx: StatsContext) -> float:
    r"""
    Bias of the sample mean relative to a target :math:`\theta`.

    Parameters
    ----------
    x : ndarray
        Input sample.
    ctx : StatsContext or Mapping
        Requires :attr:`StatsContext.target`.

    Returns
    -------
    float
        Estimated bias :math:`\bar X - \theta`.
    """
    ctx = _ensure_ctx(ctx, x)
    if ctx.target is None:
        raise ValueError("bias_to_target requires ctx.target")
    return float(mean(x, ctx) - ctx.target)


def mse_to_target(x: np.ndarray, ctx: StatsContext) -> float:
    r"""
    Mean squared error of :math:`\bar X` relative to a target :math:`\theta`.

    Approximated by

    .. math::
       \mathrm{MSE}(\bar X) \approx \frac{s^2}{n} + (\bar X - \theta)^2.

    Parameters
    ----------
    x : ndarray
        Input sample.
    ctx : StatsContext or Mapping
        Requires :attr:`StatsContext.target`.

    Returns
    -------
    float
        Estimated mean squared error relative to ``target``.
    """
    ctx = _ensure_ctx(ctx, x)
    if ctx.target is None:
        raise ValueError("mse_to_target requires ctx.target")
    arr, _ = _clean(x, ctx)
    return float(np.mean((arr - ctx.target) ** 2))


def build_default_engine(
    include_dist_free: bool = True,
    include_target_bounds: bool = True,
) -> StatsEngine:
    r"""
    Construct a :class:`StatsEngine` with a practical set of metrics.

    Parameters
    ----------
    include_dist_free : bool, default True
        Include Chebyshev-based CI and sizing.
    include_target_bounds : bool, default True
        Include :func:`bias_to_target`, :func:`mse_to_target`, :func:`markov_error_prob`.

    Returns
    -------
    StatsEngine
    """
    metrics: list[Metric] = [
        FnMetric[float]("mean", mean, "Sample mean"),
        FnMetric[float]("std", std, "Sample standard deviation"),
        FnMetric[dict[int, float]]("percentiles", percentiles, "Percentiles over the sample"),
        FnMetric[float]("skew", skew, "Fisher skewness (unbiased)"),
        FnMetric[float]("kurtosis", kurtosis, "Excess kurtosis (unbiased)"),
        FnMetric[dict[str, float | str]]("ci_mean", ci_mean, "z/t CI for the mean"),
        FnMetric[dict[str, float | str]]("ci_mean_bootstrap", ci_mean_bootstrap, "Bootstrap CI for the mean"),
    ]
    if include_dist_free:
        metrics.extend(
            [
                FnMetric[dict[str, float | str]](
                    "ci_mean_chebyshev", ci_mean_chebyshev, "Chebyshev bound CI for the mean"
                ),
                FnMetric[int](
                    "chebyshev_required_n", chebyshev_required_n, "Required n under Chebyshev to reach eps"
                ),
            ]
        )
    if include_target_bounds:
        metrics.extend(
            [
                FnMetric[float]("markov_error_prob", markov_error_prob, "Markov bound P(|X-target|>=eps)"),
                FnMetric[float]("bias_to_target", bias_to_target, "Bias relative to target"),
                FnMetric[float]("mse_to_target", mse_to_target, "Mean squared error to target"),
            ]
        )
    return StatsEngine(metrics)


# Build a default engine at import time
DEFAULT_ENGINE = build_default_engine(
    include_dist_free=True,
    include_target_bounds=True,
)

__all__ = [
    "StatsContext",
    "Metric",
    "FnMetric",
    "StatsEngine",
    "mean",
    "std",
    "percentiles",
    "skew",
    "kurtosis",
    "ci_mean",
    "ci_mean_bootstrap",
    "ci_mean_chebyshev",
    "chebyshev_required_n",
    "markov_error_prob",
    "bias_to_target",
    "mse_to_target",
    "build_default_engine",
    "DEFAULT_ENGINE",
]
