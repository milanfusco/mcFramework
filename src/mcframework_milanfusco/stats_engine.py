r"""
Statistical metrics and orchestration engine for Monte Carlo simulations.

This module provides:

Configuration
    :class:`StatsContext` — Shared configuration for all metric computations
    (confidence level, NaN policy, bootstrap settings, etc.)

Descriptive Metrics
    :func:`mean`, :func:`std`, :func:`percentiles`, :func:`skew`, :func:`kurtosis`

Confidence Intervals
    - :func:`ci_mean` — Parametric CI using z/t critical values
    - :func:`ci_mean_bootstrap` — Bootstrap CI (percentile or BCa)
    - :func:`ci_mean_chebyshev` — Distribution-free Chebyshev bound

Target-Based Metrics
    :func:`bias_to_target`, :func:`mse_to_target`, :func:`markov_error_prob`,
    :func:`chebyshev_required_n`

Engine
    :class:`StatsEngine` — Orchestrator that evaluates multiple metrics
    :class:`FnMetric` — Adapter to bind a name to a metric function
    :obj:`DEFAULT_ENGINE` — Pre-configured engine with all standard metrics

Example
-------
>>> from mcframework.stats_engine import DEFAULT_ENGINE, StatsContext
>>> import numpy as np
>>> data = np.random.normal(0, 1, 10000)
>>> ctx = StatsContext(n=len(data), confidence=0.95)
>>> result = DEFAULT_ENGINE.compute(data, ctx)
>>> result.metrics['mean']  # doctest: +SKIP
0.0012

See Also
--------
mcframework.utils.autocrit
    Selects z or t critical value based on sample size.
mcframework.core.MonteCarloSimulation
    Simulation base class that uses this engine.
"""


# DEV NOTE:
# ===========================================================================
# The type checker throws a fit since x is a numpy ndarray and the checker can't verify
# that numpy/scipy functions accept that. So, we're suppressing the error
# with type: ignore[arg-type] where needed.
# ===========================================================================

# pylint: disable=invalid-name
# Enum values (propagate, omit, auto, z, t, bootstrap, percentile, bca) are lowercase by design

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Mapping,
    Protocol,
    Sequence,
    SupportsFloat,
    TypeAlias,
    TypeVar,
)

import numpy as np
from scipy.special import erfinv  # type: ignore[import-untyped]
from scipy.stats import kurtosis as sp_kurtosis  # type: ignore[import-untyped]
from scipy.stats import norm  # type: ignore[import-untyped]
from scipy.stats import skew as sp_skew  # type: ignore[import-untyped]

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


# TODO: make these configurable
# Numerical stability constants
_SMALL_PROB_BOUND = 1e-12  # Minimum probability for BCa bootstrap to avoid log(0)
_SMALL_VARIANCE_BOUND = 1e-30  # Minimum variance denominator to prevent division by zero
_BCA_JACKKNIFE_DENOMINATOR = 6.0  # Constant in BCa acceleration calculation


# Custom exceptions for better error handling
class MissingContextError(ValueError):
    """Raised when a required context field is missing."""


class InsufficientDataError(ValueError):
    """Raised when insufficient data is available for computation."""


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

    The context keeps track of three recurring quantities:

    * Confidence level :math:`\gamma = \texttt{confidence}` with tail mass
      :math:`\alpha = 1 - \gamma`.
    * Effective sample size :math:`n_\text{eff}`, obtained from
      :meth:`eff_n` and used by every finite-sample adjustment.
    * Requested quantiles :math:`\mathcal{P} = \{p_i\}` that drive percentile
      metrics.

    Throughout the module we repeatedly use the identities

    .. math::

       \alpha = 1 - \gamma, \qquad
       q_\text{low} = 100 \frac{\alpha}{2}, \qquad
       q_\text{high} = 100 \left(1 - \frac{\alpha}{2}\right),

    which are provided via the :meth:`alpha` and :meth:`q_bound` helpers.

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
        Degrees of freedom for :func:`numpy.std` (1 => Bessel correction).
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
    ci_method: CIMethod = CIMethod.auto
    percentiles: tuple[int, ...] = (5, 25, 50, 75, 95)
    nan_policy: NanPolicy = NanPolicy.propagate
    target: float | None = None
    eps: float | None = None
    ddof: int = 1
    ess: int | None = None
    rng: int | np.random.Generator | None = None
    n_bootstrap: int = 10_000
    bootstrap: BootstrapMethod = BootstrapMethod.percentile

    # ergonomics
    def with_overrides(self, **changes) -> "StatsContext":
        r"""
        Return a shallow copy with selected fields replaced.

        Parameters
        ----------
        ``**changes`` :
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

    def eff_n(self, observed_len: int, finite_count: int | None = None) -> int:
        r"""
        Effective sample size :math:`n_\text{eff}` used by CI calculations.

        Priority is:
        1) explicit :attr:`ess`; 2) count of finite values if ``nan_policy="omit"``;
        3) declared :attr:`n` (fallback); else ``observed_len``.

        In symbols,

        .. math::

           n_\text{eff} =
           \begin{cases}
              \texttt{ess}, & \text{if provided},\\[4pt]
              \#\{i : x_i \text{ finite}\}, & \text{if nan policy = ``omit''},\\[4pt]
              \texttt{n}, & \text{otherwise}.
           \end{cases}

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
        Validate field ranges and cross-field consistency.

        Raises
        ------
        ValueError
            If any field is outside its allowed range or fields are inconsistent.
        """
        # Individual field validations
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must be in (0,1)")
        if any(p < 0 or p > 100 for p in self.percentiles):
            raise ValueError("percentiles must be in [0,100]")
        if self.n_bootstrap <= 0:
            raise ValueError("n_bootstrap must be > 0")
        if self.ddof < 0:
            raise ValueError("ddof must be >= 0")
        if self.eps is not None and self.eps <= 0:
            raise ValueError("eps must be positive")

        # Cross-field validations
        if self.ess is not None:
            if self.ess > self.n:
                raise ValueError(f"ess ({self.ess}) cannot exceed n ({self.n})")
            if self.ess <= 0:
                raise ValueError(f"ess must be positive, got {self.ess}")

        # Warn about small bootstrap samples (but don't fail)
        if self.n_bootstrap < 100:
            raise ValueError(f"n_bootstrap ({self.n_bootstrap}) should be >= 100 for reliable estimates")


@dataclass(frozen=True)
class _CIResult:
    r"""
    Internal helper that stores a confidence interval before converting to ``dict``.

    Attributes
    ----------
    confidence : float
        Confidence level :math:`\gamma`.
    method : str
        Human-readable label (e.g., ``"z"`` or ``"bootstrap-bca"``).
    low, high : float
        Interval endpoints :math:`(\ell, u)`.
    extras : Mapping[str, SupportsFloat]
        Optional diagnostics such as ``se`` or ``crit``.
    """
    confidence: float
    method: str
    low: float
    high: float
    extras: Mapping[str, SupportsFloat] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float | str]:
        result: dict[str, float | str] = {
            "confidence": float(self.confidence),
            "method": self.method.value if hasattr(self.method, 'value') else str(self.method),
            "low": float(self.low),
            "high": float(self.high),
        }
        if self.extras:
            result.update({key: float(value) for key, value in self.extras.items()})
        return result


@dataclass(frozen=True)
class ComputeResult:
    r"""
    Result from :meth:`StatsEngine.compute` with tracking of computation failures.

    Attributes
    ----------
    metrics : dict[str, Any]
        Successfully computed metric values, keyed by metric name.
    skipped : list[tuple[str, str]]
        List of (metric_name, reason) pairs for metrics that were skipped.
    errors : list[tuple[str, str]]
        List of (metric_name, error_message) pairs for metrics that raised errors.

    Examples
    --------
    >>> engine = StatsEngine([FnMetric("mean", mean)])
    >>> data = np.array([1.0, 2.0, 3.0])
    >>> ctx = StatsContext(n=3)
    >>> result = engine.compute(data, ctx)
    >>> result.metrics['mean']
    2.0
    >>> result.skipped
    []
    """

    metrics: dict[str, Any]
    skipped: list[tuple[str, str]] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  metrics={list(self.metrics.keys())},\n"
            f"  skipped={[name for name, _ in self.skipped]},\n"
            f"  errors={[name for name, _ in self.errors]}\n"
            ")"
    )

    def successful_metrics(self) -> set[str]:
        r"""
        Return names of successfully computed metrics.

        Returns
        -------
        set[str]
            Set of metric names present in :attr:`metrics`.
        """
        return set(self.metrics.keys())


MetricSet: TypeAlias = Iterable["Metric"]


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

    @property
    def name(self) -> str: ...

    def __call__(self, x: np.ndarray, ctx: StatsContext, /) -> Any: ...


_MetricT = TypeVar("_MetricT")  # Type parameter for FnMetric


@dataclass(frozen=True)
class FnMetric(Generic[_MetricT]):
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
    fn: Callable[[np.ndarray, StatsContext], _MetricT]
    doc: str = ""

    def __call__(self, x: np.ndarray, ctx: StatsContext) -> _MetricT:
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


class StatsEngine:
    r"""
    Orchestrator that evaluates a set of metrics over an input array.

    Given a collection of metric callables :math:`\{\phi_j\}_{j=1}^m` and an
    array :math:`x \in \mathbb{R}^n`, the engine returns the dictionary

    .. math::
       \{\phi_j(x, \texttt{ctx}) : j = 1,\dots,m\},

    while recording any skipped/failed evaluations for downstream inspection.

    Parameters
    ----------
    metrics : iterable of Metric
        Callables with a ``name`` and signature ``metric(x, ctx)``.

    Notes
    -----
    All metrics receive the *same* :class:`~mcframework.stats_engine.StatsContext`. Prefer field names that
    read well across multiple metrics and avoid collisions.

    Examples
    --------
    >>> eng = StatsEngine([FnMetric("mean", mean), FnMetric("std", std)])
    >>> x = np.array([1., 2., 3.])
    >>> result = eng.compute(x, StatsContext(n=len(x)))
    >>> result.metrics['mean']
    2.0
    >>> result.metrics['std']
    1.0
    """

    def __init__(self, metrics: MetricSet):
        self._metrics = list(metrics)

    def __repr__(self) -> str:
        metric_names = ", ".join(m.name for m in self._metrics)
        return f"{self.__class__.__name__}(metrics=[{metric_names}])"

    def compute(
        self,
        x: np.ndarray,
        ctx: StatsContext | None = None,
        select: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> ComputeResult:
        r"""
        Evaluate all registered metrics on ``x``.

        Parameters
        ----------
        x : ndarray
            Sample values.
        ctx : StatsContext, optional
            Context parameters. If None, one is built from ``**kwargs``.
        select : sequence of str, optional
            If given, compute only the metrics with these names.
        **kwargs : Any
            Used to build a StatsContext if ctx is None.
            Required: 'n' (int).
            Optional: 'confidence', 'ci_method', 'percentiles', etc.

        Returns
        -------
        ComputeResult
            Result object containing:

            - ``metrics``: Successfully computed metric values.
            - ``skipped``: List of (metric_name, reason) for skipped metrics.
            - ``errors``: List of (metric_name, error_message) for failed metrics.
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
        skipped: list[tuple[str, str]] = []
        errors: list[tuple[str, str]] = []

        for m in metrics_to_compute:
            try:
                result = m(x, ctx)

                # Handle None return (insufficient data)
                if result is None:
                    skipped.append((m.name, "insufficient data"))
                    logger.debug("Metric %s returned None (insufficient data), skipping", m.name)
                    continue

                out[m.name] = result

            except MissingContextError as e:
                # Skip metrics that require context fields not provided
                reason = str(e)
                skipped.append((m.name, reason))
                logger.debug("Skipping metric %s: %s", m.name, reason)
                continue
            except ValueError as e:
                # Also handle general ValueError for backward compatibility
                msg = str(e)
                if "Missing required context keys" in msg:
                    skipped.append((m.name, msg))
                    logger.debug("Skipping metric %s: %s", m.name, msg)
                    continue
                raise
            except Exception as e:  # pylint: disable=broad-exception-caught
                # Log and track unexpected errors (intentionally broad to not crash engine)
                error_msg = str(e)
                errors.append((m.name, error_msg))
                logger.exception("Error computing metric %s", m.name)
                continue

        return ComputeResult(metrics=out, skipped=skipped, errors=errors)


def _ensure_ctx(ctx: Any, x: np.ndarray) -> StatsContext:
    r"""
    Normalize arbitrary context inputs into a :class:`~mcframework.stats_engine.StatsContext`.

    Parameters
    ----------
    ctx : Any
        A :class:`~mcframework.stats_engine.StatsContext`, mapping, object with attributes, or ``None``.
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
    except TypeError as exc:
        raise TypeError(
            "ctx must be a StatsContext, dict, None, or an object with attributes"
        ) from exc
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
        Context whose :attr:`~mcframework.stats_engine.StatsContext.nan_policy` guides filtering.

    Returns
    -------
    tuple of ndarray
        ``(arr, finite_mask)`` where ``arr`` is the possibly filtered sample and
        ``finite_mask`` marks finite elements in the original sample.

    Raises
    ------
    ValueError
        If an unknown :attr:`~mcframework.stats_engine.StatsContext.nan_policy` is supplied.
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
        Effective sample size according to :meth:`~mcframework.stats_engine.StatsContext.eff_n`.
    """
    arr, finite = _clean(x, ctx)
    # finite is a boolean mask; pass the count of True values
    finite_count = int(np.sum(finite))
    return ctx.eff_n(observed_len=arr.size, finite_count=finite_count)


def mean(x: np.ndarray, ctx: StatsContext) -> float | None:
    r"""
    Sample mean :math:`\bar X = \frac{1}{n}\sum_{i=1}^n x_i`.

    Parameters
    ----------
    x : ndarray
        Input sample.
    ctx : StatsContext
        If ``nan_policy="omit"``, non-finite values are excluded.

    Returns
    -------
    float or None
        Estimate of :math:`\mathbb{E}[X]`, or ``None`` if the cleaned sample is empty.

    Notes
    -----
    The averaging is performed on the filtered array returned by :func:`~mcframework.stats_engine._clean`,
    so the effective sample size corresponds to the number of finite values that
    survive the configured NaN policy.

    Examples
    --------
    >>> mean(np.array([1, 2, 3]), StatsContext(n=3))
    2.0
    """
    ctx = _ensure_ctx(ctx, x)
    arr, _ = _clean(x, ctx)
    return float(np.mean(arr)) if arr.size else None


def std(x: np.ndarray, ctx: StatsContext) -> float | None:
    r"""
    Sample standard deviation with Bessel correction.

    The estimator is

    .. math::
       s = \sqrt{\frac{1}{n_\text{eff} - \texttt{ddof}}
       \sum_{i=1}^{n_\text{eff}} (x_i - \bar X)^2 }.

    Parameters
    ----------
    x : ndarray
        Input sample.
    ctx : StatsContext
        Uses :attr:`~mcframework.stats_engine.StatsContext.ddof` (default 1).
        If ``nan_policy="omit"``, non-finite values are excluded.
    Returns
    -------
    float or None
        :math:`s` when :math:`n_\text{eff} > \texttt{ddof}`, else ``None``.

    Examples
    --------
    >>> std(np.array([1, 2, 3]), {})
    1.0
    """
    ctx = _ensure_ctx(ctx, x)
    arr, finite = _clean(x, ctx)
    # finite is a boolean mask; pass the count of True values
    finite_count = int(np.sum(finite))
    n_eff = ctx.eff_n(observed_len=arr.size, finite_count=finite_count)
    if n_eff <= 1:
        return None
    return float(np.std(arr, ddof=ctx.ddof))


def percentiles(x: np.ndarray, ctx: StatsContext) -> dict[int, float]:
    r"""
    Empirical percentiles evaluated on the cleaned sample.

    For each :math:`p \in \mathcal{P}` we compute the empirical quantile
    :math:`Q_p(x)` using :func:`numpy.percentile` (linear interpolation).

    Parameters
    ----------
    x : ndarray
        Input sample.
    ctx : StatsContext
        Uses :attr:`~mcframework.stats_engine.StatsContext.percentiles` and
        :attr:`~mcframework.stats_engine.StatsContext.nan_policy`.

    Returns
    -------
    dict[int, float]
        Mapping :math:`p \mapsto Q_p(x)`.

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

    .. math::
       \text{skew}(x) =
       \frac{1}{n_\text{eff}} \sum_i
       \left(\frac{x_i - \bar X}{s}\right)^3.

    Parameters
    ----------
    x : ndarray
        Input sample.
    ctx : StatsContext
        Uses :attr:`~mcframework.stats_engine.StatsContext.nan_policy`.

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

    .. math::
       \text{kurt}(x) =
       \frac{1}{n_\text{eff}} \sum_i
       \left(\frac{x_i - \bar X}{s}\right)^4 - 3.

    Parameters
    ----------
    x : ndarray
        Input sample.
    ctx : StatsContext
        Uses :attr:`~mcframework.stats_engine.StatsContext.nan_policy`.

    Returns
    -------
    float
        Excess kurtosis (``0.0`` if :math:`n_\text{eff} \le 3`).

    Notes
    -----
    Uses :func:`scipy.stats.kurtosis` with ``fisher=True, bias=False``.

    Examples
    --------
    >>> round(kurtosis(np.array([1, 2, 3, 4.0]), {}), 1)
    -1.2
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
    to :attr:`~mcframework.stats_engine.StatsContext.ci_method` and :math:`n_\text{eff}`.

    Parameters
    ----------
    x : ndarray
        Input sample.
    ctx : StatsContext or Mapping
        Configuration supplying at least :attr:`~mcframework.stats_engine.StatsContext.n`. Additional
        fields such as :attr:`~mcframework.stats_engine.StatsContext.confidence`,
        :attr:`~mcframework.stats_engine.StatsContext.ddof`, and
        :attr:`~mcframework.stats_engine.StatsContext.ci_method` refine the interval.

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
        return _CIResult(
            confidence=ctx.confidence,
            method=ctx.ci_method,
            low=float("nan"),
            high=float("nan"),
        ).as_dict()

    n_eff = _effective_sample_size(arr, ctx)  # counts after cleaning if 'omit'
    if n_eff < 2:
        return _CIResult(
            confidence=ctx.confidence,
            method=ctx.ci_method,
            low=float("nan"),
            high=float("nan"),
            extras={"se": float("nan"), "crit": float("nan")},
        ).as_dict()

    mu = float(np.mean(arr))

    s = float(np.std(arr, ddof=getattr(ctx, "ddof", 1)))
    if s == 0.0:
        # degenerate data -> zero SE -> CI collapses to point
        se = 0.0
    else:
        se = s / np.sqrt(n_eff)

    crit, method = autocrit(ctx.confidence, n_eff, ctx.ci_method)

    return _CIResult(
        confidence=ctx.confidence,
        method=method,
        low=mu - crit * se,
        high=mu + crit * se,
        extras={"se": se, "crit": crit},
    ).as_dict()


def _bootstrap_means(
    arr: np.ndarray, n_resamples: int, rng: np.random.Generator
) -> np.ndarray:
    r"""
    Generate bootstrap replicates of the sample mean.

    Parameters
    ----------
    arr : ndarray
        Cleaned sample values.
    n_resamples : int
        Number of bootstrap resamples.
    rng : numpy.random.Generator
        Random generator used to draw resamples.

    Returns
    -------
    ndarray
        Array of length ``n_resamples`` containing :math:`\{\bar X_b^*\}`.
    """
    n = arr.size
    idx = rng.integers(0, n, size=(n_resamples, n), endpoint=False)
    return arr[idx].mean(axis=1)


def _bca_bias_correction(arr: np.ndarray, bootstrap_means: np.ndarray) -> float:
    r"""
    Compute :math:`z_0`, the bias-correction factor for BCa bootstrap.

    The bias correction accounts for median bias in the bootstrap distribution.
    It is computed as:

    .. math::
       z_0 = \Phi^{-1}(P(\bar X_b^* < \bar X))

    where :math:`\Phi^{-1}` is the inverse standard normal CDF, :math:`\bar X`
    is the observed sample mean, and :math:`\bar X_b^*` are the bootstrap means.

    Parameters
    ----------
    arr : ndarray
        Original sample values.
    bootstrap_means : ndarray
        Bootstrap replicate means.

    Returns
    -------
    float
        Bias correction factor :math:`z_0`.

    Notes
    -----
    The proportion is clipped to :math:`[\epsilon, 1-\epsilon]` where
    :math:`\epsilon` = ``_SMALL_PROB_BOUND`` to prevent numerical issues
    in the inverse error function.
    """
    m_hat = float(np.mean(arr))
    prop = float(np.sum(bootstrap_means < m_hat)) / bootstrap_means.size
    # Clip to prevent log(0) in erfinv (which is used by inverse normal CDF)
    prop = np.clip(prop, _SMALL_PROB_BOUND, 1 - _SMALL_PROB_BOUND)
    # Convert proportion to z-score using inverse normal CDF
    # erfinv is related to norm.ppf via: norm.ppf(p) = sqrt(2) * erfinv(2*p - 1)
    return float(np.sqrt(2) * erfinv(2 * prop - 1))


def _bca_acceleration(arr: np.ndarray) -> float:
    r"""
    Compute :math:`a`, the acceleration factor for BCa bootstrap via jackknife.

    The acceleration factor measures the rate of change of the standard error
    of the estimator as a function of the true parameter value. It is computed
    using the jackknife (leave-one-out) means:

    .. math::
       a = \frac{\sum_{i=1}^n d_i^3}{6 \left(\sum_{i=1}^n d_i^2\right)^{3/2}}

    where :math:`d_i = \bar X_{(-i)} - \bar{\bar X}_{(\cdot)}` is the deviation
    of each jackknife mean from their overall mean.

    Parameters
    ----------
    arr : ndarray
        Original sample values.

    Returns
    -------
    float
        Acceleration factor :math:`a`.

    Notes
    -----
    The formula follows Efron & Tibshirani (1993), "An Introduction to the Bootstrap".
    A small constant is added to the denominator to prevent division by zero.

    References
    ----------
    .. [1] Efron, B., & Tibshirani, R. J. (1993). An Introduction to the Bootstrap.
           Chapman & Hall/CRC.
    """
    # Jackknife: compute leave-one-out means efficiently
    # If sum(arr) = S and arr[i] = x_i, then mean without x_i = (S - x_i)/(n-1)
    s = np.sum(arr, dtype=float)
    jack_means = (s - arr) / (arr.size - 1)

    # Deviations from jackknife mean
    d = jack_means - float(np.mean(jack_means))

    # Acceleration formula from Efron & Tibshirani (1993)
    # The constant 6.0 comes from the theoretical derivation
    numerator = float(np.sum(d**3))
    denominator = _BCA_JACKKNIFE_DENOMINATOR * (np.sum(d**2) ** 1.5) + _SMALL_VARIANCE_BOUND
    return numerator / denominator


def _compute_bca_interval(arr: np.ndarray, means: np.ndarray, confidence: float) -> tuple[float, float]:
    r"""
    Compute bias-corrected and accelerated (BCa) bootstrap interval.

    The BCa method adjusts the bootstrap percentiles to account for:

    1. **Bias correction** (:math:`z_0`): Corrects for median bias in the bootstrap
       distribution relative to the observed statistic.
    2. **Acceleration** (:math:`a`): Adjusts for non-constant variance of the
       estimator as the true parameter varies.

    The adjusted percentiles are computed via the transformation:

    .. math::
       p = \Phi\left(z_0 + \frac{z_0 + z_\alpha}{1 - a(z_0 + z_\alpha)}\right)

    where :math:`\Phi` is the standard normal CDF and :math:`z_\alpha` is the
    :math:`\alpha`-level quantile.

    Parameters
    ----------
    arr : ndarray
        Original sample values.
    means : ndarray
        Bootstrap replicate means.
    confidence : float
        Confidence level in (0, 1).

    Returns
    -------
    tuple[float, float]
        (lower_bound, upper_bound)

    See Also
    --------
    _bca_bias_correction : Computes the bias correction factor :math:`z_0`.
    _bca_acceleration : Computes the acceleration factor :math:`a`.

    References
    ----------
    .. [1] Efron, B. (1987). "Better Bootstrap Confidence Intervals".
           Journal of the American Statistical Association, 82(397), 171-185.
    """
    # Compute BCa adjustment factors
    z0 = _bca_bias_correction(arr, means)
    a = _bca_acceleration(arr)

    # Standard normal quantiles for the confidence interval
    zlo = float(norm.ppf((1 - confidence) / 2))
    zhi = float(norm.ppf(1 - (1 - confidence) / 2))

    # BCa-adjusted percentiles using the transformation formula
    # Φ(z0 + (z0 + z) / (1 - a(z0 + z))) converted to percentile scale
    def adjusted_percentile(z: float) -> float:
        """Apply BCa transformation to convert z-score to adjusted percentile."""
        num = z0 + z
        den = 1.0 - a * num
        return float(norm.cdf(z0 + num / den)) * 100.0

    # Compute adjusted percentile bounds and clip to valid range
    p_lo = np.clip(adjusted_percentile(zlo), 0, 100)
    p_hi = np.clip(adjusted_percentile(zhi), 0, 100)

    # Extract the interval from the bootstrap distribution
    low, high = np.percentile(means, [p_lo, p_hi])

    return float(low), float(high)


def ci_mean_bootstrap(x: np.ndarray, ctx: StatsContext) -> dict[str, float | str] | None:
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

        - :attr:`~mcframework.stats_engine.StatsContext.confidence`
            Confidence level :math:`\in (0, 1)`.
        - :attr:`~mcframework.stats_engine.StatsContext.n_bootstrap`
            Number of bootstrap resamples (default ``10_000``).
        - :attr:`~mcframework.stats_engine.StatsContext.nan_policy`
            Whether to omit non-finite values before bootstrapping.
        - :attr:`~mcframework.stats_engine.StatsContext.bootstrap`
            Flavor (``"percentile"`` or ``"bca"``).
        - :attr:`~mcframework.stats_engine.StatsContext.rng`
            Seed or generator for reproducibility.

    Returns
    -------
    dict or None
        Mapping with keys:

        - ``confidence`` : float
          The requested confidence level.
        - ``method`` : str
          ``"bootstrap-percentile"`` or ``"bootstrap-bca"``.
        - ``low`` : float
          Lower bound of the CI.
        - ``high`` : float
          Upper bound of the CI.

        Returns ``None`` if the sample is empty after cleaning.

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
    'bootstrap-percentile'
    >>> 1.5 < result["low"] < result["high"] < 4.5
    True
    """
    ctx = _ensure_ctx(ctx, x)
    arr, _ = _clean(x, ctx)
    if arr.size == 0:
        return None

    n_resamples = int(ctx.n_bootstrap)
    g = ctx.get_generators()
    means = _bootstrap_means(arr, n_resamples, g)
    loq, hiq = ctx.q_bound()
    method = ctx.bootstrap

    # Use percentile method for small samples or when explicitly requested
    if method == "percentile" or arr.size < 3:
        low, high = np.percentile(means, [loq, hiq])
        return _CIResult(
            confidence=ctx.confidence,
            method="bootstrap-percentile",
            low=low,
            high=high,
        ).as_dict()

    # BCa method
    low, high = _compute_bca_interval(arr, means, ctx.confidence)
    return _CIResult(
        confidence=ctx.confidence,
        method="bootstrap-bca",
        low=low,
        high=high,
    ).as_dict()


def ci_mean_chebyshev(x: np.ndarray, ctx: StatsContext) -> dict[str, float | str] | None:
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
        Requires :attr:`~mcframework.stats_engine.StatsContext.n` and may specify
        :attr:`~mcframework.stats_engine.StatsContext.confidence`.

    Returns
    -------
    dict[str, float | str] or None
        Mapping with keys ``confidence``, ``method``, ``low``, and ``high``,
        or ``None`` if :math:`n_\text{eff} < 2`.
    """
    ctx = _ensure_ctx(ctx, x)
    n_eff = _effective_sample_size(x, ctx)
    if n_eff < 2:
        return None
    mu = mean(x, ctx)
    s = std(x, ctx)
    # Handle case where mean or std returns None
    if mu is None or s is None:
        return None
    k = 1.0 / np.sqrt(max(_SMALL_VARIANCE_BOUND, 1.0 - ctx.confidence))  # 1/sqrt(alpha)
    half = k * s / np.sqrt(n_eff)
    return _CIResult(
        confidence=ctx.confidence,
        method="chebyshev",
        low=mu - half,
        high=mu + half,
    ).as_dict()


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
        Must supply :attr:`~mcframework.stats_engine.StatsContext.eps` (target half-width) and may
        override :attr:`~mcframework.stats_engine.StatsContext.confidence`.

    Returns
    -------
    int
        Minimum integer :math:`n_\text{eff}`.

    Examples
    --------
    >>> chebyshev_required_n(np.array([1., 2., 3.]), {"eps": 0.5, "confidence": 0.9})
    41
    """
    ctx = _ensure_ctx(ctx, x)
    if ctx.eps is None:
        raise MissingContextError("chebyshev_required_n requires ctx.eps")
    if ctx.eps <= 0:
        raise ValueError("ctx.eps must be positive")
    s = std(x, ctx)
    k = 1.0 / np.sqrt(max(_SMALL_VARIANCE_BOUND, 1.0 - ctx.confidence))
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
        Requires :attr:`~mcframework.stats_engine.StatsContext.target` and
        :attr:`~mcframework.stats_engine.StatsContext.eps`. The declared
        :attr:`~mcframework.stats_engine.StatsContext.n` influences the context but does not enter
        directly into the bound.

    Returns
    -------
    float
        Upper bound in :math:`[0, 1]`.
    """
    ctx = _ensure_ctx(ctx, x)
    if ctx.target is None:
        raise MissingContextError("markov_error_prob requires ctx.target")
    if ctx.eps is None:
        raise MissingContextError("markov_error_prob requires ctx.eps")
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
        Requires :attr:`~mcframework.stats_engine.StatsContext.target`.

    Returns
    -------
    float
        Estimated bias :math:`\bar X - \theta`.
    """
    ctx = _ensure_ctx(ctx, x)
    if ctx.target is None:
        raise MissingContextError("bias_to_target requires ctx.target")
    sample_mean = mean(x, ctx)
    if sample_mean is None:
        raise InsufficientDataError("bias_to_target requires non-empty data after cleaning")
    return float(sample_mean - ctx.target)


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
        Requires :attr:`~mcframework.stats_engine.StatsContext.target`.

    Returns
    -------
    float
        Estimated mean squared error relative to ``target``.
    """
    ctx = _ensure_ctx(ctx, x)
    if ctx.target is None:
        raise MissingContextError("mse_to_target requires ctx.target")
    arr, _ = _clean(x, ctx)
    return float(np.mean((arr - ctx.target) ** 2))


def build_default_engine(
    include_dist_free: bool = True,
    include_target_bounds: bool = True,
) -> StatsEngine:
    r"""
    Construct a :class:`~mcframework.stats_engine.StatsEngine` with a practical set of metrics.

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
        FnMetric("mean", mean, "Sample mean"),
        FnMetric("std", std, "Sample standard deviation"),
        FnMetric("percentiles", percentiles, "Percentiles over the sample"),
        FnMetric("skew", skew, "Fisher skewness (unbiased)"),
        FnMetric("kurtosis", kurtosis, "Excess kurtosis (unbiased)"),
        FnMetric("ci_mean", ci_mean, "z/t CI for the mean"),
        FnMetric("ci_mean_bootstrap", ci_mean_bootstrap, "Bootstrap CI for the mean"),
    ]
    if include_dist_free:
        metrics.extend(
            [
                FnMetric(
                    "ci_mean_chebyshev", ci_mean_chebyshev, "Chebyshev bound CI for the mean"
                ),
                FnMetric(
                    "chebyshev_required_n", chebyshev_required_n, "Required n under Chebyshev to reach eps"
                ),
            ]
        )
    if include_target_bounds:
        metrics.extend(
            [
                FnMetric("markov_error_prob", markov_error_prob, "Markov bound P(|X-target|>=eps)"),
                FnMetric("bias_to_target", bias_to_target, "Bias relative to target"),
                FnMetric("mse_to_target", mse_to_target, "Mean squared error to target"),
            ]
        )
    return StatsEngine(metrics)


# Build a default engine at import time
DEFAULT_ENGINE = build_default_engine(
    include_dist_free=True,
    include_target_bounds=True,
)

__all__ = [
    "CIMethod",
    "NanPolicy",
    "BootstrapMethod",
    "StatsContext",
    "ComputeResult",
    "Metric",
    "MetricSet",
    "FnMetric",
    "StatsEngine",
    "MissingContextError",
    "InsufficientDataError",
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
