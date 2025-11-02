r"""
mcframework.stats_engine
================
Statistical metrics and the engine used by the Monte Carlo framework.

This module exposes a small protocol for metrics (:class:`Metric`), a light adapter
(:class:`FnMetric`), and a coordinator (:class:`StatsEngine`) that evaluates multiple
metrics over an input array. Common metrics include mean, standard deviation,
percentiles, skew, kurtosis, and several confidence-interval and target-aware
utilities.

See Also
--------
:py:mod:`mcframework.utils` : Critical values and the CI selector :func:`mcframework.utils.autocrit`.
"""


# DEV NOTE:
#===========================================================================
# The type checker throws a fit since x is ndarray and the checker can't verify
# that numpy/scipy functions accept that. So, we're suppressing the error
# with type: ignore[arg-type] where needed.
#===========================================================================


from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Literal, Protocol

import numpy as np
from scipy.stats import kurtosis as sp_kurtosis
from scipy.stats import skew as sp_skew

from .utils import autocrit


NanPolicy = Literal["propagate", "omit"]


class Metric(Protocol):
    r"""
    Protocol for metric callables used by :class:`StatsEngine`.

    A metric exposes a ``name`` attribute and a ``__call__`` that accepts
    the data array and a context mapping.

    Attributes
    ----------
    name : str
        Human-readable key under which the metric's value is returned.

    Notes
    -----
    The callable signature is::

        __call__(x: np.ndarray, ctx: Dict[str, Any]) -> Any
    """

    name: str

    def __call__(self, x: np.ndarray, ctx: Stats) -> Any: ...


@dataclass(frozen=True)
class FnMetric:
    r"""
    Lightweight adapter: give any ``fn(x, ctx)`` a ``name`` so engines can collect it.

    Attributes
    ----------
    name : str
        Key under which the metric result is stored.
    fn : callable
        Function with signature ``fn(x: ndarray, ctx: dict) -> Any``.
    """

    name: str
    fn: Callable[[np.ndarray, Dict[str, Any]], Any]

    def __call__(self, x: np.ndarray, ctx: Dict[str, Any]) -> Any:
        r"""
        Compute the metric.

        Parameters
        ----------
        x : ndarray
            Input sample.
        ctx : dict
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
    Small orchestrator that computes a dictionary of metrics over an array.

    Parameters
    ----------
    metrics : iterable of Metric
        Callables with a ``name`` attribute and signature ``m(x, ctx)``.

    Examples
    --------
    >>> eng = StatsEngine([FnMetric("mean", mean), FnMetric("std", std)])
    >>> eng.compute(np.array([1, 2, 3]), n=3)
    {'mean': 2.0, 'std': 1.0}
    """

    def __init__(self, metrics: Iterable[Metric]):
        self._metrics = list(metrics)

    def compute(self, x: np.ndarray, **ctx) -> Dict[str, Any]:
        r"""
        Evaluate all registered metrics on ``x``.

        Parameters
        ----------
        x : ndarray
            Sample values.
        **ctx
            Extra keyword arguments forwarded to each metric.

        Returns
        -------
        dict
            Mapping from metric name to computed value.

        Notes
        -----
        The same context is passed to every metric. Choose distinct names if you
        add custom metric-specific parameters.
        """
        out: Dict[str, Any] = {}
        for m in self._metrics:
            out[m.name] = m(x, ctx)  # type: ignore[arg-type]
        return out


def _effective_sample_size(x: np.ndarray, ctx: Dict[str, Any]) -> int:
    r"""
    Effective sample size used by CI calculations.

    Parameters
    ----------
    x : ndarray
        Input sample.
    ctx : dict
        Context with optional keys:
        - ``nan_policy`` : {"propagate", "omit"}, default "propagate"
          If "omit", use the count of finite values in ``x``.
        - ``n`` : int, optional
          Override for sample size when ``nan_policy != "omit"``.

    Returns
    -------
    int
        Effective ``n``.
    """
    _validate_ctx(ctx, required=set(), optional={"nan_policy", "n"})
    nan_policy = ctx.get("nan_policy", "propagate")
    if nan_policy == "omit":
        return int(np.isfinite(x).sum())  # sum of non-nan entries
    return int(ctx.get("n", x.size))


def mean(x, ctx):
    r"""
    Sample mean.

    Parameters
    ----------
    x : ndarray
        Input sample.
    ctx : dict
        Context; if ``nan_policy='omit'``, NaNs are ignored.

    Returns
    -------
    float
        :math:`\bar X = \frac{1}{n}\sum_i x_i`.

    Notes
    -----
    If ``nan_policy='omit'``, the effective size :math:`n` counts finite entries only.

    Examples
    --------
    >>> mean(np.array([1, 2, 3]), {})
    2.0
    """
    _validate_ctx(ctx, required=set(), optional={"nan_policy"})
    if ctx.get("nan_policy") == "omit":
        return float(np.nanmean(x))
    return float(np.mean(x))


def std(x, ctx):
    r"""
    Sample standard deviation with Bessel's correction.
    
    Parameters
    ----------
    x : ndarray
        Input sample.
    ctx : dict
        Context; if ``nan_policy='omit'``, NaNs are ignored.

    Returns
    -------
    float
        :math:`s = \sqrt{\frac{1}{n-1}\sum_i (x_i-\bar X)^2}` (``0.0`` if ``n<=1``).

    Examples
    --------
    >>> std(np.array([1, 2, 3]), {})
    1.0
    """
    _validate_ctx(ctx, required=set(), optional={"nan_policy"})
    if ctx.get("nan_policy") == "omit":
        n_eff = int(np.isfinite(x).sum())
        if n_eff <= 1:
            return 0.0
        return float(np.nanstd(x, ddof=1))
    if np.asarray(x).size <= 1:
        return 0.0
    return float(np.std(x, ddof=1))


def percentiles(x, ctx):
    r"""
    Percentiles of the sample.

    Parameters
    ----------
    x : ndarray
        Input sample.
    ctx : dict
        - ``percentiles`` : iterable of int, default ``(5, 25, 50, 75, 95)``
        - ``percentile_method`` : str, optional
          Method for :func:`numpy.percentile` (e.g. ``"linear"``, ``"median_unbiased"``).
        - ``nan_policy`` : {"propagate", "omit"}, default "propagate"

    Returns
    -------
    dict[int, float]
        Mapping from requested percentile to value.

    Examples
    --------
    >>> percentiles(np.array([0., 1., 2., 3.]), {"percentiles": (50, 75)})
    {50: 1.5, 75: 2.25}
    """
    _validate_ctx(
        ctx,
        required=set(),
        optional={"percentiles", "percentile_method", "nan_policy"}
    )
    ps = tuple(ctx.get("percentiles", (5, 25, 50, 75, 95)))
    if any((not np.isfinite(p)) or (p < 0) or (p > 100) for p in ps):
        raise ValueError(f"percentiles must be in [0,100], got {ps}")
    arr = np.asarray(x)
    if ctx.get("nan_policy") == "omit":
        arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {p: float("nan") for p in ps}
    method = ctx.get("percentile_method")
    if method is None:
        return {p: float(np.percentile(arr, p)) for p in ps}
    return {p: float(np.percentile(arr, p, method=method)) for p in ps}


def skew(x, ctx):
    r"""
    Unbiased sample skewness.

    Uses :func:`scipy.stats.skew` with ``bias=False``.

    Parameters
    ----------
    x : ndarray
        Input sample.
    ctx : dict
        - ``nan_policy`` : {"propagate", "omit"}, default "propagate"

    Returns
    -------
    float
        Fisher–Pearson standardized third central moment (0.0 if ``n<=2``).

    Examples
    --------
    >>> round(skew(np.array([1, 2, 3, 10.0]), {}), 3) > 0
    True
    """
    _validate_ctx(ctx, required=set(), optional={"nan_policy"})
    if ctx.get("nan_policy") == "omit":
        n_eff = int(np.isfinite(x).sum())
        if n_eff <= 2:
            return 0.0
        return float(sp_skew(x, bias=False, nan_policy="omit")) # type: ignore[arg-type]
    if np.asarray(x).size <= 2:
        return 0.0
    return float(sp_skew(x, bias=False)) # type: ignore[arg-type]


def kurtosis(x, ctx):
    r"""
    Unbiased sample **excess** kurtosis (Fisher definition).

    Uses :func:`scipy.stats.kurtosis` with ``fisher=True, bias=False``.

    Parameters
    ----------
    x : ndarray
        Input sample.
    ctx : dict
        - ``nan_policy`` : {"propagate", "omit"}, default "propagate"

    Returns
    -------
    float
        Excess kurtosis (0.0 if ``n<=3``).

    Examples
    --------
    >>> round(kurtosis(np.array([1, 2, 3, 4.0]), {}), 6)
    -1.200000
    """

    _validate_ctx(ctx, required=set(), optional={"nan_policy"})
    if ctx.get("nan_policy") == "omit":
        n_eff = int(np.isfinite(x).sum())
        if n_eff <= 3:
            return 0.0
        return float(sp_kurtosis(x, fisher=True, bias=False, nan_policy="omit"))
    if np.asarray(x).size <= 3:
        return 0.0
    return float(sp_kurtosis(x, fisher=True, bias=False))


def ci_mean(x, ctx):
    r"""
    Confidence interval for :math:`\E[X]` using z/t critical values.

    Let :math:`\Xbar` be the sample mean and :math:`\SE = s/\sqrt{n_{\text{eff}}}`.
    The CI is

    .. math::
       \Xbar \pm c\,\SE,

    where :math:`c` is chosen by :func:`mcframework.utils.autocrit`.

    Parameters
    ----------
    x : ndarray
        Input sample.
    ctx : dict
        ``confidence`` (float, default ``0.95``),
        ``ci_method`` ({``"auto"``, ``"z"``, ``"t"``}, default ``"auto"``),
        ``nan_policy`` ({``"propagate"``, ``"omit"``}, default ``"propagate"``).

    Returns
    -------
    dict or None
        ``{"confidence","method","se","crit","low","high"}`` or ``None`` if ``n_eff < 2``.


    """
    _validate_ctx(ctx, required=set(), optional={"confidence", "ci_method", "nan_policy"})
    n_eff = _effective_sample_size(x, ctx)
    if n_eff < 2:
        return None
    confidence = float(ctx.get("confidence", 0.95))
    s = std(x, ctx) # type: ignore[arg-type]
    se = s / np.sqrt(n_eff)
    crit, kind = autocrit(confidence, n_eff, ctx.get("ci_method", "auto"))
    mu = mean(x, ctx)
    return {
        "confidence": confidence,
        "method": kind,
        "se": float(se),
        "crit": float(crit),
        "low": float(mu - crit * se),
        "high": float(mu + crit * se),
    }



def ci_mean_bootstrap(x, ctx):
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
    ctx : dict
        - ``confidence`` : float, default ``0.95``
          Confidence level in :math:`(0, 1)`.
        - ``n_bootstrap`` : int, default ``10000``
          Number of bootstrap resamples.
        - ``nan_policy`` : {"propagate", "omit"}, default ``"propagate"``
          If ``"omit"``, filter out non-finite values before bootstrapping.
        - ``random_state`` : int, optional
          Seed for the random number generator.

    Returns
    -------
    dict
        Mapping with keys:

        - ``confidence`` : float
          The requested confidence level.
        - ``method`` : str
          Always ``"bootstrap"``.
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
    >>> result = ci_mean_bootstrap(x, {"confidence": 0.9, "n_bootstrap": 5000, "random_state": 42})
    >>> result["method"]
    'bootstrap'
    >>> 1.5 < result["low"] < result["high"] < 4.5
    True
    """

    _validate_ctx(
        ctx,
        required=set(),
        optional={"confidence", "n_bootstrap", "nan_policy", "random_state"})
    confidence = float(ctx.get("confidence", 0.95))
    n_bootstrap = int(ctx.get("n_bootstrap", 10_000))
    
    arr = np.asarray(x)
    if ctx.get("nan_policy") == "omit":
        x = arr[np.isfinite(arr)]
        
    n = np.asarray(x).size
    if n == 0:
        return {
            "confidence": confidence,
            "method": "bootstrap",
            "low": float("nan"),
            "high": float("nan"),
        }
    
    rng = np.random.default_rng(ctx.get("random_state"))
    bootstrap_means = np.array([rng.choice(x, size=n, replace=True).mean()
                                for _ in range(n_bootstrap)])
    
    alpha = 1 - confidence
    low, high = np.percentile(bootstrap_means, [100*alpha/2, 100*(1-alpha/2)])
    
    return {
        "confidence": confidence,
        "method": "bootstrap",
        "low": float(low),
        "high": float(high),
    }


def ci_mean_chebyshev(x, ctx):
    r"""
    Distribution-free CI for :math:`\mathbb{E}[X]` via Chebyshev’s inequality.

    This is a conservative, distribution-free alternative to :func:`ci_mean`
    that requires no assumptions beyond finite variance. It is typically
    much wider than a normal/t CI, but it is valid for any distribution.

    For :math:`\delta = 1-\text{confidence}`, choose
    :math:`z = 1/\sqrt{\delta}` so that

    .. math::
       \Pr\!\left(\,|\bar X - \mu| \ge z\,SE\,\right) \le \delta,
       \qquad SE = \frac{s}{\sqrt{n}}.

    Parameters
    ----------
    x : `ndarray`
        Input sample.
    ctx : dict
        - ``n`` : `int`
            Effective sample size to use for the bound.
        - ``confidence`` : float, default ``0.95``

    Returns
    -------
    dict or None
        Same keys as :func:`ci_mean`, with ``method="chebyshev"``.
    """

    _validate_ctx(ctx, required={"n"}, optional={"confidence"})
    n = int(ctx["n"])
    if n <= 0:
        return None
    confidence = float(ctx.get("confidence", 0.95))
    delta = max(1e-12, 1.0 - confidence)
    s = float(np.std(x, ddof=1)) if np.asarray(x).size > 1 else 0.0
    se = s / np.sqrt(max(1, n))
    z = 1.0 / np.sqrt(delta)
    mu_hat = float(np.mean(x))
    return {
        "confidence": confidence,
        "method": "chebyshev",
        "se": float(se),
        "crit": float(z),  # so half-width = crit * SE
        "low": float(mu_hat - z * se),
        "high": float(mu_hat + z * se),
    }


def chebyshev_required_n(x, ctx):
    r"""
    Required :math:`n` for Chebyshev CI half-width :math:`\le \varepsilon`.

    The half-width is :math:`z\,SE = \dfrac{s}{\sqrt{n\delta}}` with
    :math:`z = 1/\sqrt{\delta}` and :math:`\delta = 1-\text{confidence}`. Solve:

    .. math::
       n \;\ge\; \frac{s^2}{\varepsilon^2\,\delta}.

    Parameters
    ----------
    x : ndarray
        Input sample.
    ctx : dict
        - ``eps`` : float
            Target half-width :math:`\varepsilon` (must be > 0).
        - ``confidence`` : float, default ``0.95``

    Returns
    -------
    int or None
        Minimum required :math:`n` (ceiled), or ``None`` if ``eps`` invalid.

    Examples
    --------
    >>> chebyshev_required_n(np.array([1., 2., 3.]), {"eps": 0.5, "confidence": 0.9})
    8
    """
    _validate_ctx(ctx, required={"eps"}, optional={"confidence"})
    eps = ctx.get("eps")
    if eps is None or eps <= 0:
        return None
    confidence = float(ctx.get("confidence", 0.95))
    delta = max(1e-12, 1.0 - confidence)
    s = float(np.std(x, ddof=1)) if np.asarray(x).size > 1 else 0.0
    n_req = int(np.ceil((s * s) / (eps * eps * delta)))
    return n_req


def markov_error_prob(x, ctx):
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
    ctx : dict
        - ``target`` : float
            The true/reference value :math:`\theta`.
        - ``eps`` : float
            Tolerance :math:`\varepsilon` (> 0).
        - ``n`` : int
            Effective sample size.

    Returns
    -------
    float or None
        Upper bound in :math:`[0,1]`, or ``None`` if inputs missing/invalid.
    """
    _validate_ctx(ctx, required={"target", "eps", "n"}, optional=set())
    theta = ctx.get("target")
    eps = ctx.get("eps")
    if theta is None or eps is None or eps <= 0:
        return None
    n = int(ctx["n"])
    mu_hat = float(np.mean(x))
    s = float(np.std(x, ddof=1)) if np.asarray(x).size > 1 else 0.0
    bias2 = (mu_hat - float(theta)) ** 2
    mse = (s * s) / max(1, n) + bias2
    bound = float(min(1.0, mse / (eps * eps)))
    if bound >= 0.95:
        logging.warning(f"Markov bound {bound} >= 0.95, very loose estimate")
    return bound


def bias_to_target(x, ctx):
    r"""
    Bias of the sample mean relative to a target :math:`\theta`.

    Parameters
    ----------
    x : ndarray
        Input sample.
    ctx : dict
        - ``target`` : float

    Returns
    -------
    float or None
        :math:`\Xbar - \theta`, or ``None`` if ``target`` is missing.
    """
    _validate_ctx(ctx, required={"target"}, optional=set())
    theta = ctx.get("target")
    if theta is None:
        return None
    return float(np.mean(x) - theta)


def mse_to_target(x, ctx):
    r"""
    Mean squared error of :math:`\Xbar` relative to a target :math:`\theta`.

    Approximated by

    .. math::
       \mathrm{MSE}(\Xbar) \approx \frac{s^2}{n} + (\Xbar - \theta)^2.

    Parameters
    ----------
    x : ndarray
        Input sample.
    ctx : dict
        - ``target`` : float
        - ``n`` : int

    Returns
    -------
    float or None
        Estimated MSE, or ``None`` if ``target`` is missing.
    """
    _validate_ctx(ctx, required={"target", "n"}, optional=set())
    theta = ctx.get("target")
    if theta is None:
        return None
    n = int(ctx["n"])
    s2 = float(np.var(x, ddof=1)) if len(x) > 1 else 0.0 # type: ignore[arg-type]
    bias2 = (float(np.mean(x)) - theta) ** 2
    return float(s2 / max(1, n) + bias2)


def build_default_engine(
    include_dist_free: bool = True,
    include_target_bounds: bool = True,
) -> StatsEngine:
    r"""
    Build a ready-to-use :class:`StatsEngine` with common metrics.

    Parameters
    ----------
    include_dist_free : bool, default True
        Include Chebyshev-based bounds/requirements.
    include_target_bounds : bool, default True
        Include target-aware metrics (bias/MSE/Markov bound).

    Returns
    -------
    StatsEngine
        Configured engine.

    Examples
    --------
    >>> eng = build_default_engine()
    >>> sorted(eng.compute(np.array([0., 1., 2.]), n=3).keys())[:3]
    ['ci_mean', 'kurtosis', 'mean']
    """
    metrics = [
        FnMetric("mean", mean),
        FnMetric("std", std),
        FnMetric("percentiles", percentiles),
        FnMetric("skew", skew),
        FnMetric("kurtosis", kurtosis),
        FnMetric("ci_mean", ci_mean),  # z/t/auto from your utils
    ]
    if include_dist_free:
        metrics += [
            FnMetric("ci_mean_chebyshev", ci_mean_chebyshev),
            FnMetric("chebyshev_required_n", chebyshev_required_n),
        ]
    if include_target_bounds:
        metrics += [
            FnMetric("bias_to_target", bias_to_target),
            FnMetric("mse_to_target", mse_to_target),
            FnMetric("markov_error_prob", markov_error_prob),
        ]
    return StatsEngine(metrics)


DEFAULT_ENGINE = build_default_engine()


__all__ = [
    "StatsEngine",
    "FnMetric",
    "DEFAULT_ENGINE",
    "Metric",
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
]
