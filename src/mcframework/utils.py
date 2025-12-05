"""
mcframework.utils
=====================
Utility functions for critical values and CI selection.

This module provides z/t critical values and a tiny helper, :func:`autocrit`,
that chooses between normal and t criticals in a reproducible way.
"""

from __future__ import annotations

from scipy.stats import norm, t

__all__ = ["z_crit", "t_crit", "autocrit"]


def _validate_confidence(confidence: float) -> None:
    r"""
    Validate the confidence level.

    Parameters
    ----------
    confidence : float
        Confidence level to validate.

    Raises
    ------
    ValueError
        If ``confidence`` is not in the open interval ``(0, 1)``.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in the open interval (0, 1)")


def z_crit(confidence: float) -> float:
    r"""
    Two-sided normal critical value :math:`z_{\alpha/2}`.

    For a given confidence level :math:`1-\alpha`, this returns the
    upper :math:`1-\alpha/2` quantile of the standard normal,
    i.e. :math:`z_{\alpha/2} = \Phi^{-1}(1-\alpha/2)`.

    Parameters
    ----------
    confidence : float
        Confidence level in ``(0, 1)`` (e.g. ``0.95``).

    Returns
    -------
    float
        :math:`z_{\alpha/2}`.

    Raises
    ------
    ValueError
        If ``confidence`` is not in ``(0, 1)``.

    Examples
    --------
    >>> round(z_crit(0.95), 2)
    1.96
    """
    _validate_confidence(confidence)
    return float(norm.ppf(1.0 - (1.0 - confidence) / 2.0))


def t_crit(confidence: float, df: int) -> float:
    r"""
    Two-sided Student t critical value :math:`t_{\alpha/2,\;\mathrm{df}}`.

    Parameters
    ----------
    confidence : float
        Confidence level in ``(0, 1)``.
    df : int
        Degrees of freedom (``\mathrm{df} \ge 1``), typically ``n-1``.

    Returns
    -------
    float
        :math:`t_{\alpha/2,\;\mathrm{df}}\ `- the upper :math:`1-\alpha/2`
        quantile of :math:`t_{\mathrm{df}}`.

    Raises
    ------
    ValueError
        If ``confidence`` is not in ``(0, 1)`` or ``df < 1``.

    Examples
    --------
    >>> round(t_crit(0.95, df=9), 3)
    2.262
    """
    _validate_confidence(confidence)
    if df < 1:
        raise ValueError("df must be >= 1")
    return float(t.ppf(1.0 - (1.0 - confidence) / 2.0, df))


def autocrit(confidence: float, n: int, method: str = "auto") -> tuple[float, str]:
    r"""
    Select a critical value (z or t) for two-sided CIs.

    Chooses between :func:`z_crit` and :func:`t_crit` based on the requested
    ``method`` and the sample size ``n``:

    * ``method="z"`` – always use normal criticals.
    * ``method="t"`` – always use Student t with ``\mathrm{df} = \max(1, n-1)``.
    * ``method="auto"`` – use z if ``n \ge 30``, else t.

    Parameters
    ----------
    confidence : float
        Confidence level in ``(0, 1)``.
    n : int
        Sample size used to choose the rule-of-thumb cutoff for ``"auto"``.
    method : {"auto", "z", "t"}, default "auto"
        Selection policy.

    Returns
    -------
    (float, str)
        Pair ``(crit, kind)`` where ``crit`` is the critical value and
        ``kind`` is the string ``"z"`` or ``"t"`` indicating the choice.

    Raises
    ------
    ValueError
        If ``confidence`` is invalid or ``method`` is not one of
        ``{"auto","z","t"}``.

    Notes
    -----
    The returned critical is intended for two-sided intervals of the form

    .. math::
       \bar X \pm c \,\frac{s}{\sqrt{n}},

    where :math:`c` is either :math:`z_{\alpha/2}` or
    :math:`t_{\alpha/2,\;\mathrm{df}}` with
    :math:`\mathrm{df}=\max(1, n-1)`.

    See Also
    --------
    mcframework.stats_engine.ci_mean : Uses this selector to build mean CIs.

    Examples
    --------
    >>> c, kind = autocrit(0.95, n=20, method="auto")
    >>> kind
    't'
    >>> round(c, 2)
    2.09
    >>> c, kind = autocrit(0.95, n=50, method="auto")
    >>> kind
    'z'
    >>> round(c, 2)
    1.96
    """
    _validate_confidence(confidence)
    if method not in ("auto", "z", "t"):
        raise ValueError("method must be 'auto', 'z', or 't'")
    if method == "z" or (method == "auto" and n >= 30):
        return z_crit(confidence), "z"
    return t_crit(confidence, max(1, n - 1)), "t"
