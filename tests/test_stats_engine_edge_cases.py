import numpy as np
import pytest

from mcframework.stats_engine import (
    FnMetric,
    StatsContext,
    StatsEngine,
    _clean,
    _ensure_ctx,
    _validate_ctx,
    chebyshev_required_n,
    ci_mean,
    ci_mean_bootstrap,
    ci_mean_chebyshev,
    markov_error_prob,
    mean,
    percentiles,
    std,
)


def test_stats_context_overrides_and_eff_n():
    base = StatsContext(n=20, nan_policy="omit")
    ctx = base.with_overrides(confidence=0.90, ess=5)
    assert ctx.confidence == 0.90
    assert ctx.eff_n(observed_len=100, finite_count=42) == 5

    # When ess is cleared, the finite count should be used
    ctx2 = ctx.with_overrides(ess=None)
    assert ctx2.eff_n(observed_len=100, finite_count=7) == 7

    # Fall back to declared n when nan_policy != "omit"
    ctx3 = ctx2.with_overrides(nan_policy="propagate")
    assert ctx3.eff_n(observed_len=11, finite_count=3) == ctx3.n


def test_stats_context_get_generators_variants():
    seeded = np.random.default_rng(123)
    ctx_seeded = StatsContext(n=1, rng=seeded)
    assert ctx_seeded.get_generators() is seeded

    ctx_from_int = StatsContext(n=1, rng=999)
    g_from_int = ctx_from_int.get_generators()
    assert isinstance(g_from_int, np.random.Generator)

    ctx_default = StatsContext(n=1)
    g_default = ctx_default.get_generators()
    assert isinstance(g_default, np.random.Generator)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"confidence": 1.2}, "confidence"),
        ({"percentiles": (-5, 50)}, "percentiles"),
        ({"n_bootstrap": 0}, "n_bootstrap"),
        ({"ddof": -1}, "ddof"),
        ({"eps": 0}, "eps must be positive"),
    ],
)
def test_stats_context_validation_errors(kwargs, message):
    with pytest.raises(ValueError, match=message):
        StatsContext(n=1, **kwargs)


def test_validate_ctx_missing_required_keys():
    with pytest.raises(ValueError) as exc:
        _validate_ctx({}, {"n"}, {"confidence"})
    assert "Missing required context keys" in str(exc.value)


def test_stats_engine_available_and_select_branch():
    metrics = [FnMetric("mean", mean), FnMetric("std", std), FnMetric("noop", lambda x, ctx: 0)]
    engine = StatsEngine(metrics)
    assert engine.available() == ("mean", "std", "noop")

    res = engine.compute(np.array([1.0, 2.0, 3.0]), select=("std",), n=3, confidence=0.95)
    assert set(res) == {"std"}


def test_stats_engine_skips_empty_and_error_metrics():
    def empty_metric(x, ctx):
        return {}

    metrics = [
        FnMetric("mean", mean),
        FnMetric("empty", empty_metric),
        FnMetric("chebyshev_required_n", chebyshev_required_n),
        FnMetric("boom", lambda x, ctx: (_ for _ in ()).throw(RuntimeError("boom"))),
    ]
    engine = StatsEngine(metrics)

    ctx = StatsContext(n=5)  # Missing eps for chebyshev_required_n
    result = engine.compute(np.array([1.0, 2.0, 3.0]), ctx)

    # Only mean should survive; 'empty' skipped, chebyshev raises ValueError and is skipped,
    # and boom raises RuntimeError and is suppressed.
    assert result == {"mean": pytest.approx(2.0)}


def test_ensure_ctx_handles_dict_and_attributes():
    arr = np.array([1.0, 2.0])
    ctx_from_dict = _ensure_ctx({"confidence": 0.9}, arr)
    assert isinstance(ctx_from_dict, StatsContext)
    assert ctx_from_dict.n == arr.size
    assert ctx_from_dict.confidence == 0.9

    class AttrCtx:
        def __init__(self):
            self.n = 7
            self.confidence = 0.8

    ctx_from_attrs = _ensure_ctx(AttrCtx(), arr)
    assert isinstance(ctx_from_attrs, StatsContext)
    assert ctx_from_attrs.n == 7


def test_ensure_ctx_rejects_invalid_object():
    with pytest.raises(TypeError):
        _ensure_ctx(42, np.array([1.0, 2.0]))


def test_clean_respects_nan_policy_and_validates():
    ctx = StatsContext(n=2, nan_policy="omit")
    arr, mask = _clean(np.array([1.0, np.nan, 3.0]), ctx)
    assert arr.size == 2
    assert mask.shape == (3,)

    bad_ctx = ctx.with_overrides(nan_policy="unexpected")
    with pytest.raises(ValueError):
        _clean(np.array([1.0, 2.0]), bad_ctx)


def test_percentiles_with_empty_input_returns_nan():
    ctx = StatsContext(n=0, percentiles=(10, 90), nan_policy="omit")
    res = percentiles(np.array([]), ctx)
    assert set(res.keys()) == {10, 90}
    assert all(np.isnan(v) for v in res.values())


def test_ci_mean_handles_edge_cases():
    ctx_empty = StatsContext(n=0)
    empty_res = ci_mean(np.array([]), ctx_empty)
    assert np.isnan(empty_res["low"])
    assert empty_res["method"] == ctx_empty.ci_method

    ctx_small = StatsContext(n=1)
    small_res = ci_mean(np.array([1.0]), ctx_small)
    assert np.isnan(small_res["low"])

    ctx_zero = StatsContext(n=4)
    data = np.full(4, 2.0)
    zero_res = ci_mean(data, ctx_zero)
    assert zero_res["low"] == pytest.approx(2.0)
    assert zero_res["high"] == pytest.approx(2.0)
    assert zero_res["crit"] >= 0


def test_ci_mean_bootstrap_percentile_and_bca():
    arr = np.linspace(0.0, 1.0, num=6)

    ctx_percentile = StatsContext(n=arr.size, n_bootstrap=200, rng=123, bootstrap="percentile")
    perc_res = ci_mean_bootstrap(arr, ctx_percentile)
    assert perc_res["method"] == "bootstrap-percentile"

    ctx_bca = StatsContext(n=arr.size, n_bootstrap=200, rng=321, bootstrap="bca")
    bca_res = ci_mean_bootstrap(arr, ctx_bca)
    assert bca_res["method"] == "bootstrap-bca"
    assert bca_res["low"] <= bca_res["high"]


def test_ci_mean_bootstrap_empty_returns_empty_dict():
    ctx = StatsContext(n=0, n_bootstrap=50, rng=0)
    assert ci_mean_bootstrap(np.array([]), ctx) == {}


def test_ci_mean_chebyshev_small_sample_returns_empty():
    ctx_small = StatsContext(n=1)
    assert ci_mean_chebyshev(np.array([1.0]), ctx_small) == {}

    ctx = StatsContext(n=10, confidence=0.9)
    res = ci_mean_chebyshev(np.array([1.0, 2.0, 3.0, 4.0]), ctx)
    assert res["method"] == "chebyshev"


def test_chebyshev_required_n_validations_and_result():
    ctx_valid = StatsContext(n=5, eps=0.5)
    res = chebyshev_required_n(np.array([1.0, 2.0, 3.0]), ctx_valid)
    assert res > 0

    with pytest.raises(ValueError):
        chebyshev_required_n(np.array([1.0, 2.0, 3.0]), StatsContext(n=3))

    ctx_invalid = ctx_valid.with_overrides(eps=0.5)
    object.__setattr__(ctx_invalid, "eps", 0.0)
    with pytest.raises(ValueError, match="ctx.eps must be positive"):
        chebyshev_required_n(np.array([1.0, 2.0, 3.0]), ctx_invalid)


def test_markov_error_prob_validations_and_result():
    arr = np.array([1.0, 2.0, 3.0])
    ctx_valid = StatsContext(n=3, target=2.0, eps=0.5)
    value = markov_error_prob(arr, ctx_valid)
    assert value >= 0.0

    with pytest.raises(ValueError, match="requires ctx.target"):
        markov_error_prob(arr, StatsContext(n=3, eps=0.5))

    with pytest.raises(ValueError, match="requires ctx.eps"):
        markov_error_prob(arr, StatsContext(n=3, target=2.0))

    ctx_negative_eps = ctx_valid.with_overrides()
    object.__setattr__(ctx_negative_eps, "eps", -0.1)
    with pytest.raises(ValueError, match="must be positive"):
        markov_error_prob(arr, ctx_negative_eps)

