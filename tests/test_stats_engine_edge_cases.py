import numpy as np
import pytest

from mcframework.stats_engine import (
    FnMetric,
    StatsContext,
    StatsEngine,
    _clean,
    _ensure_ctx,
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


def test_stats_context_missing_required_field():
    with pytest.raises(TypeError):
        StatsContext()   # missing required argument n


def test_stats_engine_available_and_select_branch():
    metrics = [
        FnMetric("mean", mean),
        FnMetric("std", std),
        FnMetric("noop", lambda x, ctx: 0)
    ]
    engine = StatsEngine(metrics)

    rep = repr(engine)
    assert "mean" in rep
    assert "std" in rep
    assert "noop" in rep

    res = engine.compute(np.array([1.0, 2.0, 3.0]), select=("std",), n=3, confidence=0.95)
    assert set(res.metrics) == {"std"}



def test_stats_engine_skips_empty_and_error_metrics():
    def none_metric(x, ctx):
        return None

    metrics = [
        FnMetric("mean", mean),
        FnMetric("none_metric", none_metric),
        FnMetric("chebyshev_required_n", chebyshev_required_n),
        FnMetric("boom", lambda x, ctx: (_ for _ in ()).throw(RuntimeError("boom"))),
    ]
    engine = StatsEngine(metrics)

    ctx = StatsContext(n=5)  # Missing eps for chebyshev_required_n
    result = engine.compute(np.array([1.0, 2.0, 3.0]), ctx)

    # Only mean should survive; 'none_metric' returns None and is skipped,
    # chebyshev raises MissingContextError and is skipped,
    # and boom raises RuntimeError and is tracked in errors.
    assert result.metrics == {"mean": pytest.approx(2.0)}
    assert len(result.skipped) == 2  # none_metric and chebyshev_required_n
    assert len(result.errors) == 1  # boom
    assert any("none_metric" in s[0] for s in result.skipped)
    assert any("chebyshev_required_n" in s[0] for s in result.skipped)
    assert "boom" in result.errors[0][0]


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


def test_ci_mean_outputs_plain_python_floats():
    data = np.linspace(0.0, 5.0, num=8)
    ctx = StatsContext(n=data.size, confidence=0.9)
    res = ci_mean(data, ctx)
    for key in ("low", "high", "se", "crit"):
        assert isinstance(res[key], float)


def test_ci_mean_bootstrap_percentile_and_bca():
    arr = np.linspace(0.0, 1.0, num=6)

    ctx_percentile = StatsContext(n=arr.size, n_bootstrap=200, rng=123, bootstrap="percentile")
    perc_res = ci_mean_bootstrap(arr, ctx_percentile)
    assert perc_res["method"] == "bootstrap-percentile"

    ctx_bca = StatsContext(n=arr.size, n_bootstrap=200, rng=321, bootstrap="bca")
    bca_res = ci_mean_bootstrap(arr, ctx_bca)
    assert bca_res["method"] == "bootstrap-bca"
    assert bca_res["low"] <= bca_res["high"]


def test_ci_mean_bootstrap_outputs_plain_python_floats():
    arr = np.linspace(-2.0, 3.0, num=7)
    ctx = StatsContext(n=arr.size, n_bootstrap=100, rng=999, bootstrap="percentile")
    res = ci_mean_bootstrap(arr, ctx)
    assert isinstance(res["low"], float)
    assert isinstance(res["high"], float)


def test_ci_mean_bootstrap_empty_returns_none():
    ctx = StatsContext(n=0, n_bootstrap=100, rng=0)
    assert ci_mean_bootstrap(np.array([]), ctx) is None


def test_ci_mean_chebyshev_small_sample_returns_none():
    ctx_small = StatsContext(n=1)
    assert ci_mean_chebyshev(np.array([1.0]), ctx_small) is None

    ctx = StatsContext(n=10, confidence=0.9)
    res = ci_mean_chebyshev(np.array([1.0, 2.0, 3.0, 4.0]), ctx)
    assert res["method"] == "chebyshev"


def test_ci_mean_chebyshev_outputs_plain_python_floats():
    ctx = StatsContext(n=6, confidence=0.9)
    values = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    res = ci_mean_chebyshev(values, ctx)
    assert isinstance(res["low"], float)
    assert isinstance(res["high"], float)


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


def test_stats_context_cross_field_validations():
    """Test new cross-field validation in StatsContext.__post_init__"""

    # Test ess > n raises error
    with pytest.raises(ValueError, match="ess .* cannot exceed n"):
        StatsContext(n=10, ess=20)

    # Test ess <= 0 raises error
    with pytest.raises(ValueError, match="ess must be positive"):
        StatsContext(n=10, ess=0)

    # Test n_bootstrap < 100 raises error
    with pytest.raises(ValueError, match="n_bootstrap .* should be >= 100"):
        StatsContext(n=10, n_bootstrap=50)

    # Valid cases should not raise
    ctx = StatsContext(n=100, ess=50, n_bootstrap=1000)
    assert ctx.ess == 50
    assert ctx.n_bootstrap == 1000


def test_compute_result_tracking():
    """Test that ComputeResult properly tracks skipped and errored metrics"""
    from mcframework.stats_engine import ComputeResult, MissingContextError

    def failing_metric(x, ctx):
        raise RuntimeError("intentional failure")

    def none_metric(x, ctx):
        return None

    def missing_ctx_metric(x, ctx):
        raise MissingContextError("missing field")

    metrics = [
        FnMetric("mean", mean),
        FnMetric("failing", failing_metric),
        FnMetric("none", none_metric),
        FnMetric("missing", missing_ctx_metric),
    ]
    engine = StatsEngine(metrics)
    result = engine.compute(np.array([1, 2, 3]), n=3)

    # Check result type
    assert isinstance(result, ComputeResult)

    # mean should succeed
    assert "mean" in result.metrics
    assert result.metrics["mean"] == pytest.approx(2.0)

    # failing should be in errors
    assert len(result.errors) == 1
    assert result.errors[0][0] == "failing"
    assert "intentional failure" in result.errors[0][1]

    # none and missing should be in skipped
    assert len(result.skipped) == 2
    skipped_names = [s[0] for s in result.skipped]
    assert "none" in skipped_names
    assert "missing" in skipped_names

    # Test successful_metrics() method
    assert result.successful_metrics() == {"mean"}


def test_compute_result_repr():
    """Test that ComputeResult's __repr__ displays correctly"""
    from mcframework.stats_engine import ComputeResult

    result = ComputeResult(
        metrics={"mean": 5.0, "std": 1.2},
        skipped=[("metric1", "reason1"), ("metric2", "reason2")],
        errors=[("metric3", "error message")]
    )
    
    # Test that repr generates the multiline formatted output
    repr_str = repr(result)
    assert "ComputeResult" in repr_str
    assert "mean" in repr_str
    assert "std" in repr_str
    assert "metric1" in repr_str
    assert "metric2" in repr_str
    assert "metric3" in repr_str


def test_ensure_ctx_with_none():
    """Test that _ensure_ctx handles None ctx by creating a default context"""
    arr = np.array([1.0, 2.0, 3.0, 4.0])
    ctx = _ensure_ctx(None, arr)
    assert isinstance(ctx, StatsContext)
    assert ctx.n == arr.size


def test_value_error_handling_for_missing_context_keys():
    """Test that ValueError with 'Missing required context keys' is caught"""

    def metric_raising_value_error(x, ctx):
        raise ValueError("Missing required context keys: some_key")

    metrics = [
        FnMetric("mean", mean),
        FnMetric("value_error_metric", metric_raising_value_error),
    ]
    engine = StatsEngine(metrics)
    result = engine.compute(np.array([1, 2, 3]), n=3)

    # mean should succeed
    assert "mean" in result.metrics

    # value_error_metric should be skipped
    assert len(result.skipped) == 1
    assert result.skipped[0][0] == "value_error_metric"
    assert "Missing required context keys" in result.skipped[0][1]


def test_value_error_without_missing_keys_is_raised():
    """Test that ValueError without 'Missing required context keys' is re-raised"""
    
    def metric_raising_other_value_error(x, ctx):
        raise ValueError("Some other error message")

    metrics = [
        FnMetric("value_error_metric", metric_raising_other_value_error),
    ]
    engine = StatsEngine(metrics)
    
    # This should raise the ValueError since it doesn't contain "Missing required context keys"
    with pytest.raises(ValueError, match="Some other error message"):
        engine.compute(np.array([1, 2, 3]), n=3)


def test_ci_mean_chebyshev_with_none_mean_or_std():
    """Test that ci_mean_chebyshev returns None when mean or std is None"""
    from unittest.mock import patch
    
    # Create data with enough elements to pass the n_eff >= 2 check
    # but mock std to return None
    data = np.array([1.0, 2.0, 3.0, 4.0])
    ctx = StatsContext(n=4, confidence=0.95)
    
    # Mock std to return None while keeping mean working
    with patch('mcframework.stats_engine.std', return_value=None):
        result = ci_mean_chebyshev(data, ctx)
        assert result is None

