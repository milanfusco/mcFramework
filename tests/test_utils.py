import pytest

from mcframework.utils import autocrit, t_crit, z_crit


class TestCriticalValues:
    """[FR-18, FR-19] Test statistical critical value calculations."""

    def test_z_crit_95_confidence(self):
        """[FR-18] Test z critical value for 95% confidence."""
        result = z_crit(0.95)
        assert pytest.approx(result, abs=0.001) == 1.96

    def test_z_crit_99_confidence(self):
        """[FR-18] Test z critical value for 99% confidence."""
        result = z_crit(0.99)
        assert pytest.approx(result, abs=0.001) == 2.576

    def test_z_crit_invalid_confidence_too_low(self):
        """[FR-18, USA-4] Test z_crit raises error for confidence <= 0."""
        with pytest.raises(ValueError, match="confidence must be in the open interval"):
            z_crit(0.0)

    def test_z_crit_invalid_confidence_too_high(self):
        """[FR-18, USA-4] Test z_crit raises error for confidence >= 1."""
        with pytest.raises(ValueError, match="confidence must be in the open interval"):
            z_crit(1.0)

    def test_t_crit_95_confidence_30_df(self):
        """[FR-18] Test t critical value for 95% confidence with 30 df."""
        result = t_crit(0.95, 30)
        assert result > 1.96  # Should be larger than z
        assert result < 2.1

    def test_t_crit_small_df(self):
        """[FR-18] Test t critical value with small degrees of freedom."""
        result = t_crit(0.95, 5)
        assert result > 2.5  # Much larger for small df

    def test_t_crit_invalid_df(self):
        """[FR-18, USA-4] Test t_crit raises error for invalid df."""
        with pytest.raises(ValueError, match="df must be >= 1"):
            t_crit(0.95, 0)

    def test_autocrit_large_n_uses_z(self):
        """[FR-19] Test autocrit uses z for n >= 30."""
        crit, method = autocrit(0.95, 100, "auto")
        assert method == "z"
        assert pytest.approx(crit, abs=0.001) == 1.96

    def test_autocrit_small_n_uses_t(self):
        """[FR-19] Test autocrit uses t for n < 30."""
        crit, method = autocrit(0.95, 10, "auto")
        assert method == "t"
        assert crit > 1.96

    def test_autocrit_force_z(self):
        """[FR-19] Test autocrit with forced z method."""
        crit, method = autocrit(0.95, 10, "z")
        assert method == "z"
        assert pytest.approx(crit, abs=0.001) == 1.96

    def test_autocrit_force_t(self):
        """[FR-19] Test autocrit with forced t method."""
        crit, method = autocrit(0.95, 100, "t")
        assert method == "t"
        assert crit > 1.96

    def test_autocrit_invalid_method(self):
        """[FR-19, USA-4] Test autocrit raises error for invalid method."""
        with pytest.raises(ValueError, match="method must be"):
            autocrit(0.95, 100, "invalid")