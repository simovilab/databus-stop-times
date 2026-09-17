"""Tests for etaval/stats/significance.py."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats as scipy_stats

from etaval.stats.significance import SignificanceResult, diebold_mariano, wilcoxon_test


class TestWilcoxonTest:
    def test_matches_scipy_directly(self) -> None:
        diffs = [1.0, -2.0, 3.0, 4.0, -1.0, 2.0, 5.0, -3.0]
        result = wilcoxon_test(diffs)
        expected = scipy_stats.wilcoxon(np.asarray(diffs), mode="auto")
        assert result.statistic == pytest.approx(float(expected.statistic))
        assert result.p_value == pytest.approx(float(expected.pvalue))
        assert result.n == 8

    def test_all_zero_diffs_degenerate(self) -> None:
        result = wilcoxon_test([0.0, 0.0, 0.0])
        assert result.statistic == 0.0
        assert result.p_value == 1.0
        assert result.n == 3

    def test_empty_input(self) -> None:
        result = wilcoxon_test([])
        assert math.isnan(result.statistic)
        assert math.isnan(result.p_value)
        assert result.n == 0

    def test_all_positive_diffs_low_p_value(self) -> None:
        """A strongly one-sided sample should have a small p-value."""
        diffs = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        result = wilcoxon_test(diffs)
        assert result.p_value < 0.05

    def test_returns_significance_result(self) -> None:
        result = wilcoxon_test([1.0, -1.0, 2.0])
        assert isinstance(result, SignificanceResult)

    def test_frozen(self) -> None:
        import dataclasses

        result = wilcoxon_test([1.0, -2.0, 3.0])
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.n = 999  # type: ignore[misc]


class TestDieboldMariano:
    def test_zero_mean_zero_variance_degenerate(self) -> None:
        result = diebold_mariano([0.0, 0.0, 0.0, 0.0])
        assert result.statistic == 0.0
        assert result.p_value == 1.0

    def test_n_less_than_2_returns_nan(self) -> None:
        result = diebold_mariano([1.0])
        assert math.isnan(result.statistic)
        assert math.isnan(result.p_value)
        result0 = diebold_mariano([])
        assert math.isnan(result0.statistic)

    def test_positive_diffs_positive_statistic(self) -> None:
        """diffs = loss_a - loss_b; consistently positive => A worse => positive stat."""
        rng = np.random.default_rng(0)
        diffs = rng.normal(loc=5.0, scale=1.0, size=200)
        result = diebold_mariano(diffs, hac_lags=0)
        assert result.statistic > 0

    def test_negative_diffs_negative_statistic(self) -> None:
        rng = np.random.default_rng(0)
        diffs = rng.normal(loc=-5.0, scale=1.0, size=200)
        result = diebold_mariano(diffs, hac_lags=0)
        assert result.statistic < 0

    def test_strong_signal_low_p_value(self) -> None:
        rng = np.random.default_rng(1)
        diffs = rng.normal(loc=10.0, scale=1.0, size=200)
        result = diebold_mariano(diffs, hac_lags=0)
        assert result.p_value < 0.01

    def test_no_signal_high_p_value(self) -> None:
        rng = np.random.default_rng(2)
        diffs = rng.normal(loc=0.0, scale=1.0, size=500)
        result = diebold_mariano(diffs, hac_lags=0)
        assert result.p_value > 0.05

    def test_hac_lags_changes_variance_estimate(self) -> None:
        """Autocorrelated series: HAC lags should change the statistic vs lag=0."""
        rng = np.random.default_rng(3)
        n = 300
        base = rng.normal(0, 1, size=n)
        # Introduce strong positive autocorrelation via a moving average.
        diffs = np.convolve(base, np.ones(5) / 5, mode="same") + 2.0
        stat_no_hac = diebold_mariano(diffs, hac_lags=0).statistic
        stat_hac = diebold_mariano(diffs, hac_lags=5).statistic
        assert stat_no_hac != pytest.approx(stat_hac)

    def test_sign_matches_mean_sign(self) -> None:
        diffs = [1.0, 2.0, -0.5, 3.0, 1.5, 2.5, -1.0, 4.0]
        result = diebold_mariano(diffs, hac_lags=1)
        assert (result.statistic > 0) == (np.mean(diffs) > 0)

    def test_returns_significance_result(self) -> None:
        result = diebold_mariano([1.0, 2.0, 3.0, -1.0])
        assert isinstance(result, SignificanceResult)

    def test_n_recorded(self) -> None:
        diffs = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = diebold_mariano(diffs)
        assert result.n == 5
