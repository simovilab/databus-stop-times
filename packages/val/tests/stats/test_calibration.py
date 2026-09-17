"""Tests for etaval/stats/calibration.py."""

from __future__ import annotations

import numpy as np
import pytest

from etaval.stats.calibration import CalibrationBin, calibration_by_horizon


class TestCalibrationByHorizon:
    def test_empty_bands_are_skipped(self) -> None:
        # All observations in the <2m band; other bands should not appear.
        predicted = [100.0, 105.0, 95.0]
        actual = [100.0, 100.0, 100.0]
        horizon_s = [60.0, 60.0, 60.0]
        bins = calibration_by_horizon(predicted, actual, horizon_s, tolerances_s=(30.0,))
        labels = {b.band_label for b in bins}
        assert labels == {"<2m"}

    def test_n_and_counts(self) -> None:
        predicted = [100.0, 105.0, 95.0, 100.0]
        actual = [100.0, 100.0, 100.0, 100.0]
        horizon_s = [60.0, 60.0, 60.0, 60.0]
        bins = calibration_by_horizon(predicted, actual, horizon_s, tolerances_s=(30.0,))
        assert len(bins) == 1
        assert bins[0].n == 4

    def test_perfect_predictions_full_coverage(self) -> None:
        """Zero-error predictions => std=0 => nominal_coverage=1, empirical=1."""
        predicted = [100.0, 200.0, 300.0]
        actual = [100.0, 200.0, 300.0]
        horizon_s = [60.0, 60.0, 60.0]
        bins = calibration_by_horizon(predicted, actual, horizon_s, tolerances_s=(0.0, 30.0))
        for b in bins:
            assert b.nominal_coverage == pytest.approx(1.0)
            assert b.empirical_coverage == pytest.approx(1.0)

    def test_multiple_bands(self) -> None:
        predicted = [100.0, 100.0, 100.0, 100.0]
        actual = [100.0, 100.0, 100.0, 100.0]
        horizon_s = [60.0, 200.0, 400.0, 900.0]  # <2m, 2-5m, 5-10m, >10m
        bins = calibration_by_horizon(predicted, actual, horizon_s, tolerances_s=(30.0,))
        labels = {b.band_label for b in bins}
        assert labels == {"<2m", "2-5m", "5-10m", ">10m"}

    def test_tolerances_expand_rows(self) -> None:
        predicted = [110.0, 90.0, 100.0]
        actual = [100.0, 100.0, 100.0]
        horizon_s = [60.0, 60.0, 60.0]
        bins = calibration_by_horizon(predicted, actual, horizon_s, tolerances_s=(5.0, 15.0, 50.0))
        assert len(bins) == 3  # one band x 3 tolerances
        tols = sorted(b.tolerance_s for b in bins)
        assert tols == [5.0, 15.0, 50.0]

    def test_empirical_coverage_matches_manual_count(self) -> None:
        predicted = [100.0, 140.0, 60.0, 100.0]  # errors: 0, 40, -40, 0
        actual = [100.0, 100.0, 100.0, 100.0]
        horizon_s = [60.0, 60.0, 60.0, 60.0]
        bins = calibration_by_horizon(predicted, actual, horizon_s, tolerances_s=(30.0,))
        assert bins[0].empirical_coverage == pytest.approx(2 / 4)

    def test_nominal_coverage_matches_gaussian_formula(self) -> None:
        from scipy import stats as scipy_stats

        rng = np.random.default_rng(0)
        residuals = rng.normal(0, 20.0, size=500)
        predicted = (100.0 + residuals).tolist()
        actual = [100.0] * 500
        horizon_s = [60.0] * 500
        bins = calibration_by_horizon(predicted, actual, horizon_s, tolerances_s=(20.0,))
        std = float(np.std(residuals, ddof=0))
        expected_nominal = 2.0 * scipy_stats.norm.cdf(20.0 / std) - 1.0
        assert bins[0].nominal_coverage == pytest.approx(expected_nominal, rel=1e-6)

    def test_default_tolerances(self) -> None:
        predicted = [100.0, 130.0, 220.0]
        actual = [100.0, 100.0, 100.0]
        horizon_s = [60.0, 60.0, 60.0]
        bins = calibration_by_horizon(predicted, actual, horizon_s)
        tols = sorted({b.tolerance_s for b in bins})
        assert tols == [30.0, 60.0, 120.0]

    def test_returns_calibration_bin_instances(self) -> None:
        bins = calibration_by_horizon([100.0], [100.0], [60.0], tolerances_s=(30.0,))
        assert all(isinstance(b, CalibrationBin) for b in bins)

    def test_band_bounds_present(self) -> None:
        bins = calibration_by_horizon([100.0], [100.0], [60.0], tolerances_s=(30.0,))
        assert bins[0].band_lo_s == 0.0
        assert bins[0].band_hi_s == 120.0
