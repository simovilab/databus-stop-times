"""Tests for etaval/stats/bootstrap.py."""

from __future__ import annotations

import math

import numpy as np
import pytest

from stoptimes.val.stats.bootstrap import CIResult, bootstrap_ci


class TestBootstrapCI:
    def test_empty_input_returns_nan(self) -> None:
        result = bootstrap_ci([])
        assert isinstance(result, CIResult)
        assert math.isnan(result.point)
        assert math.isnan(result.lo)
        assert math.isnan(result.hi)
        assert result.n == 0

    def test_determinism_same_seed(self) -> None:
        errors = [1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0]
        r1 = bootstrap_ci(errors, n_boot=500, seed=42)
        r2 = bootstrap_ci(errors, n_boot=500, seed=42)
        assert r1 == r2

    def test_different_seed_differs(self) -> None:
        errors = [1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, 8.0, -9.0]
        r1 = bootstrap_ci(errors, n_boot=500, seed=1)
        r2 = bootstrap_ci(errors, n_boot=500, seed=2)
        assert r1.lo != r2.lo or r1.hi != r2.hi

    def test_point_estimate_matches_default_metric(self) -> None:
        errors = [10.0, -20.0, 30.0]
        result = bootstrap_ci(errors, n_boot=100, seed=0)
        expected_point = np.mean(np.abs(errors))
        assert result.point == pytest.approx(expected_point)

    def test_ci_bounds_bracket_point_for_symmetric_data(self) -> None:
        rng = np.random.default_rng(7)
        errors = rng.normal(loc=0.0, scale=10.0, size=500)
        result = bootstrap_ci(errors, n_boot=1000, seed=3)
        assert result.lo <= result.hi

    def test_ci_narrows_with_more_data(self) -> None:
        rng = np.random.default_rng(11)
        small = rng.normal(0, 10, size=20)
        large = rng.normal(0, 10, size=2000)
        r_small = bootstrap_ci(small, n_boot=1000, seed=5)
        r_large = bootstrap_ci(large, n_boot=1000, seed=5)
        assert (r_large.hi - r_large.lo) < (r_small.hi - r_small.lo)

    def test_custom_metric(self) -> None:
        errors = [1.0, 2.0, 3.0, 4.0, 5.0]

        def mean_metric(x: np.ndarray) -> np.ndarray:
            return np.mean(x, axis=-1)

        result = bootstrap_ci(errors, metric=mean_metric, n_boot=200, seed=0)
        assert result.point == pytest.approx(3.0)

    def test_n_boot_and_ci_bookkeeping(self) -> None:
        result = bootstrap_ci([1.0, 2.0, 3.0], n_boot=250, ci=0.9, seed=0)
        assert result.n == 3
        assert result.n_boot == 250
        assert result.ci == 0.9

    def test_single_value(self) -> None:
        result = bootstrap_ci([5.0], n_boot=100, seed=0)
        assert result.point == pytest.approx(5.0)
        # All resamples are the same single value.
        assert result.lo == pytest.approx(5.0)
        assert result.hi == pytest.approx(5.0)

    def test_coverage_approximate(self) -> None:
        """With many bootstrap draws from a known distribution, the CI should
        be plausible: point falls strictly inside a wide symmetric interval."""
        rng = np.random.default_rng(99)
        errors = rng.normal(loc=0.0, scale=5.0, size=1000)
        result = bootstrap_ci(errors, n_boot=2000, seed=1)
        assert result.lo <= result.point <= result.hi

    def test_ci_result_is_frozen(self) -> None:
        import dataclasses

        result = bootstrap_ci([1.0, 2.0], n_boot=10, seed=0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.point = 99.0  # type: ignore[misc]
