"""Tests for etaval/stats/correction.py."""

from __future__ import annotations

import numpy as np
import pytest

from etaval.stats.correction import CorrectionResult, benjamini_hochberg, holm


class TestHolm:
    def test_hand_computed_case(self) -> None:
        pvals = [0.01, 0.02, 0.03, 0.04, 0.05]
        result = holm(pvals, alpha=0.05)
        expected = [0.05, 0.08, 0.09, 0.09, 0.09]
        np.testing.assert_allclose(result.adjusted_p, expected, atol=1e-9)
        np.testing.assert_array_equal(result.reject, [True, False, False, False, False])

    def test_single_pvalue(self) -> None:
        result = holm([0.03], alpha=0.05)
        assert result.adjusted_p[0] == pytest.approx(0.03)
        assert result.reject[0]

    def test_empty_input(self) -> None:
        result = holm([], alpha=0.05)
        assert len(result.adjusted_p) == 0
        assert len(result.reject) == 0

    def test_monotonic_in_sorted_order(self) -> None:
        pvals = [0.2, 0.001, 0.15, 0.04, 0.5]
        result = holm(pvals, alpha=0.05)
        order = np.argsort(pvals)
        sorted_adj = np.asarray(result.adjusted_p)[order]
        assert all(sorted_adj[i] <= sorted_adj[i + 1] for i in range(len(sorted_adj) - 1))

    def test_clipped_to_one(self) -> None:
        result = holm([0.9, 0.9, 0.9], alpha=0.05)
        assert all(p <= 1.0 for p in result.adjusted_p)

    def test_reject_aligned_to_input_order(self) -> None:
        pvals = [0.9, 0.001, 0.5]
        result = holm(pvals, alpha=0.05)
        # The smallest input p-value (index 1) should be most likely rejected first.
        assert result.adjusted_p[1] < result.adjusted_p[0]

    def test_more_conservative_than_bh(self) -> None:
        pvals = [0.001, 0.02, 0.03, 0.04, 0.2]
        holm_result = holm(pvals, alpha=0.05)
        bh_result = benjamini_hochberg(pvals, alpha=0.05)
        assert sum(holm_result.reject) <= sum(bh_result.reject)

    def test_returns_correction_result(self) -> None:
        result = holm([0.01, 0.02])
        assert isinstance(result, CorrectionResult)


class TestBenjaminiHochberg:
    def test_hand_computed_case_all_reject(self) -> None:
        pvals = [0.01, 0.02, 0.03, 0.04, 0.05]
        result = benjamini_hochberg(pvals, alpha=0.05)
        np.testing.assert_allclose(result.adjusted_p, [0.05] * 5, atol=1e-9)
        np.testing.assert_array_equal(result.reject, [True] * 5)

    def test_single_pvalue(self) -> None:
        result = benjamini_hochberg([0.03], alpha=0.05)
        assert result.adjusted_p[0] == pytest.approx(0.03)
        assert result.reject[0]

    def test_empty_input(self) -> None:
        result = benjamini_hochberg([], alpha=0.05)
        assert len(result.adjusted_p) == 0
        assert len(result.reject) == 0

    def test_monotonic_by_rank(self) -> None:
        pvals = [0.5, 0.01, 0.3, 0.02, 0.4]
        result = benjamini_hochberg(pvals, alpha=0.05)
        order = np.argsort(pvals)
        sorted_adj = np.asarray(result.adjusted_p)[order]
        assert all(sorted_adj[i] <= sorted_adj[i + 1] for i in range(len(sorted_adj) - 1))

    def test_clipped_to_one(self) -> None:
        result = benjamini_hochberg([0.9, 0.95, 0.99], alpha=0.05)
        assert all(p <= 1.0 for p in result.adjusted_p)

    def test_large_pvalues_no_reject(self) -> None:
        result = benjamini_hochberg([0.8, 0.9, 0.95], alpha=0.05)
        assert not any(result.reject)

    def test_returns_correction_result(self) -> None:
        result = benjamini_hochberg([0.01, 0.02])
        assert isinstance(result, CorrectionResult)

    def test_reject_aligned_to_input_order(self) -> None:
        pvals = [0.9, 0.001, 0.5]
        result = benjamini_hochberg(pvals, alpha=0.05)
        assert result.adjusted_p[1] < result.adjusted_p[0]
