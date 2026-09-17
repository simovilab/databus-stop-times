"""Paired-difference significance tests.

wilcoxon_test(diffs)             -> SignificanceResult  (scipy.stats.wilcoxon wrapper)
diebold_mariano(diffs, hac_lags) -> SignificanceResult  (Newey-West HAC variance)

``diffs`` in both cases are per-observation loss differentials, i.e.
``loss_a - loss_b`` for two predictors evaluated on the same observations
(e.g. |error_a| - |error_b|). A negative mean favors predictor A.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats as scipy_stats


@dataclass(frozen=True)
class SignificanceResult:
    statistic: float
    p_value: float
    n: int


def wilcoxon_test(diffs: Sequence[float] | npt.NDArray[np.float64]) -> SignificanceResult:
    """Wilcoxon signed-rank test on per-observation differentials.

    Uses scipy's exact method where scipy allows it (small, tie-free samples);
    scipy falls back to its normal approximation automatically otherwise.
    Degenerate cases (n < 1, or all-zero differences) are guarded explicitly
    because scipy.stats.wilcoxon raises ValueError on an all-zero input.
    """
    arr = np.asarray(diffs, dtype=np.float64)
    n = int(arr.shape[0])

    if n < 1:
        return SignificanceResult(statistic=math.nan, p_value=math.nan, n=n)

    nonzero = arr[arr != 0.0]
    if nonzero.shape[0] == 0:
        # No information: every paired difference is exactly zero.
        return SignificanceResult(statistic=0.0, p_value=1.0, n=n)

    try:
        result = scipy_stats.wilcoxon(arr, mode="auto")
    except ValueError:
        # scipy can still refuse (e.g. n < ~10 with zero-splitting edge cases).
        return SignificanceResult(statistic=0.0, p_value=1.0, n=n)

    return SignificanceResult(statistic=float(result.statistic), p_value=float(result.pvalue), n=n)


def _newey_west_long_run_variance(d: npt.NDArray[np.float64], hac_lags: int) -> float:
    """Newey-West (Bartlett-kernel) long-run variance estimate of the mean of ``d``."""
    n = d.shape[0]
    dbar = d.mean()
    centered = d - dbar
    gamma0 = float(np.dot(centered, centered) / n)
    long_run = gamma0
    for lag in range(1, hac_lags + 1):
        if lag >= n:
            break
        cov = float(np.dot(centered[lag:], centered[:-lag]) / n)
        weight = 1.0 - lag / (hac_lags + 1)
        long_run += 2.0 * weight * cov
    return long_run


def diebold_mariano(
    diffs: Sequence[float] | npt.NDArray[np.float64],
    hac_lags: int = 0,
) -> SignificanceResult:
    """Diebold-Mariano test on per-observation loss differentials.

    ``diffs`` = loss_a - loss_b (e.g. |error_a| - |error_b|) per observation.
    A negative DM statistic indicates predictor A has lower average loss.

    The variance of the mean differential is estimated with a Newey-West (HAC)
    estimator using ``hac_lags`` lags (0 = plain sample variance / n, no
    autocovariance correction). The p-value is two-sided from the asymptotic
    normal distribution of the DM statistic.
    """
    arr = np.asarray(diffs, dtype=np.float64)
    n = int(arr.shape[0])

    if n < 2:
        return SignificanceResult(statistic=math.nan, p_value=math.nan, n=n)

    mean_d = float(arr.mean())
    long_run_var = _newey_west_long_run_variance(arr, hac_lags)

    if long_run_var <= 0.0:
        # No variability in the differential series (e.g. all diffs equal).
        if mean_d == 0.0:
            return SignificanceResult(statistic=0.0, p_value=1.0, n=n)
        return SignificanceResult(statistic=math.inf if mean_d > 0 else -math.inf, p_value=0.0, n=n)

    dm_stat = mean_d / math.sqrt(long_run_var / n)
    p_value = float(2.0 * (1.0 - scipy_stats.norm.cdf(abs(dm_stat))))

    return SignificanceResult(statistic=float(dm_stat), p_value=p_value, n=n)
