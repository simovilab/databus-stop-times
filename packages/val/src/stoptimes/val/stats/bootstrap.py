"""Percentile bootstrap confidence intervals.

bootstrap_ci(errors, metric=..., n_boot=1000, ci=0.95, seed=0) -> CIResult

Vectorized: builds an (n_boot, n) resample-index matrix with a seeded numpy
Generator, applies ``metric`` per row, and takes the percentile interval.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


def _default_metric(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Default bootstrap metric: mean of absolute values, applied along axis=-1."""
    return np.mean(np.abs(values), axis=-1)


@dataclass(frozen=True)
class CIResult:
    point: float
    lo: float
    hi: float
    n: int
    n_boot: int
    ci: float


def bootstrap_ci(
    errors: Sequence[float] | npt.NDArray[np.float64],
    metric: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]] | None = None,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> CIResult:
    """Compute a percentile bootstrap confidence interval for ``metric(errors)``.

    Args:
        errors: 1-D sequence of per-observation values (e.g. signed errors).
        metric: Vectorized function taking an array along its last axis and
            returning the reduced statistic. Defaults to mean(|x|). Must accept
            an ``axis=-1`` reduction over a 2-D ``(n_boot, n)`` resample matrix.
        n_boot: Number of bootstrap resamples.
        ci: Confidence level (e.g. 0.95 for a 95% CI).
        seed: Seed for the numpy random Generator (determinism).

    Returns:
        CIResult with the point estimate (on the original data), lo/hi bounds
        of the percentile interval, and bookkeeping fields.
    """
    arr = np.asarray(errors, dtype=np.float64)
    n = int(arr.shape[0])
    metric_fn = metric if metric is not None else _default_metric

    if n == 0:
        return CIResult(point=math.nan, lo=math.nan, hi=math.nan, n=0, n_boot=n_boot, ci=ci)

    point = float(metric_fn(arr))

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    resamples = arr[idx]  # (n_boot, n)
    boot_stats = np.asarray(metric_fn(resamples), dtype=np.float64)

    alpha = 1.0 - ci
    lo_pct = 100.0 * (alpha / 2.0)
    hi_pct = 100.0 * (1.0 - alpha / 2.0)
    lo = float(np.percentile(boot_stats, lo_pct))
    hi = float(np.percentile(boot_stats, hi_pct))

    return CIResult(point=point, lo=lo, hi=hi, n=n, n_boot=n_boot, ci=ci)
