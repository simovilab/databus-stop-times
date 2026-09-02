"""Calibration analysis: empirical vs nominal (Gaussian-residual) coverage by
prediction horizon band and error tolerance.

calibration_by_horizon(predicted, actual, horizon_s, tolerances_s) -> list[CalibrationBin]
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats as scipy_stats

# Horizon bands (seconds), matching etaval.scoring.metrics bucket conventions.
_BAND_LABELS: list[str] = ["<2m", "2-5m", "5-10m", ">10m"]
_BAND_EDGES: list[tuple[float, float]] = [
    (0.0, 120.0),
    (120.0, 300.0),
    (300.0, 600.0),
    (600.0, math.inf),
]


def _assign_band(horizon: float) -> int:
    """Return the index into _BAND_LABELS/_BAND_EDGES for a given horizon (s)."""
    if horizon < 120.0:
        return 0
    if horizon < 300.0:
        return 1
    if horizon < 600.0:
        return 2
    return 3


@dataclass(frozen=True)
class CalibrationBin:
    band_label: str
    band_lo_s: float
    band_hi_s: float
    tolerance_s: float
    n: int
    nominal_coverage: float
    empirical_coverage: float


def calibration_by_horizon(
    predicted: Sequence[float] | npt.NDArray[np.float64],
    actual: Sequence[float] | npt.NDArray[np.float64],
    horizon_s: Sequence[float] | npt.NDArray[np.float64],
    tolerances_s: Sequence[float] = (30.0, 60.0, 120.0),
) -> list[CalibrationBin]:
    """Bin observations by horizon band and compute per-(band x tolerance) coverage.

    For each band, a Gaussian residual model is fit using the band's residual
    (predicted - actual) standard deviation. ``nominal_coverage`` is the
    probability a zero-mean Gaussian with that std falls within +/- tolerance.
    ``empirical_coverage`` is the observed fraction of |predicted - actual|
    within tolerance for that band.

    Bands with no observations are skipped entirely (no rows emitted).
    """
    pred_arr = np.asarray(predicted, dtype=np.float64)
    act_arr = np.asarray(actual, dtype=np.float64)
    horizon_arr = np.asarray(horizon_s, dtype=np.float64)

    residual = pred_arr - act_arr
    abs_error = np.abs(residual)

    band_idx = np.array([_assign_band(h) for h in horizon_arr], dtype=np.int64)

    bins: list[CalibrationBin] = []
    for b in range(len(_BAND_LABELS)):
        mask = band_idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        band_residual = residual[mask]
        band_abs_error = abs_error[mask]
        std = float(np.std(band_residual, ddof=0)) if n > 1 else 0.0
        lo, hi = _BAND_EDGES[b]

        for tol in tolerances_s:
            empirical_coverage = float(np.mean(band_abs_error <= tol))
            if std <= 0.0:
                nominal_coverage = 1.0 if tol >= 0.0 else 0.0
            else:
                nominal_coverage = float(2.0 * scipy_stats.norm.cdf(tol / std) - 1.0)
            bins.append(
                CalibrationBin(
                    band_label=_BAND_LABELS[b],
                    band_lo_s=lo,
                    band_hi_s=hi,
                    tolerance_s=float(tol),
                    n=n,
                    nominal_coverage=nominal_coverage,
                    empirical_coverage=empirical_coverage,
                )
            )

    return bins
