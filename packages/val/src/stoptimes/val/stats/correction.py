"""Multiple-comparison p-value correction: Holm-Bonferroni and Benjamini-Hochberg (FDR).

Both return adjusted p-values and reject decisions aligned to the *input order*
of ``pvals``, with adjusted p-values enforced to be monotonically non-decreasing
in sorted-order (the standard step-down/step-up correction guarantee).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class CorrectionResult:
    adjusted_p: npt.NDArray[np.float64]
    reject: npt.NDArray[np.bool_]


def holm(pvals: Sequence[float] | npt.NDArray[np.float64], alpha: float = 0.05) -> CorrectionResult:
    """Holm-Bonferroni step-down correction.

    For the i-th smallest p-value (1-indexed) out of m, the adjustment factor
    is (m - i + 1). Adjusted p-values are enforced non-decreasing when read in
    ascending-p order (running maximum), then clipped to [0, 1].
    """
    p = np.asarray(pvals, dtype=np.float64)
    m = p.shape[0]

    if m == 0:
        return CorrectionResult(
            adjusted_p=np.array([], dtype=np.float64), reject=np.array([], dtype=np.bool_)
        )

    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    factors = m - np.arange(m)  # m, m-1, ..., 1
    adjusted_sorted = sorted_p * factors
    # Enforce monotonicity (running max in ascending-p order).
    adjusted_sorted = np.maximum.accumulate(adjusted_sorted)
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)

    adjusted_p = np.empty(m, dtype=np.float64)
    adjusted_p[order] = adjusted_sorted

    reject = adjusted_p <= alpha
    return CorrectionResult(adjusted_p=adjusted_p, reject=reject)


def benjamini_hochberg(
    pvals: Sequence[float] | npt.NDArray[np.float64], alpha: float = 0.05
) -> CorrectionResult:
    """Benjamini-Hochberg step-up FDR correction.

    For the i-th smallest p-value (1-indexed) out of m: adjusted = p_i * m / i,
    then enforced non-increasing from the largest p-value down (running min),
    which is the standard BH monotonicity guarantee.
    """
    p = np.asarray(pvals, dtype=np.float64)
    m = p.shape[0]

    if m == 0:
        return CorrectionResult(
            adjusted_p=np.array([], dtype=np.float64), reject=np.array([], dtype=np.bool_)
        )

    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    ranks = np.arange(1, m + 1)
    adjusted_sorted = sorted_p * m / ranks

    # Enforce monotonicity: walking from largest p (last) to smallest (first),
    # adjusted p-values must not increase.
    adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)

    adjusted_p = np.empty(m, dtype=np.float64)
    adjusted_p[order] = adjusted_sorted

    reject = adjusted_p <= alpha
    return CorrectionResult(adjusted_p=adjusted_p, reject=reject)
