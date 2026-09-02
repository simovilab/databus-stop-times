"""Statistics layer (brief A2/A3): bootstrap CIs, significance tests, multiple-
comparison correction, calibration analysis, and a CI-gated report entry point.

Modules:
- bootstrap:    bootstrap_ci — percentile bootstrap confidence intervals.
- significance: wilcoxon_test, diebold_mariano — paired-difference significance tests.
- correction:   holm, benjamini_hochberg — multiple-comparison p-value correction.
- calibration:  calibration_by_horizon — empirical vs nominal coverage by horizon band.
- report:       compute_metrics_with_ci — CI-gated leaderboard entry point.
"""

from __future__ import annotations

from etaval.stats.bootstrap import CIResult, bootstrap_ci
from etaval.stats.calibration import CalibrationBin, calibration_by_horizon
from etaval.stats.correction import CorrectionResult, benjamini_hochberg, holm
from etaval.stats.report import PredictorWithCI, ReportWithCI, compute_metrics_with_ci
from etaval.stats.significance import (
    SignificanceResult,
    diebold_mariano,
    wilcoxon_test,
)

__all__ = [
    "CIResult",
    "bootstrap_ci",
    "SignificanceResult",
    "wilcoxon_test",
    "diebold_mariano",
    "CorrectionResult",
    "holm",
    "benjamini_hochberg",
    "CalibrationBin",
    "calibration_by_horizon",
    "PredictorWithCI",
    "ReportWithCI",
    "compute_metrics_with_ci",
]
