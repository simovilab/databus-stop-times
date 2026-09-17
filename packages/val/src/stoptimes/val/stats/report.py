"""CI-gated leaderboard entry point.

compute_metrics_with_ci(results, n_boot=1000, ci=0.95, seed=0, tz="UTC") -> ReportWithCI

Reuses :mod:`stoptimes.val.scoring.metrics` for the base per-predictor numbers, and
adds a bootstrap confidence interval around each predictor's MAE (mean
absolute error), computed deterministically from the given seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from stoptimes.core.domain.models import ValidationResult
from stoptimes.val.scoring.metrics import MetricsReport, compute_metrics
from stoptimes.val.stats.bootstrap import CIResult, bootstrap_ci


@dataclass(frozen=True)
class PredictorWithCI:
    predictor_id: str
    mae: float
    mae_ci: CIResult
    count: int


@dataclass(frozen=True)
class ReportWithCI:
    base: MetricsReport
    per_predictor_ci: list[PredictorWithCI] = field(default_factory=list)


def compute_metrics_with_ci(
    results: list[ValidationResult],
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
    tz: str = "UTC",
) -> ReportWithCI:
    """Compute the base MetricsReport plus a bootstrap CI on each predictor's MAE.

    Args:
        results: ValidationResults collected from the engine (or loaded from storage).
        n_boot:  Number of bootstrap resamples per predictor.
        ci:      Confidence level (e.g. 0.95).
        seed:    Base seed for determinism; each predictor gets a derived seed so
            resampling differs across predictors while remaining fully
            reproducible given (seed, predictor order).
        tz:      Reserved for future horizon-aware CI slicing; passed through
            for API symmetry with compute_metrics(..., tz=...).

    Returns:
        ReportWithCI containing the base MetricsReport and a per-predictor list
        of MAE point estimates with bootstrap confidence intervals.
    """
    base = compute_metrics(results)

    if not results:
        return ReportWithCI(base=base, per_predictor_ci=[])

    errors_by_predictor: dict[str, list[float]] = {}
    for r in results:
        errors_by_predictor.setdefault(r.predictor_id, []).append(abs(r.error_s))

    per_predictor_ci: list[PredictorWithCI] = []
    for i, pid in enumerate(sorted(errors_by_predictor)):
        abs_errors = errors_by_predictor[pid]
        derived_seed = seed + i
        ci_result = bootstrap_ci(abs_errors, n_boot=n_boot, ci=ci, seed=derived_seed)
        per_predictor_ci.append(
            PredictorWithCI(
                predictor_id=pid,
                mae=ci_result.point,
                mae_ci=ci_result,
                count=len(abs_errors),
            )
        )

    return ReportWithCI(base=base, per_predictor_ci=per_predictor_ci)
