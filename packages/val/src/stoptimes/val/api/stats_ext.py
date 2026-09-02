"""Statistical layer computed on top of a DONE run's results.

Implements the pinned ``GET /api/runs/{id}/metrics?stats=1`` contract:

- ``mae_ci`` + ``latency`` per predictor
- top-level ``significance`` (wilcoxon + diebold_mariano, Holm-corrected
  within each test family, one row per unordered predictor pair)
- top-level ``calibration`` (:func:`~etaval.stats.calibration.calibration_by_horizon`
  bins over every result in the run)
- top-level ``seed`` (always ``STATS_SEED`` — deterministic)

Nothing here talks to the network, storage, or FastAPI — it is a pure
function of ``(results, summary)`` so :class:`~etaval.api.runs.RunManager`
can cache its output per-run and compute it exactly once.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict
from itertools import combinations
from typing import Any

from etaval.domain.models import ValidationResult
from etaval.engine.runtime import RunSummary
from etaval.stats.bootstrap import CIResult
from etaval.stats.calibration import calibration_by_horizon
from etaval.stats.correction import holm
from etaval.stats.report import compute_metrics_with_ci
from etaval.stats.significance import SignificanceResult, diebold_mariano, wilcoxon_test

# Deterministic seed for every stats computation — pinned by the API contract.
STATS_SEED: int = 0

_DM_HAC_LAGS: int = 1
_TEST_FAMILIES: list[tuple[str, Callable[[list[float]], SignificanceResult]]] = [
    ("wilcoxon", lambda diffs: wilcoxon_test(diffs)),
    ("diebold_mariano", lambda diffs: diebold_mariano(diffs, hac_lags=_DM_HAC_LAGS)),
]


def _ci_dict(ci: CIResult) -> dict[str, Any]:
    """Serialize a CIResult to the pinned ``mae_ci`` shape."""
    return {"lo": ci.lo, "hi": ci.hi, "ci": ci.ci, "n_boot": ci.n_boot}


def _latency_dict(summary: RunSummary | None, predictor_id: str) -> dict[str, Any] | None:
    """Return the pinned ``latency`` shape for *predictor_id*, or None if unavailable."""
    if summary is None:
        return None
    stats = summary.inference_latency.get(predictor_id)
    if stats is None:
        return None
    return stats.to_dict()


def _keyed_means(rows: list[ValidationResult]) -> dict[tuple[str, str], float]:
    """Average |error_s| per (trip_key, stop_id) key."""
    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        buckets[(r.trip_key, r.stop_id)].append(abs(r.error_s))
    return {k: sum(v) / len(v) for k, v in buckets.items()}


def _paired_diffs(
    results_by_predictor: dict[str, list[ValidationResult]], a: str, b: str
) -> list[float]:
    """Per-(trip, stop) matched, within-key-averaged |error| differential ``a - b``.

    Only keys observed by both predictors contribute — this is the pairing
    step required before either significance test can run.
    """
    means_a = _keyed_means(results_by_predictor.get(a, []))
    means_b = _keyed_means(results_by_predictor.get(b, []))
    shared_keys = sorted(set(means_a) & set(means_b))
    return [means_a[k] - means_b[k] for k in shared_keys]


def _is_degenerate(diffs: list[float]) -> bool:
    """True when a differential series is unusable: too short, or constant."""
    if len(diffs) < 2:
        return True
    first = diffs[0]
    return all(d == first for d in diffs)


def compute_significance(
    results_by_predictor: dict[str, list[ValidationResult]],
) -> list[dict[str, Any]]:
    """Pairwise significance tests for every predictor pair, Holm-corrected per family.

    For each test family (wilcoxon, diebold_mariano[hac_lags=1]), every
    unordered predictor pair (a, b sorted) is tested once on the matched,
    per-(trip, stop)-averaged |error| differential. Degenerate differentials
    (n < 2, or all-equal) are skipped. p-values are Holm-corrected within
    each family independently.
    """
    predictor_ids = sorted(results_by_predictor)
    entries: list[dict[str, Any]] = []

    for family_name, test_fn in _TEST_FAMILIES:
        family_rows: list[dict[str, Any]] = []
        for a, b in combinations(predictor_ids, 2):
            diffs = _paired_diffs(results_by_predictor, a, b)
            if _is_degenerate(diffs):
                continue
            result = test_fn(diffs)
            family_rows.append(
                {
                    "a": a,
                    "b": b,
                    "test": family_name,
                    "statistic": result.statistic,
                    "p_value": result.p_value,
                    "n": result.n,
                    "diff_mean": sum(diffs) / len(diffs),
                }
            )

        if not family_rows:
            continue

        corrected = holm([row["p_value"] for row in family_rows])
        for row, p_adj, reject in zip(
            family_rows, corrected.adjusted_p, corrected.reject, strict=True
        ):
            row["p_adjusted"] = float(p_adj)
            row["reject"] = bool(reject)
        entries.extend(family_rows)

    return entries


def compute_calibration(results: list[ValidationResult]) -> list[dict[str, Any]]:
    """Calibration bins (nominal vs empirical coverage by horizon band) for the run."""
    if not results:
        return []
    bins = calibration_by_horizon(
        [r.predicted_arrival_epoch for r in results],
        [r.actual_arrival_epoch for r in results],
        [r.horizon_s for r in results],
    )
    return [asdict(b) for b in bins]


def compute_stats_extension(
    results: list[ValidationResult],
    summary: RunSummary | None,
    seed: int = STATS_SEED,
) -> dict[str, Any]:
    """Compute the full ``?stats=1`` augmentation for one run's results.

    Returns a dict with:
      - ``per_predictor_ext``: ``{predictor_id: {"mae_ci": {...}, "latency": {...} | None}}``
      - ``significance``: pairwise significance rows (see :func:`compute_significance`)
      - ``calibration``: calibration bins (see :func:`compute_calibration`)
      - ``seed``: the seed used (always deterministic)
    """
    ci_report = compute_metrics_with_ci(results, seed=seed)
    per_predictor_ext: dict[str, Any] = {
        pc.predictor_id: {
            "mae_ci": _ci_dict(pc.mae_ci),
            "latency": _latency_dict(summary, pc.predictor_id),
        }
        for pc in ci_report.per_predictor_ci
    }

    results_by_predictor: dict[str, list[ValidationResult]] = defaultdict(list)
    for r in results:
        results_by_predictor[r.predictor_id].append(r)

    return {
        "per_predictor_ext": per_predictor_ext,
        "significance": compute_significance(dict(results_by_predictor)),
        "calibration": compute_calibration(results),
        "seed": seed,
    }
