"""Detector sensitivity analysis (A5).

Answers: "how much does the choice of ground-truth detector move the
scored metrics?" by re-scoring the *same* PredictionSamples against arrivals
from each configured :class:`~etaval.domain.interfaces.GroundTruthDetector`,
instead of only the single ``primary_method`` the engine uses for live
incremental scoring.

Two entry points, with different fidelity
------------------------------------------
``compute_detector_sensitivity(samples, trip_contexts, detectors, ...)``
    Full-fidelity: re-runs ``detector.detect(trip, position_history, stops)``
    for every detector over the given trip contexts (trip + chronological
    position history + stops), producing a fresh set of arrivals per
    detector, then scores *samples* against them. This requires the raw
    per-tick position history — the same input the engine's live tick loop
    has, but call sites (tests, or an in-memory pipeline) must supply it
    directly via :class:`TripContext`.

``sensitivity_from_persisted_arrivals(samples, arrivals, ...)``
    Restricted / storage-backed: does **not** re-run detection at all. It
    groups the *already-persisted* ``ActualArrival`` rows by ``method`` and
    re-scores *samples* against each group.

HONEST LIMITATION — why the CLI ``--sensitivity --db`` path cannot re-detect
------------------------------------------------------------------------------
``etaval run`` persists ``PredictionSample`` and ``ActualArrival`` rows to
SQLite (see ``stoptimes.val.engine.runtime.run_engine`` — every configured detector's
arrivals are saved each tick, not only the ``primary_method`` one used for
scoring). It does **not** persist raw per-tick position history
(lat/lon/progress_m samples): only the detectors' *output* (arrivals) and the
predictors' *output* (samples) survive a run. That means true re-detection —
constructing a fresh :class:`TripContext` and calling
``compute_detector_sensitivity`` — is infeasible from storage alone: there is
no position history to feed a detector.

``sensitivity_from_persisted_arrivals`` is the honest alternative: it re-scores
each detector method's arrivals exactly as they were computed during the
original run, rather than fabricating history or silently skipping detectors.
This is what powers ``etaval report --sensitivity --db <sqlite>``. A
Parquet-only report (no ``--db``) carries neither samples-by-predictor keyed
to raw arrivals-by-method nor position history, so ``etaval report
--sensitivity`` without ``--db`` is declined outright with an explanatory
message (see ``stoptimes.val.cli.main.report``) instead of silently doing nothing
useful.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from stoptimes.core.domain.models import ActualArrival, PredictionSample, Stop, TripInstance, VehicleState
from stoptimes.val.scoring.metrics import PerPredictorMetrics, compute_metrics
from stoptimes.val.scoring.scorer import Scorer


@dataclass(frozen=True)
class TripContext:
    """One trip's inputs to a :class:`~stoptimes.val.domain.interfaces.GroundTruthDetector`.

    Mirrors the ``(trip, position_history, stops)`` arguments detectors
    receive from the engine tick loop, plus an optional ``route_id`` used
    only to enrich re-scored :class:`~stoptimes.core.domain.models.ValidationResult`
    rows the same way the engine's ``_route_id_map`` does.
    """

    trip: TripInstance
    position_history: list[VehicleState]
    stops: list[Stop]
    route_id: str | None = None


@dataclass(frozen=True)
class DetectorSensitivityRow:
    """One detector's re-scoring outcome across all predictors.

    Attributes:
        detector_id: The detector ``method`` this row summarizes (e.g.
            ``"shape_distance"``, ``"position_radius"``, ``"stop_status"``).
        metrics_by_predictor: ``{predictor_id: PerPredictorMetrics}`` computed
            by scoring ``samples`` against this detector's arrivals.
        delta_mae_vs_default: Pooled MAE (mean ``|error_s|`` across *all*
            matched results, all predictors combined) for this detector minus
            the pooled MAE for ``default_detector_id``. Exactly ``0.0`` for
            the default detector's own row. ``NaN`` when either side has no
            matched results.
    """

    detector_id: str
    metrics_by_predictor: dict[str, PerPredictorMetrics] = field(default_factory=dict)
    delta_mae_vs_default: float = math.nan


def _pooled_mae(results: list[Any]) -> float:
    """Mean |error_s| across all ValidationResults, or NaN if empty."""
    if not results:
        return math.nan
    return sum(abs(r.error_s) for r in results) / len(results)


def _score_and_summarize(
    detector_id: str,
    samples: list[PredictionSample],
    arrivals: list[ActualArrival],
    route_id_map: dict[str, str],
) -> tuple[dict[str, PerPredictorMetrics], float]:
    """Join *samples* to *arrivals* (as if *detector_id* were the primary method).

    Returns ``(metrics_by_predictor, pooled_mae)`` for the resulting
    ValidationResults.
    """
    scorer = Scorer(primary_method=detector_id)
    results = scorer.score(samples, arrivals, route_id_map)
    metrics_by_predictor = {m.predictor_id: m for m in compute_metrics(results).per_predictor}
    return metrics_by_predictor, _pooled_mae(results)


def _rows_from_pooled(
    per_detector: dict[str, tuple[dict[str, PerPredictorMetrics], float]],
    default_detector_id: str,
) -> list[DetectorSensitivityRow]:
    """Build DetectorSensitivityRows, computing delta_mae_vs_default per detector."""
    default_mae = per_detector.get(default_detector_id, ({}, math.nan))[1]
    rows: list[DetectorSensitivityRow] = []
    for detector_id, (metrics_by_predictor, pooled_mae) in per_detector.items():
        if math.isnan(pooled_mae) or math.isnan(default_mae):
            delta = math.nan
        else:
            delta = pooled_mae - default_mae
        rows.append(
            DetectorSensitivityRow(
                detector_id=detector_id,
                metrics_by_predictor=metrics_by_predictor,
                delta_mae_vs_default=delta,
            )
        )
    return rows


# ── Full-fidelity: re-run detection over supplied trip contexts ───────────────


def compute_detector_sensitivity(
    samples: list[PredictionSample],
    trip_contexts: list[TripContext],
    detectors: list[Any],
    default_detector_id: str = "shape_distance",
) -> list[DetectorSensitivityRow]:
    """Re-run every detector over *trip_contexts* and re-score *samples*.

    For each detector: call ``detector.detect(trip, position_history, stops)``
    across every :class:`TripContext`, pool the resulting arrivals, score
    *samples* against them (as if that detector were the primary method), and
    compute per-predictor metrics.

    Args:
        samples: All collected PredictionSamples (unchanged across detectors —
            only the ground truth they are scored against varies).
        trip_contexts: One entry per trip instance being re-evaluated, with
            its chronological position history and stops.
        detectors: GroundTruthDetector instances to compare (each must expose
            ``.method`` and ``.detect(trip, position_history, stops)``).
        default_detector_id: The detector ``method`` used as the baseline for
            ``delta_mae_vs_default`` (default ``"shape_distance"``, matching
            the engine's own default ``primary_method``).

    Returns:
        One :class:`DetectorSensitivityRow` per detector, in the order given.
        Empty list if *detectors* is empty.
    """
    route_id_map = {
        tc.trip.trip_key: tc.route_id for tc in trip_contexts if tc.route_id is not None
    }

    per_detector: dict[str, tuple[dict[str, PerPredictorMetrics], float]] = {}
    for detector in detectors:
        detector_id = detector.method
        arrivals: list[ActualArrival] = []
        for tc in trip_contexts:
            arrivals.extend(detector.detect(tc.trip, tc.position_history, tc.stops))
        per_detector[detector_id] = _score_and_summarize(
            detector_id, samples, arrivals, route_id_map
        )

    return _rows_from_pooled(per_detector, default_detector_id)


# ── Restricted: re-score already-persisted arrivals (SQLite-backed) ───────────


def sensitivity_from_persisted_arrivals(
    samples: list[PredictionSample],
    arrivals: list[ActualArrival],
    default_detector_id: str = "shape_distance",
) -> list[DetectorSensitivityRow]:
    """Re-score *samples* against already-persisted arrivals, grouped by method.

    This is the storage-backed counterpart to :func:`compute_detector_sensitivity`
    used by ``etaval report --sensitivity --db <sqlite>`` — see the module
    docstring's HONEST LIMITATION section for why this cannot re-run detection
    (raw position history is never persisted).

    Args:
        samples: PredictionSamples loaded from :meth:`SqliteStorage.load_samples`.
        arrivals: ActualArrivals loaded from :meth:`SqliteStorage.load_arrivals`
            (across all detector methods that ran during the original run).
        default_detector_id: The detector ``method`` used as the baseline for
            ``delta_mae_vs_default``.

    Returns:
        One :class:`DetectorSensitivityRow` per distinct ``method`` present in
        *arrivals*, sorted alphabetically. Empty list if *arrivals* is empty.
    """
    methods = sorted({a.method for a in arrivals})
    per_detector: dict[str, tuple[dict[str, PerPredictorMetrics], float]] = {}
    for method in methods:
        method_arrivals = [a for a in arrivals if a.method == method]
        # route_id is not persisted on PredictionSample/ActualArrival, so
        # per_route enrichment is simply unavailable here (route_id=None on
        # every resulting ValidationResult) -- this is the same trade-off
        # ``Scorer.score`` already makes when no route_id_map is supplied.
        per_detector[method] = _score_and_summarize(method, samples, method_arrivals, {})

    return _rows_from_pooled(per_detector, default_detector_id)
