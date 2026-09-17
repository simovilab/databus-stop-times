"""Tests for etaval/stats/report.py."""

from __future__ import annotations

import pytest

from etaval.domain.models import ValidationResult
from etaval.stats.report import PredictorWithCI, ReportWithCI, compute_metrics_with_ci

_TRIP_KEY = "src:trip1:v1:20240101"


def _result(
    stop_id: str,
    error_s: float,
    horizon_s: float,
    predictor_id: str = "gtfsrt",
    stop_sequence: int = 1,
) -> ValidationResult:
    actual = 2000.0
    predicted = actual + error_s
    made_at = actual - horizon_s
    return ValidationResult(
        trip_key=_TRIP_KEY,
        predictor_id=predictor_id,
        stop_id=stop_id,
        stop_sequence=stop_sequence,
        route_id="r1",
        error_s=error_s,
        horizon_s=horizon_s,
        dist_remaining_m=None,
        stops_remaining=None,
        predicted_arrival_epoch=predicted,
        actual_arrival_epoch=actual,
        made_at_epoch=made_at,
    )


class TestComputeMetricsWithCI:
    def test_empty_input(self) -> None:
        report = compute_metrics_with_ci([])
        assert isinstance(report, ReportWithCI)
        assert report.per_predictor_ci == []
        assert report.base.per_predictor == []

    def test_single_predictor_point_matches_mae(self) -> None:
        results = [
            _result("s1", error_s=30.0, horizon_s=300.0),
            _result("s2", error_s=-10.0, horizon_s=300.0),
            _result("s3", error_s=20.0, horizon_s=300.0),
        ]
        report = compute_metrics_with_ci(results, n_boot=200, seed=0)
        assert len(report.per_predictor_ci) == 1
        entry = report.per_predictor_ci[0]
        assert isinstance(entry, PredictorWithCI)
        base_mae = report.base.per_predictor[0].mae
        assert entry.mae == pytest.approx(base_mae)
        assert entry.mae_ci.lo <= entry.mae <= entry.mae_ci.hi

    def test_multiple_predictors_independent_ci(self) -> None:
        results = [
            _result("s1", error_s=10.0, horizon_s=300.0, predictor_id="gtfsrt"),
            _result("s1", error_s=5.0, horizon_s=300.0, predictor_id="baseline:constant_speed"),
            _result("s2", error_s=15.0, horizon_s=300.0, predictor_id="gtfsrt"),
        ]
        report = compute_metrics_with_ci(results, n_boot=100, seed=1)
        pids = {p.predictor_id for p in report.per_predictor_ci}
        assert pids == {"gtfsrt", "baseline:constant_speed"}

    def test_deterministic_under_seed(self) -> None:
        results = [
            _result("s1", error_s=30.0, horizon_s=300.0),
            _result("s2", error_s=-10.0, horizon_s=300.0),
            _result("s3", error_s=20.0, horizon_s=300.0),
        ]
        r1 = compute_metrics_with_ci(results, n_boot=200, seed=7)
        r2 = compute_metrics_with_ci(results, n_boot=200, seed=7)
        assert r1.per_predictor_ci[0].mae_ci == r2.per_predictor_ci[0].mae_ci

    def test_base_report_matches_compute_metrics(self) -> None:
        from etaval.scoring.metrics import compute_metrics

        results = [_result("s1", error_s=10.0, horizon_s=300.0)]
        report = compute_metrics_with_ci(results)
        expected_base = compute_metrics(results)
        assert report.base.per_predictor[0].mae == expected_base.per_predictor[0].mae

    def test_count_matches_predictor_observations(self) -> None:
        results = [
            _result("s1", error_s=10.0, horizon_s=300.0),
            _result("s2", error_s=20.0, horizon_s=300.0),
        ]
        report = compute_metrics_with_ci(results, n_boot=50, seed=0)
        assert report.per_predictor_ci[0].count == 2
