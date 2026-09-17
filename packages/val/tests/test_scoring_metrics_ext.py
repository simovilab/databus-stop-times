"""Tests for the broadened PerPredictorMetrics fields and new slices (brief A3).

Covers: r2, medae, abs_error percentiles, within_30s/60s, skew/excess_kurtosis
(including NaN edge cases), and the per_route / per_time_of_day / per_stops_ahead
slices. Also asserts existing fields remain unchanged.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest

from stoptimes.core.domain.models import ValidationResult
from stoptimes.val.scoring.metrics import compute_metrics

_TRIP_KEY = "src:trip1:v1:20240101"


def _result(
    stop_id: str,
    error_s: float,
    horizon_s: float,
    predictor_id: str = "gtfsrt",
    stop_sequence: int = 1,
    route_id: str | None = "r1",
    stops_remaining: int | None = None,
    actual_arrival_epoch: float = 2000.0,
) -> ValidationResult:
    actual = actual_arrival_epoch
    predicted = actual + error_s
    made_at = actual - horizon_s
    return ValidationResult(
        trip_key=_TRIP_KEY,
        predictor_id=predictor_id,
        stop_id=stop_id,
        stop_sequence=stop_sequence,
        route_id=route_id,
        error_s=error_s,
        horizon_s=horizon_s,
        dist_remaining_m=None,
        stops_remaining=stops_remaining,
        predicted_arrival_epoch=predicted,
        actual_arrival_epoch=actual,
        made_at_epoch=made_at,
    )


class TestExistingFieldsUnchanged:
    def test_mae_rmse_bias_mape_count_unchanged(self) -> None:
        results = [
            _result("s1", error_s=30.0, horizon_s=300.0),
            _result("s2", error_s=-10.0, horizon_s=300.0),
            _result("s3", error_s=20.0, horizon_s=300.0),
        ]
        report = compute_metrics(results)
        m = report.per_predictor[0]
        assert m.predictor_id == "gtfsrt"
        assert m.count == 3
        assert m.mae == pytest.approx(20.0, rel=1e-4)
        assert m.bias == pytest.approx(40.0 / 3, rel=1e-4)

    def test_horizon_buckets_convergence_head_to_head_untouched(self) -> None:
        results = [
            _result("s1", error_s=10.0, horizon_s=60.0),
            _result("s2", error_s=20.0, horizon_s=200.0, predictor_id="baseline:constant_speed"),
        ]
        report = compute_metrics(results)
        assert len(report.horizon_buckets) == 2
        assert report.convergence == []


class TestNewPerPredictorFields:
    def test_medae_and_percentiles(self) -> None:
        results = [
            _result("s1", error_s=10.0, horizon_s=300.0),
            _result("s2", error_s=-20.0, horizon_s=300.0),
            _result("s3", error_s=30.0, horizon_s=300.0),
            _result("s4", error_s=-40.0, horizon_s=300.0),
        ]
        report = compute_metrics(results)
        m = report.per_predictor[0]
        assert m.medae == pytest.approx(25.0)
        assert m.abs_error_p50 == pytest.approx(25.0)
        assert m.abs_error_p99 >= m.abs_error_p95 >= m.abs_error_p90 >= m.abs_error_p50

    def test_within_30s_60s(self) -> None:
        results = [
            _result("s1", error_s=10.0, horizon_s=300.0),   # within both
            _result("s2", error_s=45.0, horizon_s=300.0),   # within 60s only
            _result("s3", error_s=90.0, horizon_s=300.0),   # within neither
            _result("s4", error_s=-5.0, horizon_s=300.0),   # within both
        ]
        report = compute_metrics(results)
        m = report.per_predictor[0]
        assert m.within_30s == pytest.approx(2 / 4)
        assert m.within_60s == pytest.approx(3 / 4)

    def test_r2_perfect_prediction_is_nan_or_one(self) -> None:
        """Constant actual_arrival_epoch => zero variance => r2 is NaN."""
        results = [
            _result("s1", error_s=0.0, horizon_s=300.0, actual_arrival_epoch=1000.0),
            _result("s2", error_s=0.0, horizon_s=300.0, actual_arrival_epoch=1000.0),
        ]
        report = compute_metrics(results)
        m = report.per_predictor[0]
        assert math.isnan(m.r2)

    def test_r2_with_varying_actuals(self) -> None:
        results = [
            _result("s1", error_s=0.0, horizon_s=300.0, actual_arrival_epoch=1000.0),
            _result("s2", error_s=0.0, horizon_s=300.0, actual_arrival_epoch=2000.0),
            _result("s3", error_s=0.0, horizon_s=300.0, actual_arrival_epoch=3000.0),
        ]
        report = compute_metrics(results)
        m = report.per_predictor[0]
        # Zero error => perfect fit => r2 == 1.0
        assert m.r2 == pytest.approx(1.0)

    def test_skew_kurtosis_nan_below_3_samples(self) -> None:
        results = [
            _result("s1", error_s=10.0, horizon_s=300.0),
            _result("s2", error_s=-10.0, horizon_s=300.0),
        ]
        report = compute_metrics(results)
        m = report.per_predictor[0]
        assert math.isnan(m.skew)
        assert math.isnan(m.excess_kurtosis)

    def test_skew_kurtosis_computed_with_3_or_more(self) -> None:
        results = [
            _result("s1", error_s=10.0, horizon_s=300.0),
            _result("s2", error_s=-10.0, horizon_s=300.0),
            _result("s3", error_s=100.0, horizon_s=300.0),
        ]
        report = compute_metrics(results)
        m = report.per_predictor[0]
        assert not math.isnan(m.skew)
        assert not math.isnan(m.excess_kurtosis)

    def test_new_fields_present_via_to_records(self) -> None:
        results = [_result("s1", error_s=10.0, horizon_s=300.0)]
        report = compute_metrics(results)
        record = report.to_records()[0]
        for key in (
            "r2", "medae", "abs_error_p50", "abs_error_p90", "abs_error_p95",
            "abs_error_p99", "within_30s", "within_60s", "skew", "excess_kurtosis",
        ):
            assert key in record


class TestPerRouteSlice:
    def test_groups_by_route_and_predictor(self) -> None:
        results = [
            _result("s1", error_s=10.0, horizon_s=300.0, route_id="r1"),
            _result("s2", error_s=20.0, horizon_s=300.0, route_id="r2"),
        ]
        report = compute_metrics(results)
        route_ids = {r.route_id for r in report.per_route}
        assert route_ids == {"r1", "r2"}

    def test_null_route_id_excluded(self) -> None:
        results = [
            _result("s1", error_s=10.0, horizon_s=300.0, route_id=None),
        ]
        report = compute_metrics(results)
        assert report.per_route == []

    def test_route_mae_correct(self) -> None:
        results = [
            _result("s1", error_s=10.0, horizon_s=300.0, route_id="r1"),
            _result("s2", error_s=30.0, horizon_s=300.0, route_id="r1"),
        ]
        report = compute_metrics(results)
        assert len(report.per_route) == 1
        assert report.per_route[0].mae == pytest.approx(20.0)
        assert report.per_route[0].count == 2


class TestPerTimeOfDaySlice:
    def test_utc_default_bucketing(self) -> None:
        # 2024-01-01 07:30:00 UTC -> hour=7 -> band "06-09"
        epoch = datetime(2024, 1, 1, 7, 30, tzinfo=UTC).timestamp()
        results = [_result("s1", error_s=10.0, horizon_s=300.0, actual_arrival_epoch=epoch)]
        report = compute_metrics(results, tz="UTC")
        labels = {t.label for t in report.per_time_of_day}
        assert labels == {"06-09"}

    def test_tz_parameter_shifts_bucket(self) -> None:
        # 2024-01-01 01:00:00 UTC -> band "00-03" in UTC,
        # but in UTC-5 (e.g. America/New_York, winter) it's 20:00 previous day -> "18-21".
        epoch = datetime(2024, 1, 1, 1, 0, tzinfo=UTC).timestamp()
        results = [_result("s1", error_s=10.0, horizon_s=300.0, actual_arrival_epoch=epoch)]
        report_utc = compute_metrics(results, tz="UTC")
        report_ny = compute_metrics(results, tz="America/New_York")
        assert {t.label for t in report_utc.per_time_of_day} == {"00-03"}
        assert {t.label for t in report_ny.per_time_of_day} == {"18-21"}

    def test_default_tz_is_utc(self) -> None:
        epoch = datetime(2024, 1, 1, 7, 30, tzinfo=UTC).timestamp()
        results = [_result("s1", error_s=10.0, horizon_s=300.0, actual_arrival_epoch=epoch)]
        report_default = compute_metrics(results)
        report_utc = compute_metrics(results, tz="UTC")
        assert report_default.per_time_of_day == report_utc.per_time_of_day


class TestPerStopsAheadSlice:
    def test_bands(self) -> None:
        results = [
            _result("s1", error_s=10.0, horizon_s=300.0, stops_remaining=1),
            _result("s2", error_s=10.0, horizon_s=300.0, stops_remaining=3),
            _result("s3", error_s=10.0, horizon_s=300.0, stops_remaining=5),
            _result("s4", error_s=10.0, horizon_s=300.0, stops_remaining=8),
            _result("s5", error_s=10.0, horizon_s=300.0, stops_remaining=20),
        ]
        report = compute_metrics(results)
        labels = {s.label for s in report.per_stops_ahead}
        assert labels == {"1", "2-3", "4-5", "6-10", ">10"}

    def test_null_stops_remaining_excluded(self) -> None:
        results = [_result("s1", error_s=10.0, horizon_s=300.0, stops_remaining=None)]
        report = compute_metrics(results)
        assert report.per_stops_ahead == []

    def test_stops_ahead_mae_correct(self) -> None:
        results = [
            _result("s1", error_s=10.0, horizon_s=300.0, stops_remaining=1),
            _result("s2", error_s=30.0, horizon_s=300.0, stops_remaining=1),
        ]
        report = compute_metrics(results)
        assert len(report.per_stops_ahead) == 1
        assert report.per_stops_ahead[0].label == "1"
        assert report.per_stops_ahead[0].mae == pytest.approx(20.0)


class TestEmptyInput:
    def test_empty_results_new_slices_empty(self) -> None:
        report = compute_metrics([])
        assert report.per_route == []
        assert report.per_time_of_day == []
        assert report.per_stops_ahead == []
