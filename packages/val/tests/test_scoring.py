"""Tests for etaval/scoring/scorer.py and etaval/scoring/metrics.py. Phase P5."""

from __future__ import annotations

import math

import pytest

from etaval.domain.models import ActualArrival, PredictionSample, ValidationResult
from etaval.scoring.metrics import MetricsReport, _assign_bucket, compute_metrics
from etaval.scoring.scorer import Scorer

# ── Helpers ───────────────────────────────────────────────────────────────────

_TRIP_KEY = "src:trip1:v1:20240101"
_ROUTE_ID = "r1"


def _sample(
    stop_id: str,
    predicted: float,
    made_at: float,
    predictor_id: str = "gtfsrt",
    stop_sequence: int = 1,
    dist_remaining_m: float | None = None,
) -> PredictionSample:
    return PredictionSample(
        trip_key=_TRIP_KEY,
        predictor_id=predictor_id,
        stop_id=stop_id,
        stop_sequence=stop_sequence,
        predicted_arrival_epoch=predicted,
        made_at_epoch=made_at,
        vehicle_progress_m_at_made=None,
        dist_remaining_m=dist_remaining_m,
        stops_remaining=None,
    )


def _arrival(
    stop_id: str,
    actual: float,
    method: str = "shape_distance",
    stop_sequence: int = 1,
) -> ActualArrival:
    return ActualArrival(
        trip_key=_TRIP_KEY,
        stop_id=stop_id,
        stop_sequence=stop_sequence,
        actual_arrival_epoch=actual,
        method=method,
    )


def _result(
    stop_id: str,
    error_s: float,
    horizon_s: float,
    predictor_id: str = "gtfsrt",
    stop_sequence: int = 1,
) -> ValidationResult:
    """Build a ValidationResult with controlled error_s and horizon_s."""
    actual = 2000.0
    predicted = actual + error_s
    made_at = actual - horizon_s
    return ValidationResult(
        trip_key=_TRIP_KEY,
        predictor_id=predictor_id,
        stop_id=stop_id,
        stop_sequence=stop_sequence,
        route_id=_ROUTE_ID,
        error_s=error_s,
        horizon_s=horizon_s,
        dist_remaining_m=None,
        stops_remaining=None,
        predicted_arrival_epoch=predicted,
        actual_arrival_epoch=actual,
        made_at_epoch=made_at,
    )


# ── Scorer tests ──────────────────────────────────────────────────────────────


class TestScorer:
    def test_exact_error_s(self) -> None:
        """Known error_s is correctly computed via from_sample_and_arrival."""
        actual_epoch = 1000.0
        predicted_epoch = 1060.0  # +60 s late
        made_at = 900.0

        sample = _sample("s1", predicted=predicted_epoch, made_at=made_at)
        arrival = _arrival("s1", actual=actual_epoch)
        scorer = Scorer()
        results = scorer.score([sample], [arrival])

        assert len(results) == 1
        r = results[0]
        assert r.error_s == pytest.approx(60.0)
        assert r.horizon_s == pytest.approx(actual_epoch - made_at)  # 100 s

    def test_exact_early_prediction(self) -> None:
        """Negative error_s for an early (optimistic) prediction."""
        actual_epoch = 1000.0
        predicted_epoch = 950.0  # 50 s early
        made_at = 800.0

        sample = _sample("s2", predicted=predicted_epoch, made_at=made_at, stop_sequence=2)
        arrival = _arrival("s2", actual=actual_epoch, stop_sequence=2)
        results = Scorer().score([sample], [arrival])

        assert len(results) == 1
        assert results[0].error_s == pytest.approx(-50.0)

    def test_no_match_returns_empty(self) -> None:
        """Samples without a matching arrival produce no results."""
        sample = _sample("s1", predicted=1000.0, made_at=900.0)
        results = Scorer().score([sample], [])
        assert results == []

    def test_primary_method_filtering(self) -> None:
        """Only arrivals of primary_method drive scoring; others are ignored."""
        actual_epoch = 1000.0
        sample = _sample("s1", predicted=1010.0, made_at=900.0)

        arrival_sd = _arrival("s1", actual=actual_epoch, method="shape_distance")
        arrival_pr = _arrival("s1", actual=actual_epoch + 5, method="position_radius")
        arrival_ss = _arrival("s1", actual=actual_epoch + 10, method="stop_status")

        scorer = Scorer(primary_method="shape_distance")
        results = scorer.score([sample], [arrival_sd, arrival_pr, arrival_ss])

        # Only one result (from shape_distance), not three.
        assert len(results) == 1
        assert results[0].actual_arrival_epoch == pytest.approx(actual_epoch)

    def test_non_primary_method_is_skipped(self) -> None:
        """When only a non-primary arrival is present, no results are produced."""
        sample = _sample("s1", predicted=1000.0, made_at=900.0)
        arrival = _arrival("s1", actual=990.0, method="position_radius")
        scorer = Scorer(primary_method="shape_distance")
        assert scorer.score([sample], [arrival]) == []

    def test_multiple_predictors_multiple_stops(self) -> None:
        """Each sample is independently joined to the matching arrival."""
        actual_s1 = 1000.0
        actual_s2 = 1100.0

        samples = [
            _sample("s1", predicted=1005.0, made_at=900.0, predictor_id="gtfsrt"),
            _sample("s1", predicted=1003.0, made_at=900.0, predictor_id="baseline:constant_speed"),
            _sample("s2", predicted=1110.0, made_at=950.0, predictor_id="gtfsrt", stop_sequence=2),
        ]
        arrivals = [
            _arrival("s1", actual=actual_s1),
            _arrival("s2", actual=actual_s2, stop_sequence=2),
        ]
        results = Scorer().score(samples, arrivals)
        assert len(results) == 3

        by_key = {(r.stop_id, r.predictor_id): r for r in results}
        assert by_key[("s1", "gtfsrt")].error_s == pytest.approx(5.0)
        assert by_key[("s1", "baseline:constant_speed")].error_s == pytest.approx(3.0)
        assert by_key[("s2", "gtfsrt")].error_s == pytest.approx(10.0)

    def test_route_id_enrichment(self) -> None:
        """route_id from route_id_map propagates to ValidationResult."""
        sample = _sample("s1", predicted=1000.0, made_at=900.0)
        arrival = _arrival("s1", actual=990.0)
        results = Scorer().score(
            [sample], [arrival], route_id_map={_TRIP_KEY: "route_99"}
        )
        assert results[0].route_id == "route_99"

    def test_score_stop_incremental(self) -> None:
        """score_stop returns results only for the specified (trip_key, stop_id)."""
        samples = [
            _sample("s1", predicted=1010.0, made_at=900.0, predictor_id="gtfsrt"),
            _sample("s1", predicted=1008.0, made_at=900.0, predictor_id="baseline:constant_speed"),
            _sample("s2", predicted=1110.0, made_at=950.0, predictor_id="gtfsrt", stop_sequence=2),
        ]
        arrival = _arrival("s1", actual=1000.0)

        results = Scorer().score_stop(
            trip_key=_TRIP_KEY,
            stop_id="s1",
            arrival=arrival,
            samples=samples,
            route_id="r1",
        )
        # Only 2 results: the s2 sample is excluded.
        assert len(results) == 2
        assert all(r.stop_id == "s1" for r in results)
        errors = {r.predictor_id: r.error_s for r in results}
        assert errors["gtfsrt"] == pytest.approx(10.0)
        assert errors["baseline:constant_speed"] == pytest.approx(8.0)

    def test_score_stop_different_trip_key_excluded(self) -> None:
        """Samples from a different trip_key are not scored."""
        sample = PredictionSample(
            trip_key="other:trip:v2:20240101",
            predictor_id="gtfsrt",
            stop_id="s1",
            stop_sequence=1,
            predicted_arrival_epoch=1000.0,
            made_at_epoch=900.0,
        )
        arrival = _arrival("s1", actual=990.0)
        results = Scorer().score_stop(
            trip_key=_TRIP_KEY,
            stop_id="s1",
            arrival=arrival,
            samples=[sample],
        )
        assert results == []


# ── Metrics tests ─────────────────────────────────────────────────────────────


class TestAssignBucket:
    def test_buckets(self) -> None:
        assert _assign_bucket(0.0) == "<2m"
        assert _assign_bucket(119.9) == "<2m"
        assert _assign_bucket(120.0) == "2-5m"
        assert _assign_bucket(299.9) == "2-5m"
        assert _assign_bucket(300.0) == "5-10m"
        assert _assign_bucket(599.9) == "5-10m"
        assert _assign_bucket(600.0) == ">10m"
        assert _assign_bucket(3600.0) == ">10m"


class TestComputeMetrics:
    def test_empty_input_no_crash(self) -> None:
        """compute_metrics([]) returns an empty MetricsReport without raising."""
        report = compute_metrics([])
        assert isinstance(report, MetricsReport)
        assert report.per_predictor == []
        assert report.horizon_buckets == []
        assert report.convergence == []
        assert report.head_to_head == []

    def test_mae_rmse_bias(self) -> None:
        """Known errors → exact MAE / RMSE / bias values."""
        # errors: +30, -10, +20  → mean=13.33, MAE=20, RMSE=sqrt((900+100+400)/3)=sqrt(466.7)
        results = [
            _result("s1", error_s=30.0, horizon_s=300.0),
            _result("s2", error_s=-10.0, horizon_s=300.0),
            _result("s3", error_s=20.0, horizon_s=300.0),
        ]
        report = compute_metrics(results)
        assert len(report.per_predictor) == 1
        m = report.per_predictor[0]
        assert m.predictor_id == "gtfsrt"
        assert m.count == 3
        assert m.mae == pytest.approx(20.0, rel=1e-4)
        expected_rmse = math.sqrt((900 + 100 + 400) / 3)
        assert m.rmse == pytest.approx(expected_rmse, rel=1e-4)
        assert m.bias == pytest.approx((30 - 10 + 20) / 3, rel=1e-4)

    def test_multiple_predictors(self) -> None:
        """Two predictors produce two per-predictor rows."""
        results = [
            _result("s1", error_s=10.0, horizon_s=300.0, predictor_id="gtfsrt"),
            _result("s1", error_s=5.0, horizon_s=300.0, predictor_id="baseline:constant_speed"),
        ]
        report = compute_metrics(results)
        pids = {m.predictor_id for m in report.per_predictor}
        assert "gtfsrt" in pids
        assert "baseline:constant_speed" in pids

    def test_horizon_bucket_assignment(self) -> None:
        """Results land in the correct horizon buckets."""
        results = [
            _result("s1", error_s=10.0, horizon_s=60.0),    # <2m
            _result("s2", error_s=20.0, horizon_s=200.0),   # 2-5m
            _result("s3", error_s=30.0, horizon_s=400.0),   # 5-10m
            _result("s4", error_s=40.0, horizon_s=700.0),   # >10m
        ]
        report = compute_metrics(results)
        bucket_labels = {b.label for b in report.horizon_buckets}
        assert "<2m" in bucket_labels
        assert "2-5m" in bucket_labels
        assert "5-10m" in bucket_labels
        assert ">10m" in bucket_labels

        by_label = {b.label: b for b in report.horizon_buckets}
        assert by_label["<2m"].mae == pytest.approx(10.0)
        assert by_label["2-5m"].mae == pytest.approx(20.0)
        assert by_label["5-10m"].mae == pytest.approx(30.0)
        assert by_label[">10m"].mae == pytest.approx(40.0)

    def test_head_to_head_win_rate(self) -> None:
        """Head-to-head: predictor with smaller |error| wins the contested stop."""
        # Two predictors for the same (trip, stop): gtfsrt wins (|5| < |20|).
        results = [
            _result("s1", error_s=5.0, horizon_s=300.0, predictor_id="gtfsrt"),
            _result("s1", error_s=20.0, horizon_s=300.0, predictor_id="baseline:constant_speed"),
        ]
        report = compute_metrics(results)
        assert len(report.head_to_head) == 2

        by_pid = {w.predictor_id: w for w in report.head_to_head}
        assert by_pid["gtfsrt"].wins == 1
        assert by_pid["gtfsrt"].win_fraction == pytest.approx(1.0)
        assert by_pid["baseline:constant_speed"].wins == 0
        assert by_pid["baseline:constant_speed"].win_fraction == pytest.approx(0.0)

    def test_head_to_head_no_contested_stops(self) -> None:
        """When each stop has only one predictor, head_to_head is empty."""
        results = [
            _result("s1", error_s=10.0, horizon_s=300.0, predictor_id="gtfsrt"),
            _result("s2", error_s=15.0, horizon_s=300.0, predictor_id="baseline:constant_speed"),
        ]
        report = compute_metrics(results)
        assert report.head_to_head == []

    def test_convergence_converging(self) -> None:
        """A series where |error| decreases as horizon shrinks is flagged converging.

        settling_horizon_s is the LARGEST horizon at which |error| first drops
        below 60 s (scanning from largest→smallest horizon, i.e. earliest prediction
        onwards).  With errors 90 @ 300, 50 @ 200, 10 @ 100 the first <60 hit
        scanning 300→200→100 is at horizon=200.
        """
        actual = 2000.0
        results = [
            ValidationResult(
                trip_key=_TRIP_KEY, predictor_id="gtfsrt", stop_id="s1", stop_sequence=1,
                route_id=_ROUTE_ID, error_s=90.0, horizon_s=300.0,
                dist_remaining_m=None, stops_remaining=None,
                predicted_arrival_epoch=actual + 90.0,
                actual_arrival_epoch=actual,
                made_at_epoch=actual - 300.0,
            ),
            ValidationResult(
                trip_key=_TRIP_KEY, predictor_id="gtfsrt", stop_id="s1", stop_sequence=1,
                route_id=_ROUTE_ID, error_s=50.0, horizon_s=200.0,
                dist_remaining_m=None, stops_remaining=None,
                predicted_arrival_epoch=actual + 50.0,
                actual_arrival_epoch=actual,
                made_at_epoch=actual - 200.0,
            ),
            ValidationResult(
                trip_key=_TRIP_KEY, predictor_id="gtfsrt", stop_id="s1", stop_sequence=1,
                route_id=_ROUTE_ID, error_s=10.0, horizon_s=100.0,
                dist_remaining_m=None, stops_remaining=None,
                predicted_arrival_epoch=actual + 10.0,
                actual_arrival_epoch=actual,
                made_at_epoch=actual - 100.0,
            ),
        ]
        report = compute_metrics(results)
        assert len(report.convergence) == 1
        conv = report.convergence[0]
        assert conv.is_converging is True
        # settling at horizon=200 s (|error|=50 < 60); horizon=300 had |error|=90 > 60
        assert conv.settling_horizon_s == pytest.approx(200.0)

    def test_convergence_non_converging(self) -> None:
        """A series where |error| oscillates is flagged non-converging."""
        actual = 2000.0
        results = [
            ValidationResult(
                trip_key=_TRIP_KEY, predictor_id="gtfsrt", stop_id="s1", stop_sequence=1,
                route_id=_ROUTE_ID, error_s=10.0, horizon_s=300.0,
                dist_remaining_m=None, stops_remaining=None,
                predicted_arrival_epoch=actual + 10.0,
                actual_arrival_epoch=actual,
                made_at_epoch=actual - 300.0,
            ),
            ValidationResult(
                trip_key=_TRIP_KEY, predictor_id="gtfsrt", stop_id="s1", stop_sequence=1,
                route_id=_ROUTE_ID, error_s=50.0, horizon_s=200.0,
                dist_remaining_m=None, stops_remaining=None,
                predicted_arrival_epoch=actual + 50.0,
                actual_arrival_epoch=actual,
                made_at_epoch=actual - 200.0,
            ),
        ]
        report = compute_metrics(results)
        assert len(report.convergence) == 1
        assert report.convergence[0].is_converging is False

    def test_summary_and_to_records(self) -> None:
        """summary() and to_records() return non-empty dicts/lists when data exists."""
        results = [_result("s1", error_s=10.0, horizon_s=300.0)]
        report = compute_metrics(results)
        summary = report.summary()
        assert "per_predictor" in summary
        assert len(summary["per_predictor"]) == 1

        records = report.to_records()
        assert len(records) == 1
        assert "mae" in records[0]
        assert "predictor_id" in records[0]
