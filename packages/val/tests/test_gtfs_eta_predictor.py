"""Tests for GtfsEtaModelPredictor (etaval/predictors/gtfs_eta_models.py).

Mocks ``estimate_stop_times`` rather than exercising a real trained model —
this predictor's own job is just the etaval <-> eta_service payload bridge:
building the vehicle_position/upcoming_stops dicts, calling through, and
converting the result back into StopPrediction objects (or skipping stops the
model couldn't score).
"""

from __future__ import annotations

from unittest.mock import patch

from etaval.domain.models import Stop, TripInstance, VehicleState
from etaval.predictors.gtfs_eta_models import GtfsEtaModelPredictor

TRIP = TripInstance(
    source="test",
    trip_id="t1",
    vehicle_id="v1",
    route_id="222",
    direction_id=0,
    shape_id="s1",
    start_date="20260701",
    started_at=1_000_000.0,
)

STATE = VehicleState(
    epoch=1_000_100.0,
    lat=42.1,
    lon=-71.0,
    progress_m=200.0,
    speed=5.0,
    current_status="IN_TRANSIT_TO",
)


def _stop(stop_id: str, stop_sequence: int) -> Stop:
    return Stop(stop_id=stop_id, stop_sequence=stop_sequence, lat=42.2, lon=-71.1)


class TestGtfsEtaModelPredictor:
    def test_predictor_id_includes_model_type(self):
        assert GtfsEtaModelPredictor("xgboost").predictor_id == "gtfs_eta:xgboost"
        assert GtfsEtaModelPredictor("polyreg_time").predictor_id == "gtfs_eta:polyreg_time"

    def test_no_upcoming_stops_returns_empty_without_calling_estimator(self):
        predictor = GtfsEtaModelPredictor("xgboost")
        with patch("etaval.predictors.gtfs_eta_models.estimate_stop_times") as mock_est:
            result = predictor.predict(TRIP, [], STATE, None)
        assert result == []
        mock_est.assert_not_called()

    def test_builds_stop_predictions_from_estimator_result(self):
        predictor = GtfsEtaModelPredictor("xgboost")
        stops = [_stop("s1", 1), _stop("s2", 2)]
        fake_result = {
            "predictions": [
                {"stop_id": "s1", "stop_sequence": 1, "eta_seconds": 120.0, "distance_to_stop_m": 500.0},
                {"stop_id": "s2", "stop_sequence": 2, "eta_seconds": 300.0, "distance_to_stop_m": 1200.0},
            ]
        }
        with patch(
            "etaval.predictors.gtfs_eta_models.estimate_stop_times", return_value=fake_result
        ) as mock_est:
            preds = predictor.predict(TRIP, stops, STATE, None)

        assert len(preds) == 2
        assert preds[0].stop_id == "s1"
        assert preds[0].predicted_arrival_epoch == STATE.epoch + 120.0
        assert preds[0].made_at_epoch == STATE.epoch
        assert preds[0].dist_remaining_m == 500.0
        assert preds[0].stops_remaining == 2
        assert preds[1].stops_remaining == 1

        # Verify the payload passed through to the estimator.
        call_kwargs = mock_est.call_args.kwargs
        assert call_kwargs["route_id"] == "222"
        assert call_kwargs["trip_id"] == "t1"
        assert call_kwargs["model_type"] == "xgboost"
        assert call_kwargs["vehicle_position"]["vehicle_id"] == "v1"
        assert call_kwargs["upcoming_stops"] == [
            {"stop_id": "s1", "stop_sequence": 1, "lat": 42.2, "lon": -71.1},
            {"stop_id": "s2", "stop_sequence": 2, "lat": 42.2, "lon": -71.1},
        ]

    def test_skips_stops_with_no_eta(self):
        predictor = GtfsEtaModelPredictor("xgboost")
        stops = [_stop("s1", 1)]
        fake_result = {
            "predictions": [
                {"stop_id": "s1", "stop_sequence": 1, "eta_seconds": None, "error": "no model"},
            ]
        }
        with patch(
            "etaval.predictors.gtfs_eta_models.estimate_stop_times", return_value=fake_result
        ):
            preds = predictor.predict(TRIP, stops, STATE, None)
        assert preds == []

    def test_explicit_model_key_passed_through(self):
        predictor = GtfsEtaModelPredictor("polyreg_time", model_key="polyreg_time_x_y_global_v1")
        with patch(
            "etaval.predictors.gtfs_eta_models.estimate_stop_times", return_value={"predictions": []}
        ) as mock_est:
            predictor.predict(TRIP, [_stop("s1", 1)], STATE, None)
        assert mock_est.call_args.kwargs["model_key"] == "polyreg_time_x_y_global_v1"
