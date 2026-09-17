"""Tests for contracts.prediction: field invariants and the GTFS-RT dict mapping."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from stoptimes.core.contracts.prediction import (
    StopScheduleRelationship,
    StopTimePrediction,
    TripScheduleRelationship,
    TripUpdatePrediction,
)

NOW = datetime(2026, 9, 17, 12, 0, 0, tzinfo=timezone.utc)


def _stop_time_prediction(**overrides) -> StopTimePrediction:
    base = dict(
        trip_id="t1",
        route_id="r1",
        start_date="20260917",
        stop_id="s1",
        stop_sequence=3,
        vehicle_id="v1",
        predicted_arrival_time=NOW,
        delay_s=45.0,
        uncertainty_s=30.0,
        model_id="xgboost:v1",
        generated_at=NOW,
    )
    base.update(overrides)
    return StopTimePrediction(**base)


def test_naive_datetime_rejected():
    with pytest.raises(ValidationError):
        _stop_time_prediction(predicted_arrival_time=datetime(2026, 9, 17, 12, 0, 0))


def test_non_utc_datetime_rejected():
    from datetime import timedelta, timezone as tz

    est = tz(timedelta(hours=-5))
    with pytest.raises(ValidationError):
        _stop_time_prediction(predicted_arrival_time=datetime(2026, 9, 17, 7, 0, 0, tzinfo=est))


def test_requires_arrival_or_departure_unless_skipped():
    with pytest.raises(ValidationError):
        _stop_time_prediction(predicted_arrival_time=None, predicted_departure_time=None)

    # SKIPPED is allowed with neither time set
    skipped = _stop_time_prediction(
        predicted_arrival_time=None,
        predicted_departure_time=None,
        schedule_relationship=StopScheduleRelationship.SKIPPED,
    )
    assert skipped.schedule_relationship == StopScheduleRelationship.SKIPPED


def test_to_stop_time_update_matches_gtfs_rt_json_shape():
    prediction = _stop_time_prediction()
    stu = prediction.to_stop_time_update()
    assert stu["stop_sequence"] == 3
    assert stu["stop_id"] == "s1"
    assert stu["arrival"] == {
        "time": int(NOW.timestamp()),
        "delay": 45,
        "uncertainty": 30,
    }
    assert "schedule_relationship" not in stu  # SCHEDULED is the implicit default


def test_to_stop_time_update_includes_schedule_relationship_when_not_scheduled():
    prediction = _stop_time_prediction(
        predicted_arrival_time=None,
        predicted_departure_time=None,
        schedule_relationship=StopScheduleRelationship.SKIPPED,
    )
    stu = prediction.to_stop_time_update()
    assert stu["schedule_relationship"] == "SKIPPED"
    assert "arrival" not in stu


def _trip_update(**overrides) -> TripUpdatePrediction:
    base = dict(
        trip_id="t1",
        route_id="r1",
        start_date="20260917",
        start_time="08:00:00",
        vehicle_id="v1",
        stop_time_updates=[_stop_time_prediction()],
        generated_at=NOW,
    )
    base.update(overrides)
    return TripUpdatePrediction(**base)


def test_stop_time_updates_must_belong_to_the_same_trip():
    mismatched = _stop_time_prediction(trip_id="other-trip")
    with pytest.raises(ValidationError):
        _trip_update(stop_time_updates=[mismatched])


def test_to_trip_update_dict_matches_gtfs_rt_json_shape():
    tu = _trip_update()
    payload = tu.to_trip_update_dict()
    assert payload["trip"] == {
        "trip_id": "t1",
        "start_date": "20260917",
        "route_id": "r1",
        "start_time": "08:00:00",
    }
    assert payload["vehicle"] == {"id": "v1"}
    assert payload["timestamp"] == int(NOW.timestamp())
    assert len(payload["stop_time_update"]) == 1
    assert payload["stop_time_update"][0]["stop_id"] == "s1"


def test_trip_update_schedule_relationship_included_when_not_scheduled():
    tu = _trip_update(schedule_relationship=TripScheduleRelationship.CANCELED)
    payload = tu.to_trip_update_dict()
    assert payload["trip"]["schedule_relationship"] == "CANCELED"


def test_requires_at_least_one_stop_time_update():
    with pytest.raises(ValidationError):
        _trip_update(stop_time_updates=[])
