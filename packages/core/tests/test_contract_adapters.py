"""Tests for contracts.adapters: record-archive-row -> Observation/Arrival,
domain-model -> contract mappings."""

from __future__ import annotations

import pandas as pd

from stoptimes.core.contracts.adapters import (
    arrival_from_actual_arrival,
    observation_from_tu_row,
    observation_from_vp_row,
    source_from_archive_uri,
    source_from_val_yaml,
    stop_time_prediction_from_sample,
)
from stoptimes.core.contracts.data import ArrivalMethod, ObservationKind
from stoptimes.core.contracts.source import ArchiveKind
from stoptimes.core.domain.models import ActualArrival, PredictionSample


def test_observation_from_vp_row_mirrors_record_schema_columns():
    row = {
        "feed_name": "mbta",
        "vehicle_id": "v1",
        "ts": "2026-09-17T12:00:00Z",
        "lat": 42.1,
        "lon": -71.1,
        "bearing": 90.0,
        "speed": 12.5,
        "trip_id": "t1",
        "route_id": "r1",
        "current_stop_sequence": 4,
        "current_status": "IN_TRANSIT_TO",
        "stop_id": "s1",
        "ingested_at": "2026-09-17T12:00:05Z",
    }
    obs = observation_from_vp_row(row, source_id="mbta")
    assert obs.kind == ObservationKind.VEHICLE_POSITION
    assert obs.lat == 42.1
    assert obs.ts == pd.Timestamp("2026-09-17T12:00:00", tz="UTC")
    assert obs.current_stop_sequence == 4


def test_observation_from_tu_row_mirrors_record_tu_schema_columns():
    row = {
        "feed_name": "mbta",
        "route_id": "r1",
        "trip_id": "t1",
        "start_date": "20260917",
        "direction_id": 0,
        "vehicle_id": "v1",
        "trip_schedule_relationship": "SCHEDULED",
        "revenue": True,
        "stop_id": "s1",
        "stop_sequence": 4,
        "arrival_time": "2026-09-17T12:05:00Z",
        "arrival_delay": 30,
        "arrival_uncertainty": 15,
        "departure_time": None,
        "departure_delay": None,
        "departure_uncertainty": None,
        "stu_schedule_relationship": "SCHEDULED",
        "ts": "2026-09-17T12:00:00Z",
        "ingested_at": "2026-09-17T12:00:05Z",
    }
    obs = observation_from_tu_row(row, source_id="mbta")
    assert obs.kind == ObservationKind.TRIP_UPDATE
    assert obs.stop_sequence == 4
    assert obs.arrival_delay == 30
    assert obs.departure_time is None
    assert obs.schedule_relationship == "SCHEDULED"


def test_arrival_from_actual_arrival():
    actual = ActualArrival(
        trip_key="mbta:t1:v1:20260917",
        stop_id="s1",
        stop_sequence=4,
        actual_arrival_epoch=1789646400.0,
        method="map_matched",
    )
    arrival = arrival_from_actual_arrival(
        actual, source_id="mbta", start_date="20260917", route_id="r1", vehicle_id="v1"
    )
    assert arrival.method == ArrivalMethod.MAP_MATCHED
    assert arrival.trip_id == "mbta:t1:v1:20260917"
    assert arrival.arrival_time == pd.Timestamp(1789646400.0, unit="s", tz="UTC")


def test_arrival_from_actual_arrival_falls_back_on_unknown_method():
    actual = ActualArrival(
        trip_key="mbta:t1:v1:20260917",
        stop_id="s1",
        stop_sequence=4,
        actual_arrival_epoch=1789646400.0,
        method="some_future_method",
    )
    arrival = arrival_from_actual_arrival(actual, source_id="mbta", start_date="20260917")
    assert arrival.method == ArrivalMethod.MAP_MATCHED


def test_stop_time_prediction_from_sample():
    sample = PredictionSample(
        trip_key="mbta:t1:v1:20260917",
        predictor_id="xgboost:v1",
        stop_id="s1",
        stop_sequence=4,
        predicted_arrival_epoch=1789646400.0,
        made_at_epoch=1789646100.0,
    )
    prediction = stop_time_prediction_from_sample(
        sample, trip_id="t1", start_date="20260917", route_id="r1", vehicle_id="v1"
    )
    assert prediction.model_id == "xgboost:v1"
    assert prediction.stop_id == "s1"
    assert prediction.predicted_arrival_time.timestamp() == 1789646400.0


def test_source_from_archive_uri_detects_s3_vs_local():
    source = source_from_val_yaml(
        {
            "name": "mbta",
            "gtfs_static_url": "https://cdn.mbta.com/MBTA_GTFS.zip",
            "vehicle_positions_url": "https://x/vp.json",
            "trip_updates_url": "https://x/tu.json",
            "poll_interval_s": 15,
        }
    )
    assert source.archive is None

    local = source_from_archive_uri(source, "/var/data/mbta")
    assert local.archive.kind == ArchiveKind.LOCAL
    # immutable: original untouched
    assert source.archive is None

    s3 = source_from_archive_uri(source, "s3://transit/feeds/mbta")
    assert s3.archive.kind == ArchiveKind.S3
