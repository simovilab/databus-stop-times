"""Tests for contracts.data: schemas, validate_frame, UTC coercion, Observation/Arrival."""

from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from stoptimes.core.contracts.data import (
    ARRIVAL_SCHEMA,
    OBSERVATION_TU_SCHEMA,
    OBSERVATION_VP_SCHEMA,
    REQUIRED_STATIC_TABLES,
    UTC_DTYPE,
    Arrival,
    ArrivalMethod,
    Observation,
    ObservationKind,
    StaticFeed,
    coerce_utc_ns,
    validate_frame,
)


def test_coerce_utc_ns_pins_dtype_and_treats_naive_as_utc():
    series = pd.Series(["2026-01-01T00:00:00", "2026-01-01T00:01:00"])
    out = coerce_utc_ns(series)
    assert str(out.dtype) == UTC_DTYPE
    assert out.iloc[0] == pd.Timestamp("2026-01-01T00:00:00", tz="UTC")


def test_coerce_utc_ns_is_idempotent_on_already_tz_aware():
    series = pd.Series([pd.Timestamp("2026-01-01", tz="UTC")])
    out = coerce_utc_ns(coerce_utc_ns(series))
    assert str(out.dtype) == UTC_DTYPE


def test_validate_frame_reports_missing_required_column():
    df = pd.DataFrame({"lat": [1.0], "lon": [2.0], "trip_id": ["t1"]})
    problems = validate_frame(df, OBSERVATION_VP_SCHEMA)
    assert any("ts" in p for p in problems)


def test_validate_frame_flags_wrong_timestamp_resolution():
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2026-01-01"], utc=True).astype("datetime64[us, UTC]"),
            "lat": [1.0],
            "lon": [2.0],
            "trip_id": ["t1"],
        }
    )
    problems = validate_frame(df, OBSERVATION_VP_SCHEMA)
    assert any("ts" in p and "coerce_utc_ns" in p for p in problems)


def test_validate_frame_passes_on_correct_frame():
    df = pd.DataFrame(
        {
            "ts": coerce_utc_ns(pd.Series(["2026-01-01T00:00:00Z"])),
            "lat": [42.0],
            "lon": [-71.0],
            "trip_id": ["t1"],
        }
    )
    assert validate_frame(df, OBSERVATION_VP_SCHEMA) == []


def test_static_feed_reports_missing_required_tables():
    feed = StaticFeed(source_id="mbta", tables=["stops", "trips"])
    assert feed.missing_required() == ["stop_times"]
    assert set(REQUIRED_STATIC_TABLES) == {"stops", "trips", "stop_times"}


def test_observation_vehicle_position_requires_lat_lon():
    with pytest.raises(ValidationError):
        Observation(kind=ObservationKind.VEHICLE_POSITION, source_id="mbta", trip_id="t1")

    obs = Observation(
        kind=ObservationKind.VEHICLE_POSITION,
        source_id="mbta",
        trip_id="t1",
        lat=42.0,
        lon=-71.0,
        ts=pd.Timestamp("2026-01-01", tz="UTC"),
    )
    assert obs.lat == 42.0


def test_observation_trip_update_requires_stop_id_and_sequence():
    with pytest.raises(ValidationError):
        Observation(kind=ObservationKind.TRIP_UPDATE, source_id="mbta", trip_id="t1")

    obs = Observation(
        kind=ObservationKind.TRIP_UPDATE,
        source_id="mbta",
        trip_id="t1",
        stop_id="s1",
        stop_sequence=1,
        ts=pd.Timestamp("2026-01-01", tz="UTC"),
    )
    assert obs.stop_sequence == 1


def test_observation_vp_columns_match_record_schema_shape():
    # Column names must mirror stoptimes.record.storage.schema.DATA_COLUMNS,
    # without importing that module (core must not depend on record).
    expected = {
        "feed_name",
        "vehicle_id",
        "ts",
        "lat",
        "lon",
        "bearing",
        "speed",
        "trip_id",
        "current_stop_sequence",
        "current_status",
        "stop_id",
        "ingested_at",
    }
    assert expected <= set(OBSERVATION_VP_SCHEMA.column_names)


def test_observation_tu_columns_match_record_tu_schema_shape():
    expected = {
        "feed_name",
        "route_id",
        "trip_id",
        "start_date",
        "direction_id",
        "vehicle_id",
        "trip_schedule_relationship",
        "revenue",
        "stop_id",
        "stop_sequence",
        "arrival_time",
        "arrival_delay",
        "arrival_uncertainty",
        "departure_time",
        "departure_delay",
        "departure_uncertainty",
        "ts",
        "ingested_at",
    }
    assert expected <= set(OBSERVATION_TU_SCHEMA.column_names)


def test_arrival_round_trip_and_schema():
    arrival = Arrival(
        source_id="mbta",
        trip_id="t1",
        start_date="20260917",
        stop_id="s1",
        stop_sequence=4,
        arrival_time=pd.Timestamp("2026-09-17T12:00:00", tz="UTC"),
        method=ArrivalMethod.MAP_MATCHED,
    )
    assert arrival.method == ArrivalMethod.MAP_MATCHED
    assert set(ARRIVAL_SCHEMA.required_columns) <= {
        "source_id",
        "trip_id",
        "start_date",
        "stop_id",
        "stop_sequence",
        "arrival_time",
        "method",
    }
