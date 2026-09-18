"""Tests for readers.read_gtfsrt_json / read_gtfsrt_protobuf."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from stoptimes.core.contracts.data import (
    OBSERVATION_TU_SCHEMA,
    OBSERVATION_VP_SCHEMA,
    ObservationKind,
    validate_frame,
)
from stoptimes.core.readers import read_gtfsrt_json, read_gtfsrt_protobuf

FIXTURES = Path(__file__).parent / "fixtures"
VP_JSON = FIXTURES / "gtfsrt_vp.json"
TU_JSON = FIXTURES / "gtfsrt_tu.json"


# --- JSON: vehicle positions -------------------------------------------------


def test_read_vp_json_happy_path():
    df = read_gtfsrt_json(str(VP_JSON), ObservationKind.VEHICLE_POSITION, source_id="mbta")
    assert validate_frame(df, OBSERVATION_VP_SCHEMA) == []
    assert len(df) == 2
    assert set(df["trip_id"]) == {"t1", "t2"}
    row = df[df["trip_id"] == "t1"].iloc[0]
    assert row["route_id"] == "r1"
    assert row["vehicle_id"] == "v1"
    assert row["bearing"] == 45.0
    assert row["current_stop_sequence"] == 3
    assert row["ts"] == pd.Timestamp(1700000010, unit="s", tz="UTC")
    assert (df["source_id"] == "mbta").all()
    assert (df["kind"] == "vehicle_position").all()


def test_read_vp_json_falls_back_to_header_timestamp():
    df = read_gtfsrt_json(str(VP_JSON), ObservationKind.VEHICLE_POSITION, source_id="mbta")
    row = df[df["trip_id"] == "t2"].iloc[0]
    assert row["ts"] == pd.Timestamp(1700000020, unit="s", tz="UTC")


def test_read_vp_json_skips_entity_with_no_vehicle():
    df = read_gtfsrt_json(str(VP_JSON), ObservationKind.VEHICLE_POSITION, source_id="mbta")
    # entity id "3" is an alert, not a vehicle — must not appear at all.
    assert len(df) == 2


def test_read_vp_json_empty_feed(tmp_path):
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"header": {"timestamp": 1700000000}, "entity": []}))
    df = read_gtfsrt_json(str(empty), ObservationKind.VEHICLE_POSITION, source_id="mbta")
    assert len(df) == 0
    assert validate_frame(df, OBSERVATION_VP_SCHEMA) == []
    expected_cols = OBSERVATION_VP_SCHEMA.column_names
    assert list(df.columns[: len(expected_cols)]) == expected_cols


def test_read_vp_json_entity_missing_lat_lon_is_skipped(tmp_path):
    feed = {
        "header": {"timestamp": 1700000000},
        "entity": [{"id": "1", "vehicle": {"trip": {"trip_id": "t1"}, "position": {}}}],
    }
    path = tmp_path / "nolatlon.json"
    path.write_text(json.dumps(feed))
    df = read_gtfsrt_json(str(path), ObservationKind.VEHICLE_POSITION, source_id="mbta")
    assert len(df) == 0


# --- JSON: trip updates -------------------------------------------------------


def test_read_tu_json_happy_path():
    df = read_gtfsrt_json(str(TU_JSON), ObservationKind.TRIP_UPDATE, source_id="mbta")
    assert validate_frame(df, OBSERVATION_TU_SCHEMA) == []
    assert len(df) == 2  # t2's trip_update has an empty stop_time_update list -> 0 rows
    assert set(df["trip_id"]) == {"t1"}
    first = df[df["stop_sequence"] == 3].iloc[0]
    assert first["stop_id"] == "s1"
    assert first["arrival_delay"] == 45
    assert first["arrival_time"] == pd.Timestamp(1700000300, unit="s", tz="UTC")
    assert first["schedule_relationship_norm"] == "SCHEDULED"
    second = df[df["stop_sequence"] == 4].iloc[0]
    assert pd.isna(second["departure_time"])


def test_read_tu_json_empty_stop_time_update_contributes_no_rows():
    df = read_gtfsrt_json(str(TU_JSON), ObservationKind.TRIP_UPDATE, source_id="mbta")
    assert "t2" not in set(df["trip_id"])


def test_read_tu_json_empty_feed(tmp_path):
    empty = tmp_path / "empty_tu.json"
    empty.write_text(json.dumps({"header": {"timestamp": 1700000000}, "entity": []}))
    df = read_gtfsrt_json(str(empty), ObservationKind.TRIP_UPDATE, source_id="mbta")
    assert len(df) == 0
    assert validate_frame(df, OBSERVATION_TU_SCHEMA) == []


# --- protobuf ------------------------------------------------------------------

pb2 = pytest.importorskip("google.transit.gtfs_realtime_pb2")


def _build_vp_message():
    feed = pb2.FeedMessage()
    feed.header.gtfs_realtime_version = "2.0"
    feed.header.timestamp = 1700000000

    entity = feed.entity.add()
    entity.id = "1"
    entity.vehicle.trip.trip_id = "t1"
    entity.vehicle.trip.route_id = "r1"
    entity.vehicle.vehicle.id = "v1"
    entity.vehicle.position.latitude = 9.93
    entity.vehicle.position.longitude = -84.08
    entity.vehicle.position.bearing = 45.0
    entity.vehicle.timestamp = 1700000010
    entity.vehicle.current_stop_sequence = 3
    entity.vehicle.current_status = pb2.VehiclePosition.IN_TRANSIT_TO

    # An entity with no `vehicle` field set at all (an alert-only entity).
    no_vehicle = feed.entity.add()
    no_vehicle.id = "2"
    no_vehicle.alert.cause = pb2.Alert.OTHER_CAUSE

    return feed.SerializeToString()


def _build_tu_message():
    feed = pb2.FeedMessage()
    feed.header.gtfs_realtime_version = "2.0"
    feed.header.timestamp = 1700000000

    entity = feed.entity.add()
    entity.id = "1"
    entity.trip_update.trip.trip_id = "t1"
    entity.trip_update.trip.route_id = "r1"
    entity.trip_update.timestamp = 1700000010
    stu = entity.trip_update.stop_time_update.add()
    stu.stop_sequence = 3
    stu.stop_id = "s1"
    stu.arrival.time = 1700000300
    stu.arrival.delay = 45
    stu.schedule_relationship = pb2.TripUpdate.StopTimeUpdate.SCHEDULED

    return feed.SerializeToString()


def test_read_vp_protobuf_happy_path_and_no_vehicle_entity(tmp_path):
    path = tmp_path / "vp.pb"
    path.write_bytes(_build_vp_message())
    df = read_gtfsrt_protobuf(str(path), ObservationKind.VEHICLE_POSITION, source_id="mbta")
    assert validate_frame(df, OBSERVATION_VP_SCHEMA) == []
    assert len(df) == 1  # the alert-only entity is skipped
    row = df.iloc[0]
    assert row["trip_id"] == "t1"
    assert row["bearing"] == 45.0
    assert row["current_status"] == "IN_TRANSIT_TO"


def test_read_tu_protobuf_happy_path(tmp_path):
    path = tmp_path / "tu.pb"
    path.write_bytes(_build_tu_message())
    df = read_gtfsrt_protobuf(str(path), ObservationKind.TRIP_UPDATE, source_id="mbta")
    assert validate_frame(df, OBSERVATION_TU_SCHEMA) == []
    assert len(df) == 1
    row = df.iloc[0]
    assert row["trip_id"] == "t1"
    assert row["arrival_delay"] == 45
    assert row["schedule_relationship_norm"] == "SCHEDULED"


def test_read_protobuf_bad_bytes_raises(tmp_path):
    path = tmp_path / "garbage.pb"
    path.write_bytes(b"not a protobuf feed message")
    with pytest.raises(ValueError):
        read_gtfsrt_protobuf(str(path), ObservationKind.VEHICLE_POSITION, source_id="mbta")


def test_read_gtfsrt_protobuf_missing_extra_gives_clear_error(monkeypatch, tmp_path):
    import sys

    import stoptimes.core.readers as readers_module

    monkeypatch.setitem(sys.modules, "google.transit", None)
    monkeypatch.setitem(sys.modules, "google.transit.gtfs_realtime_pb2", None)
    path = tmp_path / "whatever.pb"
    path.write_bytes(b"")
    with pytest.raises(ImportError, match=r"\[pb\]"):
        readers_module.read_gtfsrt_protobuf(
            str(path), ObservationKind.VEHICLE_POSITION, source_id="mbta"
        )
