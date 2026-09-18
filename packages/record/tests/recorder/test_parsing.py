"""Tests for stoptimes.record.parsing: pb/json -> archive rows, malformed input."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from google.transit import gtfs_realtime_pb2
from stoptimes.core.contracts.source import FeedFormat
from stoptimes.record import parsing


def test_malformed_protobuf_raises_feed_parse_error():
    with pytest.raises(parsing.FeedParseError):
        parsing.parse_payload(b"\xff\xff\xff not a real feed", FeedFormat.PROTOBUF)


def test_malformed_json_raises_feed_parse_error():
    with pytest.raises(parsing.FeedParseError):
        parsing.parse_payload(b"{not json", FeedFormat.JSON)


def test_json_non_object_raises_feed_parse_error():
    with pytest.raises(parsing.FeedParseError):
        parsing.parse_payload(b"[1, 2, 3]", FeedFormat.JSON)


def test_vp_rows_from_json(vp_feed_dict):
    feed = parsing.parse_payload(json.dumps(vp_feed_dict).encode(), FeedFormat.JSON)
    ingested = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = parsing.vp_rows(feed, FeedFormat.JSON, feed_name="testagency", ingested_at=ingested)

    assert len(rows) == 1
    row = rows[0]
    assert row["vehicle_id"] == "v1"
    assert row["route_id"] == "r1"
    assert row["trip_id"] == "t1"
    assert row["lat"] == 1.0
    assert row["lon"] == 2.0
    assert row["current_status"] == "IN_TRANSIT_TO"
    assert row["current_stop_sequence"] == 3
    assert row["ingested_at"] == ingested


def _pb_vp_feed() -> bytes:
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.header.gtfs_realtime_version = "2.0"
    feed.header.timestamp = 1000
    entity = feed.entity.add()
    entity.id = "1"
    entity.vehicle.trip.trip_id = "t1"
    entity.vehicle.trip.route_id = "r1"
    entity.vehicle.vehicle.id = "v1"
    entity.vehicle.position.latitude = 1.0
    entity.vehicle.position.longitude = 2.0
    entity.vehicle.position.bearing = 90.0
    entity.vehicle.current_stop_sequence = 3
    entity.vehicle.current_status = gtfs_realtime_pb2.VehiclePosition.IN_TRANSIT_TO
    entity.vehicle.timestamp = 1000
    return feed.SerializeToString()


def test_vp_rows_from_protobuf_matches_json_shape(vp_feed_dict):
    ingested = datetime(2026, 1, 1, tzinfo=timezone.utc)
    pb_feed = parsing.parse_payload(_pb_vp_feed(), FeedFormat.PROTOBUF)
    pb_rows = parsing.vp_rows(pb_feed, FeedFormat.PROTOBUF, feed_name="testagency", ingested_at=ingested)

    json_feed = parsing.parse_payload(json.dumps(vp_feed_dict).encode(), FeedFormat.JSON)
    json_rows = parsing.vp_rows(json_feed, FeedFormat.JSON, feed_name="testagency", ingested_at=ingested)

    assert len(pb_rows) == len(json_rows) == 1
    for key in ("vehicle_id", "route_id", "trip_id", "lat", "lon", "current_status", "current_stop_sequence"):
        assert pb_rows[0][key] == json_rows[0][key], key


def _tu_json_feed() -> dict:
    return {
        "header": {"timestamp": 2000},
        "entity": [
            {
                "id": "e1",
                "trip_update": {
                    "trip": {"trip_id": "t1", "route_id": "r1", "start_date": "20260101"},
                    "timestamp": 2000,
                    "stop_time_update": [
                        {
                            "stop_id": "s1",
                            "stop_sequence": 5,
                            "arrival": {"time": 2060, "delay": 30, "uncertainty": 15},
                        }
                    ],
                },
            }
        ],
    }


def test_tu_rows_from_json():
    feed = _tu_json_feed()
    ingested = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = parsing.tu_rows(feed, FeedFormat.JSON, feed_name="testagency", ingested_at=ingested)

    assert len(rows) == 1
    row = rows[0]
    assert row["trip_id"] == "t1"
    assert row["stop_id"] == "s1"
    assert row["arrival_delay"] == 30
    assert row["arrival_uncertainty"] == 15


def test_tu_rows_from_json_route_scoping():
    feed = _tu_json_feed()
    ingested = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = parsing.tu_rows(
        feed, FeedFormat.JSON, feed_name="testagency", ingested_at=ingested, routes=frozenset({"other"})
    )
    assert rows == []


def test_tu_rows_from_json_horizon_cap():
    feed = _tu_json_feed()
    ingested = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = parsing.tu_rows(
        feed, FeedFormat.JSON, feed_name="testagency", ingested_at=ingested, max_horizon_seconds=10
    )
    assert rows == []  # predicted arrival is 60s beyond ts, cap is 10s
