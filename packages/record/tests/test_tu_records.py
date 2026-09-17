"""Tests for GTFS-RT TripUpdates -> archive records (rt_pipeline.tu_records).

Django-free: these build FeedMessage objects in memory, so they exercise the
scoping and field-mapping decisions without a settings module, a database, or
the live MBTA feed.
"""

from __future__ import annotations

import datetime as dt

import pytest
from google.transit import gtfs_realtime_pb2

from rt_pipeline.tu_records import feed_records, parse_routes, trip_update_records

INGESTED = dt.datetime(2026, 9, 4, 12, 0, 30, tzinfo=dt.timezone.utc)
HEADER_TS = dt.datetime(2026, 9, 4, 12, 0, 0, tzinfo=dt.timezone.utc)


def _entity(
    *,
    trip_id: str = "trip-1",
    route_id: str = "15",
    ts: int | None = 1788523200,  # 2026-09-04T12:00:00Z
    stops: list[dict] | None = None,
) -> gtfs_realtime_pb2.FeedEntity:
    ent = gtfs_realtime_pb2.FeedEntity()
    ent.id = f"e-{trip_id}"
    tu = ent.trip_update
    tu.trip.trip_id = trip_id
    tu.trip.route_id = route_id
    tu.trip.start_date = "20260904"
    tu.trip.direction_id = 1
    tu.trip.schedule_relationship = gtfs_realtime_pb2.TripDescriptor.SCHEDULED
    tu.vehicle.id = "y1234"
    if ts is not None:
        tu.timestamp = ts
    for spec in stops or [{"stop_sequence": 3, "stop_id": "s3", "arrival": 1788523500}]:
        stu = tu.stop_time_update.add()
        stu.stop_sequence = spec["stop_sequence"]
        stu.stop_id = spec["stop_id"]
        if spec.get("arrival") is not None:
            stu.arrival.time = spec["arrival"]
            if spec.get("uncertainty") is not None:
                stu.arrival.uncertainty = spec["uncertainty"]
            if spec.get("delay") is not None:
                stu.arrival.delay = spec["delay"]
        if spec.get("skipped"):
            stu.schedule_relationship = (
                gtfs_realtime_pb2.TripUpdate.StopTimeUpdate.SKIPPED
            )
    return ent


def _records(entity, **kwargs):
    params = dict(
        feed_name="mbta", header_ts=HEADER_TS, ingested_at=INGESTED,
    )
    params.update(kwargs)
    return trip_update_records(entity, **params)


# --- route scoping ---------------------------------------------------------


def test_route_outside_scope_yields_nothing():
    assert _records(_entity(route_id="Red"), routes=frozenset({"15"})) == []


def test_route_in_scope_is_kept():
    rows = _records(_entity(route_id="15"), routes=frozenset({"222", "15", "37"}))
    assert len(rows) == 1
    assert rows[0]["route_id"] == "15"


def test_routes_none_means_every_route():
    assert len(_records(_entity(route_id="Red"), routes=None)) == 1


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("222,15,37", frozenset({"222", "15", "37"})),
        (" 15 , 37 ", frozenset({"15", "37"})),
        ("", None),          # empty config -> every route, never "no routes"
        (None, None),
        (["15"], frozenset({"15"})),
    ],
)
def test_parse_routes(raw, expected):
    assert parse_routes(raw) == expected


# --- horizon cap -----------------------------------------------------------


def test_horizon_cap_drops_far_predictions_only():
    ent = _entity(stops=[
        {"stop_sequence": 1, "stop_id": "near", "arrival": 1788523800},   # +10 min
        {"stop_sequence": 2, "stop_id": "far", "arrival": 1788530400},    # +2 h
    ])
    rows = _records(ent, max_horizon_seconds=3600)
    assert [r["stop_id"] for r in rows] == ["near"]


def test_no_horizon_cap_keeps_everything():
    ent = _entity(stops=[
        {"stop_sequence": 1, "stop_id": "near", "arrival": 1788523800},
        {"stop_sequence": 2, "stop_id": "far", "arrival": 1788530400},
    ])
    assert len(_records(ent, max_horizon_seconds=None)) == 2


def test_stop_with_no_predicted_time_survives_the_cap():
    # A SKIPPED stop carries no time and is still information; the cap must
    # not silently swallow it.
    ent = _entity(stops=[
        {"stop_sequence": 4, "stop_id": "skipped", "arrival": None, "skipped": True},
    ])
    rows = _records(ent, max_horizon_seconds=3600)
    assert len(rows) == 1
    assert rows[0]["stu_schedule_relationship"] == "SKIPPED"
    assert rows[0]["arrival_time"] is None


# --- field mapping ---------------------------------------------------------


def test_uncertainty_is_preserved():
    ent = _entity(stops=[
        {"stop_sequence": 1, "stop_id": "s1", "arrival": 1788523500,
         "uncertainty": 300, "delay": -45},
    ])
    row = _records(ent)[0]
    assert row["arrival_uncertainty"] == 300
    assert row["arrival_delay"] == -45


def test_schedule_relationship_is_stored_as_a_name_not_an_int():
    row = _records(_entity())[0]
    assert row["trip_schedule_relationship"] == "SCHEDULED"


def test_ts_falls_back_to_header_then_ingest():
    row = _records(_entity(ts=None))[0]
    assert row["ts"] == HEADER_TS
    row = _records(_entity(ts=None), header_ts=None)[0]
    assert row["ts"] == INGESTED


def test_trip_update_timestamp_wins_over_header():
    row = _records(_entity(ts=1788523260))[0]
    assert row["ts"] == dt.datetime(2026, 9, 4, 12, 1, tzinfo=dt.timezone.utc)


def test_entity_without_trip_id_is_dropped():
    ent = _entity()
    ent.trip_update.trip.ClearField("trip_id")
    assert _records(ent) == []


def test_non_trip_update_entity_is_ignored():
    ent = gtfs_realtime_pb2.FeedEntity()
    ent.id = "vehicle-only"
    ent.vehicle.vehicle.id = "y1"
    assert _records(ent) == []


def test_feed_records_uses_the_feed_header_timestamp():
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.header.gtfs_realtime_version = "2.0"
    feed.header.timestamp = 1788523200
    feed.entity.append(_entity(ts=None))
    rows = feed_records(feed, feed_name="mbta", ingested_at=INGESTED)
    assert len(rows) == 1
    assert rows[0]["ts"] == HEADER_TS
    assert rows[0]["header_ts"] == HEADER_TS
