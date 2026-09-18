"""test_feeds_pb — GTFS-RT protobuf feed parsing + pollers (T6 deliverable 1).

Builds ``FeedMessage`` protobuf objects in-memory (no fixture files, no
network) and checks they parse to the exact same record shape the JSON
reader produces, then exercises the async pollers offline via respx.
"""

from __future__ import annotations

import httpx
import pytest
import respx
from google.transit import gtfs_realtime_pb2

from stoptimes.val.feeds.pb import (
    AutoTripUpdatesPoller,
    AutoVehiclePositionsPoller,
    TripUpdatesPbPoller,
    VehiclePositionsPbPoller,
    parse_tu_feed_pb,
    parse_vp_feed_pb,
    sniff_format,
)

# ── Builders ───────────────────────────────────────────────────────────────────


def _build_vp_feed_message() -> bytes:
    msg = gtfs_realtime_pb2.FeedMessage()
    msg.header.gtfs_realtime_version = "2.0"
    msg.header.timestamp = 1700000000

    entity = msg.entity.add()
    entity.id = "1"
    entity.vehicle.trip.trip_id = "trip_1"
    entity.vehicle.trip.route_id = "route_1"
    entity.vehicle.trip.direction_id = 0
    entity.vehicle.trip.start_date = "20240101"
    entity.vehicle.vehicle.id = "veh_1"
    entity.vehicle.position.latitude = 42.35
    entity.vehicle.position.longitude = -71.05
    entity.vehicle.position.speed = 12.5
    entity.vehicle.current_status = gtfs_realtime_pb2.VehiclePosition.IN_TRANSIT_TO
    entity.vehicle.current_stop_sequence = 3
    entity.vehicle.timestamp = 1700000001

    return msg.SerializeToString()


def _build_tu_feed_message() -> bytes:
    msg = gtfs_realtime_pb2.FeedMessage()
    msg.header.gtfs_realtime_version = "2.0"
    msg.header.timestamp = 1700000000

    entity = msg.entity.add()
    entity.id = "1"
    entity.trip_update.trip.trip_id = "trip_1"
    entity.trip_update.trip.route_id = "route_1"
    entity.trip_update.trip.direction_id = 0
    entity.trip_update.trip.start_date = "20240101"

    stu = entity.trip_update.stop_time_update.add()
    stu.stop_sequence = 1
    stu.stop_id = "stop_1"
    stu.arrival.time = 1700000300

    return msg.SerializeToString()


# ── Pure parsing ────────────────────────────────────────────────────────────────


def test_parse_vp_feed_pb_matches_json_shape() -> None:
    header_ts, records = parse_vp_feed_pb(_build_vp_feed_message())

    assert header_ts == pytest.approx(1700000000.0)
    assert len(records) == 1
    record = records[0]
    assert record.trip_id == "trip_1"
    assert record.route_id == "route_1"
    assert record.direction_id == 0
    assert record.start_date == "20240101"
    assert record.vehicle_id == "veh_1"
    assert record.state.lat == pytest.approx(42.35)
    assert record.state.lon == pytest.approx(-71.05)
    assert record.state.speed == pytest.approx(12.5)
    assert record.state.current_stop_sequence == 3


def test_parse_tu_feed_pb_matches_json_shape() -> None:
    header_ts, records = parse_tu_feed_pb(_build_tu_feed_message())

    assert header_ts == pytest.approx(1700000000.0)
    assert len(records) == 1
    record = records[0]
    assert record.trip_id == "trip_1"
    assert record.route_id == "route_1"
    assert len(record.stop_time_updates) == 1
    stu = record.stop_time_updates[0]
    assert stu.stop_id == "stop_1"
    assert stu.stop_sequence == 1
    assert stu.arrival_time == pytest.approx(1700000300.0)


def test_parse_vp_feed_pb_rejects_garbage_bytes() -> None:
    from google.protobuf.message import DecodeError

    with pytest.raises(DecodeError):
        parse_vp_feed_pb(b"not a protobuf feed message, definitely not")


# ── Format sniffing ──────────────────────────────────────────────────────────────


def test_sniff_format_from_json_content_type() -> None:
    assert sniff_format("application/json; charset=utf-8", b"whatever") == "json"


def test_sniff_format_from_protobuf_content_type() -> None:
    assert sniff_format("application/x-protobuf", b"whatever") == "pb"
    assert sniff_format("application/octet-stream", b"whatever") == "pb"


def test_sniff_format_falls_back_to_body_json() -> None:
    assert sniff_format(None, b'  {"header": {}}') == "json"


def test_sniff_format_falls_back_to_body_pb() -> None:
    assert sniff_format(None, _build_vp_feed_message()) == "pb"


def test_sniff_format_falls_back_when_content_type_unrecognized() -> None:
    assert sniff_format("text/plain", _build_vp_feed_message()) == "pb"
    assert sniff_format("text/plain", b'{"a": 1}') == "json"


# ── Async pollers (respx mock; no network) ────────────────────────────────────────


async def test_vp_pb_poller_returns_parsed_result() -> None:
    body = _build_vp_feed_message()
    with respx.mock:
        respx.get("http://test.example.com/vp.pb").mock(
            return_value=httpx.Response(
                200, content=body, headers={"content-type": "application/x-protobuf"}
            )
        )
        poller = VehiclePositionsPbPoller("http://test.example.com/vp.pb")
        result = await poller.poll()

    assert result.header_timestamp == pytest.approx(1700000000.0)
    assert result.records[0].trip_id == "trip_1"


async def test_tu_pb_poller_returns_parsed_result() -> None:
    body = _build_tu_feed_message()
    with respx.mock:
        respx.get("http://test.example.com/tu.pb").mock(
            return_value=httpx.Response(
                200, content=body, headers={"content-type": "application/x-protobuf"}
            )
        )
        poller = TripUpdatesPbPoller("http://test.example.com/tu.pb")
        result = await poller.poll()

    assert result.header_timestamp == pytest.approx(1700000000.0)
    assert result.records[0].stop_time_updates[0].arrival_time == pytest.approx(1700000300.0)


async def test_auto_vp_poller_sniffs_protobuf_by_content_type() -> None:
    body = _build_vp_feed_message()
    with respx.mock:
        respx.get("http://test.example.com/vp").mock(
            return_value=httpx.Response(
                200, content=body, headers={"content-type": "application/x-protobuf"}
            )
        )
        poller = AutoVehiclePositionsPoller("http://test.example.com/vp")
        result = await poller.poll()

    assert result.records[0].trip_id == "trip_1"


async def test_auto_vp_poller_sniffs_json_by_content_type() -> None:
    feed_json = {
        "header": {"timestamp": 1700000000},
        "entity": [
            {
                "id": "1",
                "vehicle": {
                    "trip": {"trip_id": "trip_json"},
                    "vehicle": {"id": "veh_json"},
                    "position": {"latitude": 1.0, "longitude": 2.0},
                },
            }
        ],
    }
    with respx.mock:
        respx.get("http://test.example.com/vp").mock(return_value=httpx.Response(200, json=feed_json))
        poller = AutoVehiclePositionsPoller("http://test.example.com/vp")
        result = await poller.poll()

    assert result.records[0].trip_id == "trip_json"


async def test_auto_tu_poller_respects_explicit_format_override() -> None:
    body = _build_tu_feed_message()
    with respx.mock:
        # No content-type header at all -- explicit format="pb" must still work.
        respx.get("http://test.example.com/tu").mock(return_value=httpx.Response(200, content=body))
        poller = AutoTripUpdatesPoller("http://test.example.com/tu", format="pb")
        result = await poller.poll()

    assert result.records[0].trip_id == "trip_1"
