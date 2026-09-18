"""GTFS-RT TripUpdates encoding: spec shape + JSON/protobuf round-trips."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from stoptimes.core.contracts.prediction import StopTimePrediction, TripUpdatePrediction
from stoptimes.core.realtime.gtfsrt_json import parse_tu_feed
from stoptimes.serve.tripupdates import encode_json, encode_protobuf, feed_message_dict

GENERATED_AT = datetime(2026, 9, 17, 12, 0, 0, tzinfo=timezone.utc)


def _trip_update() -> TripUpdatePrediction:
    stops = [
        StopTimePrediction(
            trip_id="t1",
            route_id="r1",
            direction_id=0,
            start_date="20260917",
            stop_id="s1",
            stop_sequence=1,
            predicted_arrival_time=datetime(2026, 9, 17, 12, 5, 0, tzinfo=timezone.utc),
            delay_s=45.0,
            uncertainty_s=30.0,
            model_id="fake",
            generated_at=GENERATED_AT,
        ),
        StopTimePrediction(
            trip_id="t1",
            route_id="r1",
            direction_id=0,
            start_date="20260917",
            stop_id="s2",
            stop_sequence=2,
            predicted_arrival_time=datetime(2026, 9, 17, 12, 8, 0, tzinfo=timezone.utc),
            delay_s=60.0,
            uncertainty_s=30.0,
            model_id="fake",
            generated_at=GENERATED_AT,
        ),
    ]
    return TripUpdatePrediction(
        trip_id="t1",
        route_id="r1",
        direction_id=0,
        start_date="20260917",
        start_time="08:00:00",
        vehicle_id="v1",
        stop_time_updates=stops,
        generated_at=GENERATED_AT,
    )


def test_feed_message_dict_shape():
    feed = feed_message_dict([_trip_update()], generated_at=GENERATED_AT)
    assert feed["header"]["gtfs_realtime_version"] == "2.0"
    assert feed["header"]["incrementality"] == "FULL_DATASET"
    assert feed["header"]["timestamp"] == int(GENERATED_AT.timestamp())
    entity = feed["entity"][0]
    trip = entity["trip_update"]["trip"]
    assert trip["trip_id"] == "t1"
    assert trip["start_date"] == "20260917"
    assert trip["start_time"] == "08:00:00"
    stop_updates = entity["trip_update"]["stop_time_update"]
    assert [s["stop_sequence"] for s in stop_updates] == [1, 2]
    assert stop_updates[0]["arrival"]["time"] == int(datetime(2026, 9, 17, 12, 5, 0, tzinfo=timezone.utc).timestamp())
    assert stop_updates[0]["arrival"]["delay"] == 45
    assert stop_updates[0]["arrival"]["uncertainty"] == 30


def test_json_round_trip_through_gtfsrt_json_parser():
    tu = _trip_update()
    raw = encode_json([tu], generated_at=GENERATED_AT)
    import json

    feed = json.loads(raw)
    _header_ts, records = parse_tu_feed(feed)

    assert len(records) == 1
    record = records[0]
    assert record.trip_id == tu.trip_id
    assert record.route_id == tu.route_id
    assert record.direction_id == tu.direction_id
    assert record.start_date == tu.start_date
    assert [u.stop_sequence for u in record.stop_time_updates] == [1, 2]
    assert record.stop_time_updates[0].stop_id == "s1"
    assert record.stop_time_updates[0].arrival_time == pytest.approx(
        tu.stop_time_updates[0].predicted_arrival_time.timestamp()
    )


def test_protobuf_round_trip():
    pytest.importorskip("google.transit.gtfs_realtime_pb2")
    tu = _trip_update()
    raw = encode_protobuf([tu], generated_at=GENERATED_AT)

    from google.transit import gtfs_realtime_pb2

    message = gtfs_realtime_pb2.FeedMessage()
    message.ParseFromString(raw)

    assert message.header.gtfs_realtime_version == "2.0"
    assert message.header.incrementality == gtfs_realtime_pb2.FeedHeader.FULL_DATASET
    assert message.header.timestamp == int(GENERATED_AT.timestamp())

    assert len(message.entity) == 1
    entity = message.entity[0]
    assert entity.trip_update.trip.trip_id == "t1"
    assert entity.trip_update.trip.route_id == "r1"
    assert entity.trip_update.trip.direction_id == 0
    assert entity.trip_update.trip.start_date == "20260917"
    assert entity.trip_update.trip.start_time == "08:00:00"
    assert entity.trip_update.vehicle.id == "v1"

    stop_updates = entity.trip_update.stop_time_update
    assert [s.stop_sequence for s in stop_updates] == [1, 2]
    assert stop_updates[0].stop_id == "s1"
    assert stop_updates[0].arrival.time == int(datetime(2026, 9, 17, 12, 5, 0, tzinfo=timezone.utc).timestamp())
    assert stop_updates[0].arrival.delay == 45
    assert stop_updates[0].arrival.uncertainty == 30


def test_skipped_stop_time_update_carries_schedule_relationship():
    pytest.importorskip("google.transit.gtfs_realtime_pb2")
    skipped = StopTimePrediction(
        trip_id="t2",
        start_date="20260917",
        stop_id="s1",
        stop_sequence=1,
        schedule_relationship="SKIPPED",
        model_id="fake",
        generated_at=GENERATED_AT,
    )
    tu = TripUpdatePrediction(
        trip_id="t2", start_date="20260917", stop_time_updates=[skipped], generated_at=GENERATED_AT
    )

    as_dict = feed_message_dict([tu], generated_at=GENERATED_AT)
    assert as_dict["entity"][0]["trip_update"]["stop_time_update"][0]["schedule_relationship"] == "SKIPPED"

    from google.transit import gtfs_realtime_pb2

    raw = encode_protobuf([tu], generated_at=GENERATED_AT)
    message = gtfs_realtime_pb2.FeedMessage()
    message.ParseFromString(raw)
    stu = message.entity[0].trip_update.stop_time_update[0]
    assert stu.schedule_relationship == gtfs_realtime_pb2.TripUpdate.StopTimeUpdate.SKIPPED
