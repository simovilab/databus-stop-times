"""Encode predictions as spec-valid GTFS-RT TripUpdates (protobuf + JSON).

`contracts.prediction.TripUpdatePrediction.to_trip_update_dict()` already
renders one trip's worth of predictions as a GTFS-RT-shaped dict; this module
wraps that into a full `FeedMessage` (header + entities) and offers both a
JSON encoding (used by `core.realtime.gtfsrt_json.parse_tu_feed`, which reads
this exact snake_case shape) and a protobuf encoding via the declared
`stoptimes-core[pb]` extra (`google.transit.gtfs_realtime_pb2`).

Both directions round-trip: `encode_json` -> `core.realtime.gtfsrt_json.parse_tu_feed`
and `encode_protobuf` -> `gtfs_realtime_pb2.FeedMessage.FromString` recover the
same values that went in (see `packages/serve/tests/test_tripupdates.py`).
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from stoptimes.core.contracts.prediction import TripUpdatePrediction

GTFS_REALTIME_VERSION = "2.0"

__all__ = [
    "GTFS_REALTIME_VERSION",
    "feed_message_dict",
    "encode_json",
    "encode_protobuf",
    "decode_protobuf",
]


def _entity_id(trip_update: TripUpdatePrediction) -> str:
    return f"tu:{trip_update.trip_id}:{trip_update.start_date}"


def feed_message_dict(
    trip_updates: list[TripUpdatePrediction],
    *,
    generated_at: datetime,
    incrementality: str = "FULL_DATASET",
) -> dict[str, Any]:
    """A full GTFS-RT `FeedMessage` as a JSON-shaped dict.

    `header.incrementality` is a string ("FULL_DATASET" | "DIFFERENTIAL")
    rather than the protobuf enum so this dict has no protobuf dependency,
    matching `contracts.prediction`'s own no-protobuf policy.
    """
    return {
        "header": {
            "gtfs_realtime_version": GTFS_REALTIME_VERSION,
            "incrementality": incrementality,
            "timestamp": int(generated_at.timestamp()),
        },
        "entity": [
            {"id": _entity_id(tu), "trip_update": tu.to_trip_update_dict()} for tu in trip_updates
        ],
    }


def encode_json(trip_updates: list[TripUpdatePrediction], *, generated_at: datetime, **kwargs: Any) -> bytes:
    """UTF-8 JSON bytes for a `FeedMessage`, parseable by
    `core.realtime.gtfsrt_json.parse_tu_feed`."""
    return json.dumps(feed_message_dict(trip_updates, generated_at=generated_at, **kwargs)).encode("utf-8")


_SCHEDULE_RELATIONSHIP_TRIP = None
_SCHEDULE_RELATIONSHIP_STOP = None


def _pb2():
    from google.transit import gtfs_realtime_pb2

    return gtfs_realtime_pb2


def encode_protobuf(
    trip_updates: list[TripUpdatePrediction],
    *,
    generated_at: datetime,
    incrementality: str = "FULL_DATASET",
) -> bytes:
    """Serialized `gtfs_realtime_pb2.FeedMessage` bytes.

    Needs the `stoptimes-core[pb]` extra (`gtfs-realtime-bindings`); the
    import is deferred here so a JSON-only deployment never needs it.
    """
    pb2 = _pb2()
    message = pb2.FeedMessage()
    message.header.gtfs_realtime_version = GTFS_REALTIME_VERSION
    message.header.incrementality = getattr(pb2.FeedHeader, incrementality)
    message.header.timestamp = int(generated_at.timestamp())

    for tu in trip_updates:
        entity = message.entity.add()
        entity.id = _entity_id(tu)
        trip_update = entity.trip_update

        trip_update.trip.trip_id = tu.trip_id
        trip_update.trip.start_date = tu.start_date
        if tu.route_id is not None:
            trip_update.trip.route_id = tu.route_id
        if tu.direction_id is not None:
            trip_update.trip.direction_id = tu.direction_id
        if tu.start_time is not None:
            trip_update.trip.start_time = tu.start_time
        trip_update.trip.schedule_relationship = getattr(pb2.TripDescriptor, tu.schedule_relationship.value)

        if tu.vehicle_id is not None:
            trip_update.vehicle.id = tu.vehicle_id

        trip_update.timestamp = int(tu.generated_at.timestamp())

        for stu in tu.stop_time_updates:
            update = trip_update.stop_time_update.add()
            update.stop_sequence = stu.stop_sequence
            update.stop_id = stu.stop_id
            if stu.schedule_relationship.value != "SCHEDULED":
                update.schedule_relationship = getattr(pb2.TripUpdate.StopTimeUpdate, stu.schedule_relationship.value)
            if stu.predicted_arrival_time is not None:
                update.arrival.time = int(stu.predicted_arrival_time.timestamp())
                if stu.delay_s is not None:
                    update.arrival.delay = int(round(stu.delay_s))
                if stu.uncertainty_s is not None:
                    update.arrival.uncertainty = int(round(stu.uncertainty_s))
            if stu.predicted_departure_time is not None:
                update.departure.time = int(stu.predicted_departure_time.timestamp())
                if stu.delay_s is not None:
                    update.departure.delay = int(round(stu.delay_s))
                if stu.uncertainty_s is not None:
                    update.departure.uncertainty = int(round(stu.uncertainty_s))

    return message.SerializeToString()


def decode_protobuf(data: bytes) -> Any:
    """Parse `data` back into a `gtfs_realtime_pb2.FeedMessage` (for tests
    and any caller that wants to inspect what was just encoded)."""
    pb2 = _pb2()
    message = pb2.FeedMessage()
    message.ParseFromString(data)
    return message
