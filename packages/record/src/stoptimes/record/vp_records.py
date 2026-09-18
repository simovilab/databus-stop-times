"""GTFS-RT VehiclePositions -> flat archive records.

Django-free counterpart to ``tu_records.py`` for the VehiclePositions entity.
The legacy Postgres/Celery collector (``legacy/collector/rt_pipeline/tasks.py``
``_vp_entity_to_record``) does the same flattening but reads ``feed_name``
from Django settings; this module takes it as a parameter so it is
unit-testable without a settings module or a database.

Produces rows shaped for ``storage.schema.DATA_COLUMNS`` (+ ``route_id``,
the spool's partition key).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from google.transit import gtfs_realtime_pb2

_STOP_STATUS = gtfs_realtime_pb2.VehiclePosition.VehicleStopStatus


def _to_ts(seconds: Optional[int]) -> Optional[datetime]:
    return datetime.fromtimestamp(seconds, tz=timezone.utc) if seconds else None


def vehicle_position_record(
    entity,
    *,
    feed_name: str,
    ingested_at: datetime,
) -> Optional[dict[str, Any]]:
    """Flatten one FeedEntity into one VehiclePositions archive row.

    Returns ``None`` when the entity carries no ``vehicle`` payload.
    ``route_id`` may be ``None`` -- unlike TripUpdates, a VP row with no
    route is still a valid observation (e.g. a vehicle not yet assigned).
    """
    if not entity.HasField("vehicle"):
        return None

    v = entity.vehicle
    vehicle_id = (
        (v.vehicle.id or v.vehicle.label) if v.HasField("vehicle") else None
    ) or entity.id or "unknown"

    return {
        "feed_name": feed_name,
        "vehicle_id": vehicle_id.strip(),
        "ts": _to_ts(v.timestamp) if v.HasField("timestamp") else ingested_at,
        "lat": v.position.latitude if v.HasField("position") else None,
        "lon": v.position.longitude if v.HasField("position") else None,
        "bearing": (
            v.position.bearing
            if v.HasField("position") and v.position.HasField("bearing")
            else None
        ),
        "speed": (
            v.position.speed
            if v.HasField("position") and v.position.HasField("speed")
            else None
        ),
        "route_id": v.trip.route_id or None if v.HasField("trip") else None,
        "trip_id": v.trip.trip_id or None if v.HasField("trip") else None,
        "current_stop_sequence": (
            v.current_stop_sequence if v.HasField("current_stop_sequence") else None
        ),
        "current_status": (
            _STOP_STATUS.Name(v.current_status) if v.HasField("current_status") else None
        ),
        "stop_id": v.stop_id or None if v.HasField("stop_id") else None,
        "ingested_at": ingested_at,
    }


def feed_records(
    feed,
    *,
    feed_name: str,
    ingested_at: datetime,
) -> list[dict[str, Any]]:
    """Flatten a whole VehiclePositions FeedMessage. See :func:`vehicle_position_record`."""
    out: list[dict[str, Any]] = []
    for entity in feed.entity:
        row = vehicle_position_record(entity, feed_name=feed_name, ingested_at=ingested_at)
        if row is not None:
            out.append(row)
    return out
