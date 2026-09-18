"""Format-agnostic entry points: raw feed bytes -> archive rows.

The recorder polls both wire formats (``FeedFormat.PROTOBUF`` / ``.JSON``,
``stoptimes.core.contracts.source``) for both entity kinds (vehicle
positions, trip updates). Protobuf parsing reuses the existing, tested
flatteners (``vp_records``, ``tu_records``); JSON parsing mirrors them field
for field against the standard GTFS-RT JSON mapping (snake_case keys, enum
names as strings, epoch seconds as numbers or numeric strings).

A malformed payload (bad protobuf bytes, invalid JSON, a JSON document that
isn't a feed message) raises :class:`FeedParseError` -- callers treat this
like an HTTP failure: log it, keep the previous poll's cache state, retry
next cycle. It never corrupts the spool.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

from google.protobuf.message import DecodeError
from google.transit import gtfs_realtime_pb2
from stoptimes.core.contracts.source import FeedFormat

from . import tu_records, vp_records


class FeedParseError(ValueError):
    """Raised for malformed protobuf bytes or invalid/unexpected JSON."""


def _opt_float(obj: Any, *keys: str) -> Optional[float]:
    cur: Any = obj
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    if cur is None:
        return None
    try:
        return float(cur)
    except (TypeError, ValueError):
        return None


def _opt_int(obj: Any, *keys: str) -> Optional[int]:
    val = _opt_float(obj, *keys)
    return int(val) if val is not None else None


def _opt_str(obj: Any, *keys: str) -> Optional[str]:
    cur: Any = obj
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    if cur is None or cur == "":
        return None
    return str(cur)


def _to_ts(seconds: Optional[float]) -> Optional[datetime]:
    return datetime.fromtimestamp(seconds, tz=timezone.utc) if seconds else None


def parse_payload(payload: bytes, fmt: FeedFormat) -> Any:
    """Decode raw response bytes into a feed message.

    Protobuf -> a ``gtfs_realtime_pb2.FeedMessage``. JSON -> a ``dict``.
    Raises :class:`FeedParseError` on malformed input of either kind.
    """
    if fmt == FeedFormat.PROTOBUF:
        feed = gtfs_realtime_pb2.FeedMessage()
        try:
            feed.MergeFromString(payload)
        except DecodeError as exc:
            raise FeedParseError(f"malformed protobuf feed: {exc}") from exc
        return feed

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise FeedParseError(f"malformed JSON feed: {exc}") from exc
    if not isinstance(data, dict):
        raise FeedParseError("JSON feed body is not an object")
    return data


# --- VehiclePositions -------------------------------------------------------


def _vp_rows_from_json(feed: dict[str, Any], *, feed_name: str, ingested_at: datetime) -> list[dict[str, Any]]:
    header_ts = _opt_float(feed, "header", "timestamp")
    out: list[dict[str, Any]] = []
    for entity in feed.get("entity") or []:
        veh = entity.get("vehicle") if isinstance(entity, dict) else None
        if not isinstance(veh, dict):
            continue
        trip = veh.get("trip") or {}
        pos = veh.get("position") or {}
        vehicle_id = (
            _opt_str(veh, "vehicle", "id")
            or _opt_str(veh, "vehicle", "label")
            or _opt_str(entity, "id")
            or "unknown"
        )
        ts = _to_ts(_opt_float(veh, "timestamp")) or _to_ts(header_ts) or ingested_at
        out.append({
            "feed_name": feed_name,
            "vehicle_id": vehicle_id.strip(),
            "ts": ts,
            "lat": _opt_float(pos, "latitude"),
            "lon": _opt_float(pos, "longitude"),
            "bearing": _opt_float(pos, "bearing"),
            "speed": _opt_float(pos, "speed"),
            "route_id": _opt_str(trip, "route_id"),
            "trip_id": _opt_str(trip, "trip_id"),
            "current_stop_sequence": _opt_int(veh, "current_stop_sequence"),
            "current_status": _opt_str(veh, "current_status"),
            "stop_id": _opt_str(veh, "stop_id"),
            "ingested_at": ingested_at,
        })
    return out


def vp_rows(
    feed: Any,
    fmt: FeedFormat,
    *,
    feed_name: str,
    ingested_at: datetime,
) -> list[dict[str, Any]]:
    """VehiclePositions archive rows from an already-decoded feed message."""
    if fmt == FeedFormat.PROTOBUF:
        return vp_records.feed_records(feed, feed_name=feed_name, ingested_at=ingested_at)
    return _vp_rows_from_json(feed, feed_name=feed_name, ingested_at=ingested_at)


# --- TripUpdates --------------------------------------------------------


def _tu_rows_from_json(
    feed: dict[str, Any],
    *,
    feed_name: str,
    ingested_at: datetime,
    routes: Optional[frozenset[str]] = None,
    max_horizon_seconds: Optional[int] = None,
) -> list[dict[str, Any]]:
    header_ts = _to_ts(_opt_float(feed, "header", "timestamp"))
    out: list[dict[str, Any]] = []
    for entity in feed.get("entity") or []:
        tu = entity.get("trip_update") if isinstance(entity, dict) else None
        if not isinstance(tu, dict):
            continue
        trip = tu.get("trip") or {}
        route_id = _opt_str(trip, "route_id")
        if routes is not None and route_id not in routes:
            continue
        trip_id = _opt_str(trip, "trip_id")
        if not trip_id:
            continue

        ts = _to_ts(_opt_float(tu, "timestamp")) or header_ts or ingested_at
        vehicle = tu.get("vehicle") or {}
        base = {
            "feed_name": feed_name,
            "route_id": route_id,
            "trip_id": trip_id,
            "start_date": _opt_str(trip, "start_date"),
            "start_time": _opt_str(trip, "start_time"),
            "direction_id": _opt_int(trip, "direction_id"),
            "vehicle_id": _opt_str(vehicle, "id") or _opt_str(vehicle, "label"),
            "trip_schedule_relationship": _opt_str(trip, "schedule_relationship"),
            "revenue": trip.get("revenue") if isinstance(trip.get("revenue"), bool) else None,
            "ts": ts,
            "header_ts": header_ts,
            "ingested_at": ingested_at,
        }

        for stu in tu.get("stop_time_update") or []:
            if not isinstance(stu, dict):
                continue
            arrival = stu.get("arrival") or {}
            departure = stu.get("departure") or {}
            arr_time = _to_ts(_opt_float(arrival, "time"))
            dep_time = _to_ts(_opt_float(departure, "time"))

            if max_horizon_seconds is not None:
                predicted = arr_time or dep_time
                if predicted is not None and (predicted - ts).total_seconds() > max_horizon_seconds:
                    continue

            out.append({
                **base,
                "stop_id": _opt_str(stu, "stop_id"),
                "stop_sequence": _opt_int(stu, "stop_sequence"),
                "arrival_time": arr_time,
                "arrival_delay": _opt_int(arrival, "delay"),
                "arrival_uncertainty": _opt_int(arrival, "uncertainty"),
                "departure_time": dep_time,
                "departure_delay": _opt_int(departure, "delay"),
                "departure_uncertainty": _opt_int(departure, "uncertainty"),
                "stu_schedule_relationship": _opt_str(stu, "schedule_relationship"),
            })
    return out


def tu_rows(
    feed: Any,
    fmt: FeedFormat,
    *,
    feed_name: str,
    ingested_at: datetime,
    routes: Optional[frozenset[str]] = None,
    max_horizon_seconds: Optional[int] = None,
) -> list[dict[str, Any]]:
    """TripUpdates archive rows from an already-decoded feed message."""
    if fmt == FeedFormat.PROTOBUF:
        return tu_records.feed_records(
            feed,
            feed_name=feed_name,
            ingested_at=ingested_at,
            routes=routes,
            max_horizon_seconds=max_horizon_seconds,
        )
    return _tu_rows_from_json(
        feed,
        feed_name=feed_name,
        ingested_at=ingested_at,
        routes=routes,
        max_horizon_seconds=max_horizon_seconds,
    )
