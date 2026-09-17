"""GTFS-RT TripUpdates -> flat archive records.

Django-free on purpose: this is the part worth unit-testing (route scoping,
horizon capping, enum naming, the fields MBTA does and does not publish), and
it should be testable without a settings module or a database. The legacy
collector task in ``legacy/collector/rt_pipeline/tasks.py`` supplies
``feed_name`` and the config values.

Scoping decisions live here rather than downstream because storage is the
constraint: the TU feed is ~22x the VP feed per poll, and a row not built is
a row never spooled, never flushed and never paid for.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from google.transit import gtfs_realtime_pb2

# MBTA publishes `revenue` (false on deadhead/pull-out trips) as a proto
# extension to TripDescriptor, which the upstream gtfs-realtime-bindings do
# not declare. Rather than vendoring MBTA's .proto, read the field only if the
# installed bindings happen to expose it and record NULL otherwise -- an
# absent column would silently change meaning if the bindings later gain it.
_TRIP_FIELDS = {f.name for f in gtfs_realtime_pb2.TripDescriptor.DESCRIPTOR.fields}
TRIP_HAS_REVENUE = "revenue" in _TRIP_FIELDS

_TRIP_SR = gtfs_realtime_pb2.TripDescriptor.ScheduleRelationship
_STU_SR = gtfs_realtime_pb2.TripUpdate.StopTimeUpdate.ScheduleRelationship


def _to_ts(seconds: Optional[int]) -> Optional[datetime]:
    return datetime.fromtimestamp(seconds, tz=timezone.utc) if seconds else None


def _enum_name(enum, value: Optional[int]) -> Optional[str]:
    """Enum value -> its name, or None. Never raises on an unknown code.

    The legacy Postgres TU path stored ``str(int)`` here; storing the name
    keeps the archive readable a year from now without a lookup table, and an
    unrecognised future code degrades to its number rather than an exception.
    """
    if value is None:
        return None
    try:
        return enum.Name(value)
    except ValueError:
        return str(value)


def _event_fields(event, present: bool) -> tuple[Optional[datetime], Optional[int], Optional[int]]:
    """(time, delay, uncertainty) for a StopTimeEvent, all None when absent.

    ``uncertainty`` is the field that makes the archive worth keeping: MBTA
    marks schedule-plus-historical predictions with a large uncertainty (300
    on bus, 60/360 on rail) and realtime ones with none, so a baseline built
    from this feed can be stratified instead of silently blending the two.
    """
    if not present:
        return None, None, None
    return (
        _to_ts(event.time) if event.HasField("time") else None,
        event.delay if event.HasField("delay") else None,
        event.uncertainty if event.HasField("uncertainty") else None,
    )


def trip_update_records(
    entity,
    *,
    feed_name: str,
    header_ts: Optional[datetime],
    ingested_at: datetime,
    routes: Optional[frozenset[str]] = None,
    max_horizon_seconds: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Flatten one FeedEntity into one record per StopTimeUpdate.

    Returns ``[]`` (cheaply, before touching the stop times) when the entity
    is not a trip update or its route is outside ``routes``. ``routes=None``
    means every route -- the unscoped whole-feed archive, which is ~80x the
    storage of the corpus routes and should be a deliberate choice.

    ``max_horizon_seconds`` drops predictions further than that beyond the
    update's own timestamp. Rows with no predicted time at all are kept: a
    SKIPPED stop carries no time and is still information.
    """
    if not entity.HasField("trip_update"):
        return []

    tu = entity.trip_update
    trip = tu.trip
    route_id = trip.route_id or None
    if routes is not None and route_id not in routes:
        return []

    # The TripUpdate's own timestamp is the instant the prediction was in
    # effect; the header timestamp is the fallback when a producer omits it.
    ts = _to_ts(tu.timestamp) if tu.HasField("timestamp") else None
    ts = ts or header_ts or ingested_at

    trip_id = trip.trip_id or None
    if not trip_id:
        # The natural key is (feed, trip, start_date, stop, sequence, ts); a
        # trip update with no trip_id cannot be keyed or replayed, so it is
        # dropped rather than stored under a synthetic id that would collide.
        return []

    base = {
        "feed_name": feed_name,
        "route_id": route_id,
        "trip_id": trip_id,
        "start_date": trip.start_date or None,
        "start_time": trip.start_time or None,
        "direction_id": trip.direction_id if trip.HasField("direction_id") else None,
        "vehicle_id": (tu.vehicle.id or tu.vehicle.label or None)
        if tu.HasField("vehicle") else None,
        "trip_schedule_relationship": _enum_name(
            _TRIP_SR, trip.schedule_relationship
            if trip.HasField("schedule_relationship") else None
        ),
        # HasField, not a bare read: if a future bindings upgrade declares
        # `revenue` as a plain proto3 bool, "absent" and "explicitly false"
        # both read as False, which would quietly turn "unknown" into
        # "confirmed deadhead".
        "revenue": (
            bool(trip.revenue)
            if TRIP_HAS_REVENUE and trip.HasField("revenue")
            else None
        ),
        "ts": ts,
        "header_ts": header_ts,
        "ingested_at": ingested_at,
    }

    records: list[dict[str, Any]] = []
    for stu in tu.stop_time_update:
        arr_time, arr_delay, arr_unc = _event_fields(stu.arrival, stu.HasField("arrival"))
        dep_time, dep_delay, dep_unc = _event_fields(
            stu.departure, stu.HasField("departure")
        )

        if max_horizon_seconds is not None:
            predicted = arr_time or dep_time
            if predicted is not None and (predicted - ts).total_seconds() > max_horizon_seconds:
                continue

        records.append({
            **base,
            "stop_id": stu.stop_id or None,
            "stop_sequence": stu.stop_sequence if stu.HasField("stop_sequence") else None,
            "arrival_time": arr_time,
            "arrival_delay": arr_delay,
            "arrival_uncertainty": arr_unc,
            "departure_time": dep_time,
            "departure_delay": dep_delay,
            "departure_uncertainty": dep_unc,
            "stu_schedule_relationship": _enum_name(
                _STU_SR, stu.schedule_relationship
                if stu.HasField("schedule_relationship") else None
            ),
        })
    return records


def feed_records(
    feed,
    *,
    feed_name: str,
    ingested_at: datetime,
    routes: Optional[frozenset[str]] = None,
    max_horizon_seconds: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Flatten a whole FeedMessage. See :func:`trip_update_records`."""
    header_ts = _to_ts(feed.header.timestamp) if feed.HasField("header") else None
    out: list[dict[str, Any]] = []
    for entity in feed.entity:
        out.extend(trip_update_records(
            entity,
            feed_name=feed_name,
            header_ts=header_ts,
            ingested_at=ingested_at,
            routes=routes,
            max_horizon_seconds=max_horizon_seconds,
        ))
    return out


def parse_routes(raw: Optional[Iterable[str] | str]) -> Optional[frozenset[str]]:
    """Config value -> route filter. Empty/absent means "every route".

    Accepts a comma-separated string (how it arrives from the environment) or
    any iterable of ids.
    """
    if raw is None:
        return None
    items = raw.split(",") if isinstance(raw, str) else list(raw)
    cleaned = frozenset(s.strip() for s in items if s and s.strip())
    return cleaned or None
