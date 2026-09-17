"""GTFS-RT JSON feed adapters (httpx, async). Phase P1.

VehiclePositionsPoller + TripUpdatesPoller: parse the standard GTFS-RT
FeedMessage (header.timestamp, entity[]).

Parsing is separated from network IO so unit tests can exercise it without
httpx.

VP entity → VehiclePositionRecord (carries VehicleState + trip identity).
TU entity → TripUpdateRecord (carries per-stop arrival predictions).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

from etaval.domain.models import VehicleState

# ---------------------------------------------------------------------------
# Data containers (returned by parse_* functions and poll())
# ---------------------------------------------------------------------------


@dataclass
class VehiclePositionRecord:
    """Parsed VP entity: trip identity + one VehicleState sample."""

    trip_id: str
    route_id: str
    direction_id: int | None
    start_date: str | None
    vehicle_id: str
    state: VehicleState


@dataclass
class StopTimeUpdate:
    """One stop_time_update element from a TripUpdate entity."""

    stop_sequence: int | None
    stop_id: str | None
    arrival_time: float | None
    departure_time: float | None


@dataclass
class TripUpdateRecord:
    """Parsed TU entity: trip identity + list of stop-time predictions."""

    trip_id: str
    route_id: str | None
    direction_id: int | None
    start_date: str | None
    stop_time_updates: list[StopTimeUpdate] = field(default_factory=list)


@dataclass(frozen=True)
class VpPollResult:
    """Return type from VehiclePositionsPoller.poll()."""

    header_timestamp: float
    records: list[VehiclePositionRecord]


@dataclass(frozen=True)
class TuPollResult:
    """Return type from TripUpdatesPoller.poll()."""

    header_timestamp: float
    records: list[TripUpdateRecord]


# ---------------------------------------------------------------------------
# Defensive nested-dict accessors
# ---------------------------------------------------------------------------


def _opt_float(obj: Any, *keys: str) -> float | None:
    """Traverse nested dict keys → float, or None if any step is missing/wrong type."""
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


def _opt_int(obj: Any, *keys: str) -> int | None:
    """Traverse nested dict keys → int, or None if any step is missing/wrong type."""
    cur: Any = obj
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    if cur is None:
        return None
    try:
        return int(cur)
    except (TypeError, ValueError):
        return None


def _opt_str(obj: Any, *keys: str) -> str | None:
    """Traverse nested dict keys → str, or None if any step is missing."""
    cur: Any = obj
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    if cur is None:
        return None
    return str(cur)


# ---------------------------------------------------------------------------
# Pure parsing functions (no network — unit-testable)
# ---------------------------------------------------------------------------


def parse_vp_feed(
    feed: dict[str, Any],
) -> tuple[float, list[VehiclePositionRecord]]:
    """Parse a VehiclePositions FeedMessage dict.

    Returns (header_timestamp, list[VehiclePositionRecord]).
    Entities missing lat/lon or trip_id are silently skipped.
    All other optional fields fall back to None.
    """
    header_ts = float(feed.get("header", {}).get("timestamp", 0) or 0)
    records: list[VehiclePositionRecord] = []

    for entity in feed.get("entity", []):
        veh = entity.get("vehicle")
        if not isinstance(veh, dict):
            continue

        trip = veh.get("trip") or {}
        trip_id = _opt_str(trip, "trip_id")
        if not trip_id:
            continue

        route_id = _opt_str(trip, "route_id") or ""
        direction_id = _opt_int(trip, "direction_id")
        start_date = _opt_str(trip, "start_date")

        # vehicle.vehicle.id is the canonical vehicle identifier; fall back to entity.id
        vehicle_id = _opt_str(veh, "vehicle", "id") or str(entity.get("id") or "")

        pos = veh.get("position") or {}
        lat = _opt_float(pos, "latitude")
        lon = _opt_float(pos, "longitude")
        if lat is None or lon is None:
            continue

        # epoch: prefer per-vehicle timestamp, fall back to feed header
        epoch = float(veh.get("timestamp") or header_ts)

        state = VehicleState(
            epoch=epoch,
            lat=lat,
            lon=lon,
            progress_m=None,
            speed=_opt_float(pos, "speed"),
            current_status=_opt_str(veh, "current_status"),
            current_stop_sequence=_opt_int(veh, "current_stop_sequence"),
        )
        records.append(
            VehiclePositionRecord(
                trip_id=trip_id,
                route_id=route_id,
                direction_id=direction_id,
                start_date=start_date,
                vehicle_id=vehicle_id,
                state=state,
            )
        )

    return header_ts, records


def parse_tu_feed(
    feed: dict[str, Any],
) -> tuple[float, list[TripUpdateRecord]]:
    """Parse a TripUpdates FeedMessage dict.

    Returns (header_timestamp, list[TripUpdateRecord]).
    Entities missing trip_id are silently skipped.
    """
    header_ts = float(feed.get("header", {}).get("timestamp", 0) or 0)
    records: list[TripUpdateRecord] = []

    for entity in feed.get("entity", []):
        tu = entity.get("trip_update")
        if not isinstance(tu, dict):
            continue

        trip = tu.get("trip") or {}
        trip_id = _opt_str(trip, "trip_id")
        if not trip_id:
            continue

        route_id = _opt_str(trip, "route_id")
        direction_id = _opt_int(trip, "direction_id")
        start_date = _opt_str(trip, "start_date")

        updates: list[StopTimeUpdate] = []
        for stu in tu.get("stop_time_update") or []:
            if not isinstance(stu, dict):
                continue
            updates.append(
                StopTimeUpdate(
                    stop_sequence=_opt_int(stu, "stop_sequence"),
                    stop_id=_opt_str(stu, "stop_id"),
                    arrival_time=_opt_float(stu, "arrival", "time"),
                    departure_time=_opt_float(stu, "departure", "time"),
                )
            )

        records.append(
            TripUpdateRecord(
                trip_id=trip_id,
                route_id=route_id,
                direction_id=direction_id,
                start_date=start_date,
                stop_time_updates=updates,
            )
        )

    return header_ts, records


# ---------------------------------------------------------------------------
# Async pollers (FeedAdapter-compatible)
# ---------------------------------------------------------------------------


class VehiclePositionsPoller:
    """Async GTFS-RT VehiclePositions JSON poller using httpx."""

    def __init__(self, url: str, headers: dict[str, str] | None = None) -> None:
        self._url = url
        self._headers: dict[str, str] = headers or {}

    async def poll(self) -> VpPollResult:
        """GET the VP feed, parse, and return a VpPollResult."""
        async with httpx.AsyncClient() as client:
            response = await client.get(self._url, headers=self._headers)
            response.raise_for_status()
            feed: dict[str, Any] = response.json()
        header_ts, records = parse_vp_feed(feed)
        return VpPollResult(header_timestamp=header_ts, records=records)


class TripUpdatesPoller:
    """Async GTFS-RT TripUpdates JSON poller using httpx."""

    def __init__(self, url: str, headers: dict[str, str] | None = None) -> None:
        self._url = url
        self._headers: dict[str, str] = headers or {}

    async def poll(self) -> TuPollResult:
        """GET the TU feed, parse, and return a TuPollResult."""
        async with httpx.AsyncClient() as client:
            response = await client.get(self._url, headers=self._headers)
            response.raise_for_status()
            feed: dict[str, Any] = response.json()
        header_ts, records = parse_tu_feed(feed)
        return TuPollResult(header_timestamp=header_ts, records=records)
