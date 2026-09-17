"""ActiveTripTracker. Phase P1.

Maintains TripInstances keyed by (trip_id, vehicle_id), appends position
history, and expires stale trips.
"""

from __future__ import annotations

from stoptimes.core.domain.models import TripInstance, VehicleState
from stoptimes.core.realtime.gtfsrt_json import VehiclePositionRecord

# (trip_id, vehicle_id)
TripKey = tuple[str, str]


class ActiveTripTracker:
    """Track live trips from a single GTFS-RT source.

    State is keyed by (trip_id, vehicle_id) — a composite that survives
    trip_id reassignment across service dates while keeping vehicles
    distinguishable when multiple vehicles share a trip (a rare but valid case).

    Usage::

        tracker = ActiveTripTracker(source="mbta")
        tracker.update(vp_records, now=time.time())
        trips = tracker.active_trips()
        history = tracker.position_history(("trip_id", "vehicle_id"))
        tracker.expire(now=time.time(), max_age_s=300)

    Note: shape_id is set to "" on creation — the engine populates it from
    GTFS static data (P2) by replacing the TripInstance.
    """

    def __init__(self, source: str = "") -> None:
        self._source = source
        self._trips: dict[TripKey, TripInstance] = {}
        self._history: dict[TripKey, list[VehicleState]] = {}
        self._last_update: dict[TripKey, float] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def update(self, vp_records: list[VehiclePositionRecord], now: float) -> None:
        """Ingest a batch of VP records at epoch *now*.

        New keys create a TripInstance (started_at = now).
        All keys append the record's VehicleState to position history.
        """
        for record in vp_records:
            key: TripKey = (record.trip_id, record.vehicle_id)

            if key not in self._trips:
                self._trips[key] = TripInstance(
                    source=self._source,
                    trip_id=record.trip_id,
                    vehicle_id=record.vehicle_id,
                    route_id=record.route_id,
                    direction_id=record.direction_id if record.direction_id is not None else 0,
                    shape_id="",  # assigned from GTFS static in P2
                    start_date=record.start_date or "",
                    started_at=now,
                )
                self._history[key] = []

            self._history[key].append(record.state)
            self._last_update[key] = now

    def expire(self, now: float, max_age_s: float) -> None:
        """Drop trips whose most recent update is older than *max_age_s* seconds."""
        stale: list[TripKey] = [
            k for k, last in self._last_update.items() if now - last > max_age_s
        ]
        for k in stale:
            self._trips.pop(k, None)
            self._history.pop(k, None)
            self._last_update.pop(k, None)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def position_history(self, trip_key: TripKey) -> list[VehicleState]:
        """Return the position history for *trip_key* (insertion order, immutable copy)."""
        return list(self._history.get(trip_key, []))

    def active_trips(self) -> list[TripInstance]:
        """Return all currently tracked TripInstances (insertion order)."""
        return list(self._trips.values())

    def get_trip(self, trip_key: TripKey) -> TripInstance | None:
        """Return the TripInstance for *trip_key*, or None if not tracked."""
        return self._trips.get(trip_key)
