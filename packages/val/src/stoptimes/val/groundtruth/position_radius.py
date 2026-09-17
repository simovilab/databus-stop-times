"""PositionRadiusDetector. Phase P4.

Arrival fires when the vehicle comes within *arrival_radius_m* (great-circle,
haversine) of a stop's lat/lon.

``actual_arrival_epoch`` is the epoch of the **first entry** into the radius.

Optional dwell requirement
--------------------------
If ``dwell_s > 0`` the detector requires the vehicle to remain continuously
within the radius for at least *dwell_s* seconds before the arrival is confirmed.
The span of the confirmed dwell is measured from the first-entry epoch.  If the
vehicle exits the radius before the dwell window is satisfied, the run is reset
and the next entry starts a fresh window.
"""

from __future__ import annotations

from stoptimes.core.domain.models import ActualArrival, Stop, TripInstance, VehicleState
from stoptimes.core.spatial.polyline import haversine_m


class PositionRadiusDetector:
    """Detect arrivals by geographic proximity to each stop's lat/lon."""

    method: str = "position_radius"

    def __init__(
        self,
        arrival_radius_m: float = 40.0,
        dwell_s: float = 0.0,
    ) -> None:
        self.arrival_radius_m = arrival_radius_m
        self.dwell_s = dwell_s

    def detect(
        self,
        trip: TripInstance,
        position_history: list[VehicleState],
        stops: list[Stop],
    ) -> list[ActualArrival]:
        """Return at most one ActualArrival per stop on first confirmed entry.

        Args:
            trip: The TripInstance being validated (provides trip_key).
            position_history: Chronologically ordered VehicleState samples.
            stops: Stops with lat/lon populated.

        Returns:
            List of ActualArrivals.  For each stop the arrival epoch is the
            first in-radius sample epoch (or the first-entry epoch once the
            optional dwell window is satisfied).
        """
        arrivals: list[ActualArrival] = []

        for stop in stops:
            first_entry: float | None = None

            for state in position_history:
                dist_m = haversine_m(state.lat, state.lon, stop.lat, stop.lon)
                in_radius = dist_m <= self.arrival_radius_m

                if in_radius:
                    if first_entry is None:
                        # Record the moment the vehicle entered the radius.
                        first_entry = state.epoch

                    # Confirm immediately (no dwell) or once dwell window is met.
                    if self.dwell_s <= 0.0 or (state.epoch - first_entry) >= self.dwell_s:
                        arrivals.append(
                            ActualArrival(
                                trip_key=trip.trip_key,
                                stop_id=stop.stop_id,
                                stop_sequence=stop.stop_sequence,
                                actual_arrival_epoch=first_entry,
                                method=self.method,
                            )
                        )
                        break  # at most one arrival per stop
                else:
                    # Vehicle exited the radius; reset the dwell window.
                    first_entry = None

        return arrivals
