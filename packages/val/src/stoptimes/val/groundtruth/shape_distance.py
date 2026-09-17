"""ShapeDistanceDetector (default). Phase P4.

Arrival fires when the vehicle's monotonic progress_m reaches a stop's progress_m
(within shape_threshold_m, at closest approach); loop-back safe because progress_m
is monotonic (assigned by assign_stops_monotonic).

Algorithm
---------
For each stop with a valid progress_m, scan the chronological position history and
find the sample where ``abs(state.progress_m - stop.progress_m)`` is minimised.
If that minimum is within *shape_threshold_m*, emit an ActualArrival whose
``actual_arrival_epoch`` is the epoch of that closest-approach sample.

Because position history progress_m is also monotonic (the vehicle only moves
forward along the shape), the closest-approach sample is loop-back safe: a later
stop will always have its closest approach found at a later (or equal) point in
time, never wrapping back to a previous segment.
"""

from __future__ import annotations

from stoptimes.core.domain.models import ActualArrival, Stop, TripInstance, VehicleState


class ShapeDistanceDetector:
    """Detect arrivals via closest-approach in shape-distance (progress_m) space."""

    method: str = "shape_distance"

    def __init__(self, shape_threshold_m: float = 30.0) -> None:
        self.shape_threshold_m = shape_threshold_m

    def detect(
        self,
        trip: TripInstance,
        position_history: list[VehicleState],
        stops: list[Stop],
    ) -> list[ActualArrival]:
        """Return at most one ActualArrival per stop, at the epoch of closest approach.

        Args:
            trip: The TripInstance being validated (provides trip_key).
            position_history: Chronologically ordered VehicleState samples.
            stops: Stops with ``progress_m`` assigned (via assign_stops_monotonic).

        Returns:
            List of ActualArrivals for stops the vehicle approached within
            *shape_threshold_m* in progress space. Stops the vehicle never
            approached within the threshold are omitted.
        """
        # Narrow type: only keep states with a concrete progress_m value.
        valid_states: list[tuple[VehicleState, float]] = [
            (s, s.progress_m)
            for s in position_history
            if s.progress_m is not None
        ]
        if not valid_states or not stops:
            return []

        arrivals: list[ActualArrival] = []
        for stop in stops:
            if stop.progress_m is None:
                continue

            stop_pm: float = stop.progress_m

            # Closest approach: minimise absolute delta in progress space.
            best_state, best_pm = min(
                valid_states,
                key=lambda pair: abs(pair[1] - stop_pm),
            )
            min_delta = abs(best_pm - stop_pm)

            if min_delta <= self.shape_threshold_m:
                arrivals.append(
                    ActualArrival(
                        trip_key=trip.trip_key,
                        stop_id=stop.stop_id,
                        stop_sequence=stop.stop_sequence,
                        actual_arrival_epoch=best_state.epoch,
                        method=self.method,
                    )
                )

        return arrivals
