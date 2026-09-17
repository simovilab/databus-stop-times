"""StopStatusDetector (generic GTFS-RT cross-check). Phase P4.

Arrival fires when a VehicleState has ``current_status == "STOPPED_AT"`` with
``current_stop_sequence`` matching a stop's ``stop_sequence``.

``actual_arrival_epoch`` is the epoch of the **first** such sample for each stop.
Subsequent samples with the same stopped status for the same stop are ignored
(one arrival per stop).

This uses the standard GTFS-RT VehiclePosition ``current_status`` /
``current_stop_sequence`` fields and therefore works with any conforming agency
(MBTA, bUCR, etc.) without agency-specific configuration.
"""

from __future__ import annotations

from etaval.domain.models import ActualArrival, Stop, TripInstance, VehicleState


class StopStatusDetector:
    """Detect arrivals from GTFS-RT STOPPED_AT status with matching stop sequence."""

    method: str = "stop_status"

    def detect(
        self,
        trip: TripInstance,
        position_history: list[VehicleState],
        stops: list[Stop],
    ) -> list[ActualArrival]:
        """Return at most one ActualArrival per stop, at the first STOPPED_AT epoch.

        Args:
            trip: The TripInstance being validated (provides trip_key).
            position_history: Chronologically ordered VehicleState samples.
            stops: Stops to match against (by stop_sequence).

        Returns:
            List of ActualArrivals for each stop that had a matching
            ``STOPPED_AT`` sample in the position history.
        """
        # Index stops by sequence for O(1) look-up.
        stop_by_seq: dict[int, Stop] = {s.stop_sequence: s for s in stops}

        # Walk history chronologically; record first STOPPED_AT epoch per stop.
        first_epoch_by_seq: dict[int, float] = {}
        for state in position_history:
            if (
                state.current_status == "STOPPED_AT"
                and state.current_stop_sequence is not None
                and state.current_stop_sequence in stop_by_seq
                and state.current_stop_sequence not in first_epoch_by_seq
            ):
                first_epoch_by_seq[state.current_stop_sequence] = state.epoch

        arrivals: list[ActualArrival] = []
        for seq, epoch in first_epoch_by_seq.items():
            stop = stop_by_seq[seq]
            arrivals.append(
                ActualArrival(
                    trip_key=trip.trip_key,
                    stop_id=stop.stop_id,
                    stop_sequence=stop.stop_sequence,
                    actual_arrival_epoch=epoch,
                    method=self.method,
                )
            )

        return arrivals
