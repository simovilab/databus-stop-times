"""GtfsRtTripUpdatesPredictor (predictor_id="gtfsrt"). Phase P3.

For the active trip's matching TU entity, emit one prediction per upcoming stop
using arrival.time; made_at = feed header.timestamp.

Input contract
--------------
The engine extracts the matching ``TripUpdateRecord`` from the ``TuPollResult``
and wraps it in a :class:`TripUpdateInput` before calling ``predict()``.

``TripUpdateInput`` carries:
- ``header_timestamp: float`` — the TU feed ``header.timestamp`` epoch; used as
  ``made_at_epoch`` on every emitted ``StopPrediction``.
- ``stop_time_updates: list[StopTimeUpdate]`` — per-stop predictions from
  ``stoptimes.core.realtime.gtfsrt_json.StopTimeUpdate``.

Matching priority:  stop_id first (exact string match), then stop_sequence as
fallback.  Stops with no matching TU entry or with ``arrival_time=None`` are
silently skipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from stoptimes.core.domain.models import Stop, TripInstance, VehicleState
from stoptimes.core.realtime.gtfsrt_json import StopTimeUpdate
from stoptimes.val.predictors.base import StopPrediction


@dataclass(frozen=True)
class TripUpdateInput:
    """Input record passed as ``trip_updates`` to :class:`GtfsRtTripUpdatesPredictor`.

    The engine builds this from a ``TripUpdateRecord`` + ``TuPollResult``:

    .. code-block:: python

        TripUpdateInput(
            header_timestamp=tu_poll_result.header_timestamp,
            stop_time_updates=matching_record.stop_time_updates,
        )

    Fields
    ------
    header_timestamp:
        ``header.timestamp`` from the TU feed; used as ``made_at_epoch`` on every
        emitted :class:`~stoptimes.val.predictors.base.StopPrediction`.
    stop_time_updates:
        Per-stop arrival/departure predictions for the matching trip entity,
        as parsed by ``stoptimes.core.realtime.gtfsrt_json.parse_tu_feed``.
    """

    header_timestamp: float
    stop_time_updates: list[StopTimeUpdate] = field(default_factory=list)


class GtfsRtTripUpdatesPredictor:
    """Replays GTFS-RT TripUpdate arrival times as per-stop predictions.

    ``predictor_id = "gtfsrt"``

    This predictor is a **pass-through**: it takes whatever the agency's GTFS-RT
    TripUpdates feed says and re-emits it as :class:`StopPrediction` objects so
    they can be scored against detected actual arrivals.

    Input contract for ``trip_updates``
    ------------------------------------
    Pass a :class:`TripUpdateInput` carrying ``header_timestamp`` and the list of
    :class:`~stoptimes.core.realtime.gtfsrt_json.StopTimeUpdate` objects for this trip.
    If ``trip_updates`` is ``None`` or not a ``TripUpdateInput``, ``predict()``
    returns an empty list without raising.
    """

    predictor_id: str = "gtfsrt"

    def predict(
        self,
        trip: TripInstance,
        upcoming_stops: list[Stop],
        vehicle_state: VehicleState,
        trip_updates: Any,
    ) -> list[StopPrediction]:
        """Emit one StopPrediction per upcoming stop that has a TU arrival_time.

        Args:
            trip: The active TripInstance (used for context; not inspected directly).
            upcoming_stops: Ordered list of stops ahead of the vehicle.
            vehicle_state: Latest vehicle position (not used by this predictor).
            trip_updates: A :class:`TripUpdateInput` for this trip, or ``None``/any
                other value if no TU data is available (returns ``[]``).

        Returns:
            One :class:`StopPrediction` per upcoming stop that has a matching
            TU entry with a non-``None`` ``arrival_time``.  ``stops_remaining``
            counts how many upcoming stops remain at or after each predicted stop
            (inclusive).  ``dist_remaining_m`` is always ``None`` (the TU feed
            does not carry shape-distance information).
        """
        if not isinstance(trip_updates, TripUpdateInput):
            return []

        header_ts = trip_updates.header_timestamp
        stu_list = trip_updates.stop_time_updates

        # Build O(1) lookup indices.
        by_stop_id: dict[str, StopTimeUpdate] = {}
        by_stop_seq: dict[int, StopTimeUpdate] = {}
        for stu in stu_list:
            if stu.stop_id is not None:
                by_stop_id[stu.stop_id] = stu
            if stu.stop_sequence is not None:
                by_stop_seq[stu.stop_sequence] = stu

        predictions: list[StopPrediction] = []
        n_upcoming = len(upcoming_stops)

        for idx, stop in enumerate(upcoming_stops):
            # Priority: stop_id match, then stop_sequence fallback.
            stu = by_stop_id.get(stop.stop_id)
            if stu is None:
                stu = by_stop_seq.get(stop.stop_sequence)
            if stu is None or stu.arrival_time is None:
                continue

            stops_remaining = n_upcoming - idx
            predictions.append(
                StopPrediction(
                    stop_id=stop.stop_id,
                    stop_sequence=stop.stop_sequence,
                    predicted_arrival_epoch=stu.arrival_time,
                    made_at_epoch=header_ts,
                    dist_remaining_m=None,
                    stops_remaining=stops_remaining,
                )
            )

        return predictions
