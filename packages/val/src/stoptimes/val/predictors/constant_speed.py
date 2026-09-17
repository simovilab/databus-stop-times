"""ConstantSpeedPredictor (predictor_id="baseline:constant_speed"). Phase P3.

etaval's own white-box baseline (zero external deps).

Speed estimation (priority order)
----------------------------------
1. ``Δprogress_m / Δt`` between the two most recent :class:`VehicleState` entries
   in ``position_history`` that have a valid (non-``None``) ``progress_m`` and
   distinct epochs.
2. ``vehicle_state.speed`` (as reported by the VP feed) when history is
   insufficient or yields a non-positive delta.
3. If neither is available or the resolved speed is ≤ 0, emit nothing.

Receiving position history
--------------------------
The :class:`~etaval.domain.interfaces.Predictor` Protocol passes only
``vehicle_state`` (the latest sample).  To use history the engine passes it as
the optional keyword-only argument ``position_history``.  The argument has a
default of ``None`` so calling ``predict(trip, upcoming, state, tu)`` — i.e.
without history — still satisfies the Protocol and gracefully falls back to the
reported speed.
"""

from __future__ import annotations

from typing import Any

from etaval.domain.models import Stop, TripInstance, VehicleState
from etaval.predictors.base import StopPrediction


class ConstantSpeedPredictor:
    """etaval's built-in white-box baseline; zero external dependencies.

    ``predictor_id = "baseline:constant_speed"``

    Estimates the vehicle's current speed once per tick, then projects a linear
    ETA to every upcoming stop using ``dist_remaining_m / speed``.

    See module docstring for speed-estimation priority and history passing.
    """

    predictor_id: str = "baseline:constant_speed"

    #: Declares that this predictor wants the engine to pass
    #: ``position_history`` — see ``Predictor.accepts_position_history``.
    accepts_position_history: bool = True

    def predict(
        self,
        trip: TripInstance,
        upcoming_stops: list[Stop],
        vehicle_state: VehicleState,
        trip_updates: Any = None,
        *,
        position_history: list[VehicleState] | None = None,
    ) -> list[StopPrediction]:
        """Emit one StopPrediction per upcoming stop with a computable ETA.

        Args:
            trip:
                The active TripInstance (carried for context; not inspected directly).
            upcoming_stops:
                Ordered stops ahead of the vehicle, each with a valid ``progress_m``;
                stops whose ``progress_m`` is ``None`` are skipped.
            vehicle_state:
                Latest vehicle position sample.  ``progress_m`` must be set for
                any stop distances to be computable; ``speed`` is used as a
                fallback when history is unavailable.
            trip_updates:
                Not used by this predictor; accepted for Protocol compatibility.
            position_history:
                Optional list of :class:`VehicleState` records in chronological
                order (oldest first, most-recent last).  When provided and at
                least two entries have valid ``progress_m`` and distinct epochs,
                speed is derived from ``Δprogress_m / Δt``.  Defaults to ``None``.

        Returns:
            One :class:`StopPrediction` per upcoming stop for which
            ``dist_remaining_m > 0`` and speed could be estimated.  Empty if
            speed cannot be determined or is ≤ 0.
        """
        now: float = vehicle_state.epoch
        vehicle_progress_m = vehicle_state.progress_m

        speed = self._estimate_speed(vehicle_state, position_history)
        if speed is None or speed <= 0.0:
            return []

        predictions: list[StopPrediction] = []
        n_upcoming = len(upcoming_stops)

        for idx, stop in enumerate(upcoming_stops):
            if stop.progress_m is None or vehicle_progress_m is None:
                continue
            dist_remaining_m = stop.progress_m - vehicle_progress_m
            if dist_remaining_m <= 0.0:
                continue

            predicted_arrival_epoch = now + dist_remaining_m / speed
            stops_remaining = n_upcoming - idx

            predictions.append(
                StopPrediction(
                    stop_id=stop.stop_id,
                    stop_sequence=stop.stop_sequence,
                    predicted_arrival_epoch=predicted_arrival_epoch,
                    made_at_epoch=now,
                    dist_remaining_m=dist_remaining_m,
                    stops_remaining=stops_remaining,
                )
            )

        return predictions

    @staticmethod
    def _estimate_speed(
        vehicle_state: VehicleState,
        position_history: list[VehicleState] | None,
    ) -> float | None:
        """Return estimated speed in m/s, or ``None`` if not determinable.

        Attempt 1: ``Δprogress_m / Δt`` from the two most-recent history entries
        that have valid ``progress_m``.  Returns ``None`` for that attempt if:
        - fewer than two such entries exist,
        - ``Δt ≤ 0`` (same epoch or reversed),
        - ``Δprogress_m < 0`` (vehicle moving backward — pathological; skip).

        Attempt 2: ``vehicle_state.speed`` (must be > 0).
        """
        # Attempt 1: history-based Δprogress_m / Δt
        if position_history is not None and len(position_history) >= 2:
            with_progress = [s for s in position_history if s.progress_m is not None]
            if len(with_progress) >= 2:
                s1 = with_progress[-2]
                s2 = with_progress[-1]
                pm1 = s1.progress_m
                pm2 = s2.progress_m
                dt = s2.epoch - s1.epoch
                # Guarded narrowing: with_progress filter guarantees non-None above.
                if pm1 is not None and pm2 is not None and dt > 0.0:
                    dp = pm2 - pm1
                    if dp >= 0.0:
                        return dp / dt

        # Attempt 2: reported speed from the feed
        if vehicle_state.speed is not None and vehicle_state.speed > 0.0:
            return vehicle_state.speed

        return None
