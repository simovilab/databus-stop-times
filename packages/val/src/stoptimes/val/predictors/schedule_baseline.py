"""ScheduleBaselinePredictor (predictor_id="baseline:schedule"). Phase A4.

Schedule-anchored ETA baseline: for each upcoming stop,

    predicted_arrival_epoch = scheduled_time(stop) + offset

where ``offset = now - scheduled_time_at_current_progress``. The vehicle's
"scheduled time at current progress" is estimated by linearly interpolating
the scheduled times of the FIRST TWO upcoming stops over the vehicle's shape
progress (``progress_m``).

KEY PROPERTY — constant-offset cancellation
--------------------------------------------
Only *differences* of scheduled times enter the arithmetic. Substituting the
interpolation into the top formula collapses to::

    predicted = now + (stop_sched_sec - progress_sched_sec)

so ``scheduled_time()`` never needs (and never adds) any service-day epoch or
timezone anchor -- it returns the raw GTFS "seconds since service-day
midnight" value straight out of ``stop_times.txt``. Adding any constant ``C``
to every scheduled time in the feed leaves every prediction bit-for-bit
unchanged, because ``C`` appears once with a ``+`` sign (``stop_sched_sec``)
and once with a ``-`` sign (``progress_sched_sec``) in the same difference —
see ``tests/test_schedule_baseline.py::test_constant_offset_cancellation``.

Past-midnight parsing
----------------------
GTFS ``stop_times.txt`` times may exceed ``24:00:00`` for trips that run past
midnight (e.g. ``"25:15:00"``). ``_parse_gtfs_time`` accepts any non-negative
``HH:MM:SS`` string (``HH`` unbounded) and returns the raw seconds-since
service-day-midnight integer.

Receiving the static schedule
-------------------------------
The engine has no built-in concept of "predictors that need GTFS static
data" — instead this predictor duck-types an optional hook,
``attach_static_loader(loader)``, that the run assembly (see
``stoptimes.val.engine.assembly._attach_static_loader``) calls with the run's
``GtfsStaticLoader`` for any predictor that exposes it. Only ``loader.stop_times``
(a DataFrame with ``trip_id``/``stop_id``/``stop_sequence``/``arrival_time``/
``departure_time`` columns) is required, so the hook accepts anything
GtfsStaticLoader-shaped -- real or a test double.

Starvation
----------
Returns ``[]`` (never raises) whenever a usable prediction cannot be formed:

- No static loader has been attached.
- The trip is not present in ``stop_times`` (or its schedule rows have no
  parseable ``arrival_time``/``departure_time``).
- Fewer than two upcoming stops are given, or the first two upcoming stops
  lack a matching schedule entry or a known ``progress_m``.
- The vehicle's own ``progress_m`` is unknown.
- The first two upcoming stops have equal ``progress_m`` (degenerate
  interpolation -- division by zero avoided).
"""

from __future__ import annotations

from typing import Any, Protocol

from stoptimes.core.domain.models import Stop, TripInstance, VehicleState
from stoptimes.val.predictors.base import StopPrediction


def _parse_gtfs_time(value: Any) -> int | None:
    """Parse a GTFS ``HH:MM:SS`` time string into seconds since service-day midnight.

    Accepts ``HH >= 24`` (past-midnight trips, e.g. ``"25:15:00"`` -> ``90900``).

    Returns:
        The integer second count, or ``None`` if *value* is missing, blank,
        NaN, or not a well-formed ``HH:MM:SS`` string.
    """
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    parts = text.split(":")
    if len(parts) != 3:
        return None
    try:
        hours, minutes, seconds = (int(p) for p in parts)
    except ValueError:
        return None
    if hours < 0 or minutes < 0 or seconds < 0:
        return None
    return hours * 3600 + minutes * 60 + seconds


class _StaticLoaderLike(Protocol):
    """Structural shape this predictor needs from a GtfsStaticLoader-like object."""

    # pandas.DataFrame: trip_id, stop_id, stop_sequence, arrival_time, departure_time
    stop_times: Any


class ScheduleBaselinePredictor:
    """Schedule-anchored ETA baseline; requires an attached static GTFS loader.

    ``predictor_id = "baseline:schedule"``

    See module docstring for the ETA formula, the constant-offset-cancellation
    property, past-midnight parsing, and starvation behavior.
    """

    predictor_id: str = "baseline:schedule"

    #: This predictor derives its own ETA from stop_times.txt, not from
    #: recent Δprogress/Δt — it does not need position_history.
    accepts_position_history: bool = False

    def __init__(self) -> None:
        self._loader: _StaticLoaderLike | None = None
        self._schedule_cache: dict[str, dict[int, int]] = {}

    def attach_static_loader(self, loader: Any) -> None:
        """Receive the run's static GTFS loader (duck-typed: needs ``.stop_times``).

        Called by the engine assembly (``build_live_run`` / ``build_replay_run``)
        for every predictor that exposes this method — see
        ``stoptimes.val.engine.assembly._attach_static_loader``. Resets the per-trip
        schedule cache so a new loader is never mixed with stale entries.
        """
        self._loader = loader
        self._schedule_cache = {}

    # ------------------------------------------------------------------
    # Schedule lookup (memoized per trip_id)
    # ------------------------------------------------------------------

    def _schedule_for_trip(self, trip_id: str) -> dict[int, int]:
        """Return ``{stop_sequence: scheduled_seconds}`` for *trip_id*, memoized.

        Empty dict when no loader is attached, ``stop_times`` is empty, the
        trip has no rows, or no row has a parseable arrival/departure time.
        """
        if trip_id in self._schedule_cache:
            return self._schedule_cache[trip_id]

        schedule: dict[int, int] = {}
        loader = self._loader
        stop_times = getattr(loader, "stop_times", None) if loader is not None else None
        if stop_times is not None and not stop_times.empty:
            rows = stop_times[stop_times["trip_id"] == trip_id]
            for _, row in rows.iterrows():
                seq_raw = row.get("stop_sequence")
                if seq_raw is None:
                    continue
                try:
                    if seq_raw != seq_raw:  # NaN check (NaN != NaN)
                        continue
                    stop_sequence = int(seq_raw)
                except (TypeError, ValueError):
                    continue

                sched_sec = _parse_gtfs_time(row.get("arrival_time"))
                if sched_sec is None:
                    sched_sec = _parse_gtfs_time(row.get("departure_time"))
                if sched_sec is None:
                    continue

                schedule[stop_sequence] = sched_sec

        self._schedule_cache[trip_id] = schedule
        return schedule

    # ------------------------------------------------------------------
    # Predictor protocol
    # ------------------------------------------------------------------

    def predict(
        self,
        trip: TripInstance,
        upcoming_stops: list[Stop],
        vehicle_state: VehicleState,
        trip_updates: Any = None,
    ) -> list[StopPrediction]:
        """Emit one StopPrediction per upcoming stop with a matching schedule entry.

        Args:
            trip: The active TripInstance; ``trip.trip_id`` selects the
                ``stop_times.txt`` rows to use.
            upcoming_stops: Ordered stops ahead of the vehicle, each ideally
                carrying ``progress_m`` (assigned by ``assign_stops_monotonic``).
            vehicle_state: Latest vehicle position sample; ``progress_m`` must
                be set to interpolate the vehicle's position in schedule-time.
            trip_updates: Not used by this predictor; accepted for Protocol
                compatibility.

        Returns:
            One :class:`StopPrediction` per upcoming stop whose
            ``stop_sequence`` has a parseable scheduled time. Empty on any
            starvation condition (see module docstring).
        """
        if self._loader is None:
            return []
        if len(upcoming_stops) < 2:
            return []

        vehicle_progress_m = vehicle_state.progress_m
        if vehicle_progress_m is None:
            return []

        schedule = self._schedule_for_trip(trip.trip_id)
        if not schedule:
            return []

        stop0, stop1 = upcoming_stops[0], upcoming_stops[1]
        if stop0.progress_m is None or stop1.progress_m is None:
            return []

        sched0 = schedule.get(stop0.stop_sequence)
        sched1 = schedule.get(stop1.stop_sequence)
        if sched0 is None or sched1 is None:
            return []

        progress_span = stop1.progress_m - stop0.progress_m
        if progress_span == 0.0:
            return []

        frac = (vehicle_progress_m - stop0.progress_m) / progress_span
        # scheduled seconds-of-day estimated at the vehicle's current progress;
        # only ever used in a *difference* below, so any constant baked into
        # every entry of `schedule` cancels exactly.
        progress_sched_sec = sched0 + frac * (sched1 - sched0)

        now = vehicle_state.epoch
        n_upcoming = len(upcoming_stops)
        predictions: list[StopPrediction] = []
        for idx, stop in enumerate(upcoming_stops):
            sched_sec = schedule.get(stop.stop_sequence)
            if sched_sec is None:
                continue

            predicted_arrival_epoch = now + (sched_sec - progress_sched_sec)
            dist_remaining_m = (
                stop.progress_m - vehicle_progress_m if stop.progress_m is not None else None
            )
            predictions.append(
                StopPrediction(
                    stop_id=stop.stop_id,
                    stop_sequence=stop.stop_sequence,
                    predicted_arrival_epoch=predicted_arrival_epoch,
                    made_at_epoch=now,
                    dist_remaining_m=dist_remaining_m,
                    stops_remaining=n_upcoming - idx,
                )
            )

        return predictions
