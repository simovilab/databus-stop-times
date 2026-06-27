"""Engine runtime (asyncio tick loop). Phase P5.

Feed-source seam
----------------
The engine never calls httpx directly.  Instead it consumes FeedSnapshot objects
from a FeedSource async iterator.  Two implementations are provided:

- LiveFeedSource: wraps VehiclePositionsPoller + TripUpdatesPoller (real HTTP).
- ReplayFeedSource: iterates an in-memory list of pre-built FeedSnapshots with no
  real sleeping, so test_engine_replay.py runs fully offline.

Tick loop
---------
Each tick:
1. Get next FeedSnapshot (parsed VP records + TU records).
2. tracker.update(vp_records, now=header_timestamp).
3. For each active trip: lazily build & cache geometry (polyline + stop progress_m).
4. Enrich position history: project each raw VehicleState onto the polyline to
   obtain a progress_m-annotated history.
5. Run every configured predictor → StopPredictions → PredictionSamples.
   Persist via storage.save_samples; keep in-memory for incremental scoring.
6. Run every configured detector → ActualArrivals.
   Deduplicate by (trip_key, stop_id, method); persist new arrivals.
7. For primary-method arrivals that newly resolved, score all stored samples for
   that (trip_key, stop_id) → ValidationResults; persist.
8. tracker.expire(now, max_age_s=300).

Public API
----------
    summary = await run_engine(feed_source, static_loader, predictors, detectors,
                                storage, source_name="mbta")
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from etaval.domain.models import (
    ActualArrival,
    PredictionSample,
    ValidationResult,
    VehicleState,
)
from etaval.feeds.gtfsrt_json import (
    TripUpdateRecord,
    TripUpdatesPoller,
    VehiclePositionRecord,
    VehiclePositionsPoller,
)
from etaval.feeds.static import GtfsStaticLoader
from etaval.feeds.tracker import ActiveTripTracker
from etaval.predictors.gtfsrt_tripupdates import TripUpdateInput
from etaval.scoring.scorer import Scorer
from etaval.spatial.polyline import (
    PolylinePoint,
    assign_stops_monotonic,
    build_polyline,
    project_point_to_polyline,
    upcoming_stops,
)
from etaval.storage.sqlite import SqliteStorage

# ── Feed-source seam ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FeedSnapshot:
    """One tick's worth of parsed feed data — the unit passed through the seam.

    Using already-parsed records (rather than raw JSON dicts) keeps the engine
    testable without needing to re-parse inside the loop and lets
    ReplayFeedSource build snapshots directly from parse_vp_feed / parse_tu_feed.
    """

    header_timestamp: float            # canonical tick time (from VP feed)
    vp_records: list[VehiclePositionRecord]
    tu_header_timestamp: float
    tu_records: list[TripUpdateRecord]


class LiveFeedSource:
    """Async iterator that polls live GTFS-RT HTTP feeds each iteration.

    Usage::

        source = LiveFeedSource(
            VehiclePositionsPoller(vp_url),
            TripUpdatesPoller(tu_url),
        )
        async for snapshot in source:
            ...
    """

    def __init__(
        self,
        vp_poller: VehiclePositionsPoller,
        tu_poller: TripUpdatesPoller,
    ) -> None:
        self._vp = vp_poller
        self._tu = tu_poller

    def __aiter__(self) -> LiveFeedSource:
        return self

    async def __anext__(self) -> FeedSnapshot:
        vp_result, tu_result = await asyncio.gather(
            self._vp.poll(),
            self._tu.poll(),
        )
        return FeedSnapshot(
            header_timestamp=vp_result.header_timestamp,
            vp_records=vp_result.records,
            tu_header_timestamp=tu_result.header_timestamp,
            tu_records=tu_result.records,
        )


class ReplayFeedSource:
    """Async iterator that replays a pre-built sequence of FeedSnapshots.

    Consumes the snapshot list to exhaustion with no real sleeping — the engine
    loop runs fully offline and deterministically, which is what
    test_engine_replay.py depends on.

    Usage::

        source = ReplayFeedSource(snapshots)
        async for snapshot in source:
            ...
    """

    def __init__(self, snapshots: list[FeedSnapshot]) -> None:
        self._snapshots = list(snapshots)
        self._idx = 0

    def __aiter__(self) -> ReplayFeedSource:
        return self

    async def __anext__(self) -> FeedSnapshot:
        if self._idx >= len(self._snapshots):
            raise StopAsyncIteration
        snap = self._snapshots[self._idx]
        self._idx += 1
        return snap


# ── Engine output ─────────────────────────────────────────────────────────────


@dataclass
class RunSummary:
    """Counts returned by run_engine on completion."""

    ticks: int = 0
    samples_saved: int = 0
    arrivals_saved: int = 0
    results_saved: int = 0


@dataclass(frozen=True)
class VehiclePing:
    """Position ping for one active trip, emitted each tick."""

    trip_key: str
    route_id: str
    lat: float
    lon: float
    progress_m: float | None


@dataclass(frozen=True)
class TickEvent:
    """Event fired once per engine tick when ``on_tick`` is provided.

    Cumulative counts (``samples_saved``, ``arrivals_saved``, ``results_saved``)
    match the RunSummary totals *at the end of this tick*.  ``new_results``
    contains only the ValidationResults produced *this tick*.  ``vehicles`` is
    one VehiclePing per active trip processed this tick.
    """

    tick: int                              # 1-based tick counter
    header_timestamp: float                # VP feed header_timestamp for this tick
    samples_saved: int                     # cumulative summary.samples_saved
    arrivals_saved: int                    # cumulative summary.arrivals_saved
    results_saved: int                     # cumulative summary.results_saved
    new_results: list[ValidationResult]    # THIS tick's scored results
    vehicles: list[VehiclePing]            # one ping per active trip this tick


# ── Geometry cache entry ──────────────────────────────────────────────────────

@dataclass
class _GeometryEntry:
    polyline: list[PolylinePoint]
    stops_with_progress: list[Any]  # list[Stop] with progress_m assigned


# ── Main entry point ──────────────────────────────────────────────────────────


async def run_engine(
    feed_source: LiveFeedSource | ReplayFeedSource,
    static_loader: GtfsStaticLoader,
    predictors: list[Any],
    detectors: list[Any],
    storage: SqliteStorage,
    *,
    primary_method: str = "shape_distance",
    duration_s: float | None = None,
    max_ticks: int | None = None,
    poll_interval_s: float = 15.0,
    source_name: str = "",
    on_tick: Callable[[TickEvent], Awaitable[None]] | None = None,
) -> RunSummary:
    """Run the validation engine tick loop.

    In **live** mode (LiveFeedSource) the loop sleeps ``poll_interval_s`` between
    ticks and stops after ``duration_s`` wall-clock seconds.  Pass
    ``duration_s=None`` with ``max_ticks`` to cap by tick count instead.

    In **replay** mode (ReplayFeedSource) the loop consumes the snapshot sequence
    to exhaustion (or up to ``max_ticks``) with no real sleeping.

    Args:
        feed_source:    Async iterator of FeedSnapshots.
        static_loader:  Loaded GTFS static zip (provides trip_geometry).
        predictors:     List of Predictor instances to run each tick.
        detectors:      List of GroundTruthDetector instances to run each tick.
        storage:        SqliteStorage for persistence.
        primary_method: Detector method used for scoring (default "shape_distance").
        duration_s:     Max wall-clock seconds to run (live mode).  None = unlimited.
        max_ticks:      Max ticks to process.  None = unlimited.
        poll_interval_s: Sleep between ticks in live mode.
        source_name:    Source label forwarded to ActiveTripTracker.

    Returns:
        :class:`RunSummary` with counts of saved rows.
    """
    is_live = isinstance(feed_source, LiveFeedSource)
    scorer = Scorer(primary_method=primary_method)
    tracker = ActiveTripTracker(source=source_name)

    # Geometry cache keyed by trip_id (shape_id starts "" in tracker).
    _geo_cache: dict[str, _GeometryEntry] = {}

    # In-memory samples for incremental scoring.
    # Key: (trip_key, stop_id)
    _samples: dict[tuple[str, str], list[PredictionSample]] = {}

    # Seen arrivals: (trip_key, stop_id, method) — dedup across ticks.
    _seen_arrivals: set[tuple[str, str, str]] = set()

    # Route-id map for ValidationResult enrichment.
    _route_id_map: dict[str, str] = {}

    summary = RunSummary()
    start_wall = time.monotonic()

    async for snapshot in feed_source:
        # ── Duration / tick cap ───────────────────────────────────────────────
        if max_ticks is not None and summary.ticks >= max_ticks:
            break
        if is_live and duration_s is not None:
            if time.monotonic() - start_wall >= duration_s:
                break

        now = snapshot.header_timestamp

        # ── 1. Update tracker ─────────────────────────────────────────────────
        tracker.update(snapshot.vp_records, now=now)

        # ── 2. Index TU records by trip_id ────────────────────────────────────
        tu_by_trip_id: dict[str, TripUpdateRecord] = {
            r.trip_id: r for r in snapshot.tu_records
        }

        # ── 3–7. Per-trip processing ──────────────────────────────────────────
        tick_samples: list[PredictionSample] = []
        tick_arrivals: list[ActualArrival] = []
        tick_results = []
        tick_vehicles: list[VehiclePing] = []

        for trip in tracker.active_trips():
            tracker_key = (trip.trip_id, trip.vehicle_id)

            # Keep route_id map current.
            _route_id_map[trip.trip_key] = trip.route_id

            # ── Geometry (lazy, cached) ───────────────────────────────────────
            if trip.trip_id not in _geo_cache:
                try:
                    shape_pts, raw_stops = static_loader.trip_geometry(trip.trip_id)
                except ValueError:
                    continue  # trip not in static data — skip gracefully
                if len(shape_pts) < 2:
                    continue
                polyline = build_polyline(shape_pts)
                stops_with_progress = assign_stops_monotonic(raw_stops, polyline)
                _geo_cache[trip.trip_id] = _GeometryEntry(
                    polyline=polyline,
                    stops_with_progress=stops_with_progress,
                )

            geo = _geo_cache[trip.trip_id]
            polyline = geo.polyline
            stops = geo.stops_with_progress

            # ── Enrich position history with progress_m ───────────────────────
            raw_history = tracker.position_history(tracker_key)
            enriched_history: list[VehicleState] = []
            for state in raw_history:
                try:
                    proj = project_point_to_polyline(state.lat, state.lon, polyline)
                    enriched = state.model_copy(update={"progress_m": proj.progress_m})
                except ValueError:
                    enriched = state
                enriched_history.append(enriched)

            if not enriched_history:
                continue

            latest = enriched_history[-1]
            tick_vehicles.append(
                VehiclePing(
                    trip_key=trip.trip_key,
                    route_id=trip.route_id,
                    lat=latest.lat,
                    lon=latest.lon,
                    progress_m=latest.progress_m,
                )
            )
            current_progress_m = latest.progress_m if latest.progress_m is not None else 0.0
            upcoming = upcoming_stops(current_progress_m, stops)

            # ── TripUpdateInput for gtfsrt predictor ──────────────────────────
            tu_record = tu_by_trip_id.get(trip.trip_id)
            trip_update_input: TripUpdateInput | None = (
                TripUpdateInput(
                    header_timestamp=snapshot.tu_header_timestamp,
                    stop_time_updates=tu_record.stop_time_updates,
                )
                if tu_record is not None
                else None
            )

            # ── 5. Run predictors ─────────────────────────────────────────────
            for predictor in predictors:
                try:
                    if hasattr(predictor, "predict"):
                        # Constant-speed predictor accepts position_history kwarg.
                        import inspect
                        sig = inspect.signature(predictor.predict)
                        if "position_history" in sig.parameters:
                            preds = predictor.predict(
                                trip,
                                upcoming,
                                latest,
                                trip_update_input,
                                position_history=enriched_history,
                            )
                        else:
                            preds = predictor.predict(
                                trip, upcoming, latest, trip_update_input
                            )
                    else:
                        continue
                except Exception:
                    continue

                for pred in preds:
                    sample = pred.to_sample(
                        trip_key=trip.trip_key,
                        predictor_id=predictor.predictor_id,
                        vehicle_progress_m_at_made=current_progress_m,
                    )
                    tick_samples.append(sample)
                    # Keep in memory for incremental scoring.
                    key = (sample.trip_key, sample.stop_id)
                    _samples.setdefault(key, []).append(sample)

            # ── 6. Run detectors ──────────────────────────────────────────────
            for detector in detectors:
                try:
                    arrivals = detector.detect(trip, enriched_history, stops)
                except Exception:
                    continue

                for arrival in arrivals:
                    dedup_key = (arrival.trip_key, arrival.stop_id, arrival.method)
                    if dedup_key in _seen_arrivals:
                        continue
                    _seen_arrivals.add(dedup_key)
                    tick_arrivals.append(arrival)

                    # ── 7. Incremental scoring for primary-method arrivals ─────
                    if arrival.method == primary_method:
                        route_id = _route_id_map.get(arrival.trip_key)
                        stored = _samples.get((arrival.trip_key, arrival.stop_id), [])
                        new_results = scorer.score_stop(
                            trip_key=arrival.trip_key,
                            stop_id=arrival.stop_id,
                            arrival=arrival,
                            samples=stored,
                            route_id=route_id,
                        )
                        tick_results.extend(new_results)

        # ── Persist ───────────────────────────────────────────────────────────
        if tick_samples:
            storage.save_samples(tick_samples)
            summary.samples_saved += len(tick_samples)

        if tick_arrivals:
            storage.save_arrivals(tick_arrivals)
            summary.arrivals_saved += len(tick_arrivals)

        if tick_results:
            storage.save_results(tick_results)
            summary.results_saved += len(tick_results)

        # ── 8. Expire stale trips (5-minute window) ───────────────────────────
        tracker.expire(now=now, max_age_s=300.0)

        summary.ticks += 1

        # ── Stream tick event (before the live sleep so events fire promptly) ──
        if on_tick is not None:
            await on_tick(
                TickEvent(
                    tick=summary.ticks,
                    header_timestamp=now,
                    samples_saved=summary.samples_saved,
                    arrivals_saved=summary.arrivals_saved,
                    results_saved=summary.results_saved,
                    new_results=tick_results,
                    vehicles=tick_vehicles,
                )
            )

        # Sleep between ticks only in live mode.
        if is_live:
            await asyncio.sleep(poll_interval_s)

    return summary
