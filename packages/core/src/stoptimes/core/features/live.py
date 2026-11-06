"""Live feature assembly: the one place the study's 19 servable-plus features are computed.

Training, serving and ``val`` all call this module, so a column means the same thing
everywhere.  Inputs are only GTFS static (schedule, stop progress along the shape) and
VehiclePositions (through a :class:`~stoptimes.core.features.trip_state.TripState`),
plus two artifact sidecar assets (segment statistics, route vocabulary).

Two entry points compute the same numbers:

* :func:`assemble_live_features` - one prediction time ``t`` for one trip, straight from the
  F1-F3 primitives (the serving path).
* :func:`assemble_live_batch` - many prediction times for one trip at once, vectorised (the
  training path).  Parity with the first is a tested guarantee (replaying pings one by one
  through ``update_trip_state`` gives bit-identical features).

Definitions at prediction time ``t`` (``ping`` = latest valid ping with epoch <= ``t``,
``p`` its forward-matched progress, ``j0 = searchsorted(stop_progress, p, 'right')``,
target stops ``j >= j0``):

``distance_to_stop = S_j - p``; ``stops_ahead = j - j0``; ``minute_of_day`` from ``t`` in the
agency timezone; ``delay_now = ping_ts - sched(p)``; ``sched_rem = sched(j) - sched(p)``;
``sched_rem_t = sched(j) - t``; ``delay_last`` from the last passed stop whose arrival was
detectable at ``ping_ts``; ``prior_rem`` from the segment statistics; ``route_code`` the
vocabulary index of the route.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd

from stoptimes.core.features.identity import RouteVocabulary
from stoptimes.core.features.prior import (
    SegmentStats,
    daytype_of,
    local_hour,
    local_offset_s,
    prior_remaining,
)
from stoptimes.core.features.schedule import (
    TripSchedule,
    scheduled_at_progress,
    scheduled_epochs,
    schedule_features,
)
from stoptimes.core.features.trip_state import (
    ForwardMatcher,
    TripState,
    delay_last,
)

TEMPORAL_FEATURES: tuple[str, ...] = (
    "hour",
    "day_of_week",
    "is_weekend",
    "is_holiday",
    "is_peak_hour",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
)
"""The nine registry temporal columns the study's model uses (``time_of_day_bin`` is a string, excluded)."""

LIVE_FEATURES: tuple[str, ...] = (
    "distance_to_stop",
    "stops_ahead",
    "minute_of_day",
    *TEMPORAL_FEATURES,
    "route_code",
    "stop_sequence",
    "delay_now",
    "sched_rem",
    "sched_rem_t",
    "delay_last",
    "prior_rem",
)
"""The 19 ``SERVABLE_PLUS`` features, in the study's column order (``route_i`` is ``route_code``)."""

IDENTITY_COLUMNS: tuple[str, ...] = (
    "trip_id",
    "route_id",
    "service_date",
    "stop_id",
    "stop_idx",
    "t",
    "ping_ts",
)


class MissingAssetError(ValueError):
    """A requested feature needs a sidecar asset that was not supplied."""


@dataclass(frozen=True)
class TripContext:
    """Static facts about one trip instance (everything except the vehicle's history)."""

    trip_id: str
    route_id: str
    service_date: str  # YYYYMMDD, the trip's service day (not the ping's calendar date)
    tz: str
    stop_ids: tuple[str, ...]
    stop_sequences: tuple[int, ...]
    stop_progress_m: tuple[float, ...]
    sched_epochs: tuple[float, ...]
    holiday_region: str | None = None

    def __post_init__(self) -> None:
        n = len(self.stop_ids)
        if not (len(self.stop_sequences) == len(self.stop_progress_m) == len(self.sched_epochs) == n):
            raise ValueError("stop_ids, stop_sequences, stop_progress_m and sched_epochs must have equal length")

    @property
    def n_stops(self) -> int:
        return len(self.stop_ids)

    @property
    def trip_key(self) -> Hashable:
        return (self.trip_id, self.service_date)


@dataclass(frozen=True)
class LiveAssets:
    """Artifact sidecar assets the features read (either may be absent if no feature needs it)."""

    segment_stats: SegmentStats | None = None
    route_vocabulary: RouteVocabulary | None = None


class StaticTripSource(Protocol):
    """What :func:`build_trip_context` needs from a loaded static feed (``GtfsStaticLoader`` fits)."""

    def trip_geometry(self, trip_id: str) -> tuple[list[tuple[float, float, float | None]], list[Any]]: ...

    def trip_schedule(self, trip_id: str) -> TripSchedule: ...


def build_trip_context(
    static: StaticTripSource,
    trip_id: str,
    route_id: str,
    service_date: str,
    tz: str,
    *,
    holiday_region: str | None = None,
    geometry_cache: dict[str, tuple[ForwardMatcher, tuple[float, ...]]] | None = None,
) -> tuple[TripContext, ForwardMatcher]:
    """Context and shape matcher of a trip from GTFS static.

    Stop progress comes from the monotonic stop-to-shape assignment; the schedule is
    anchored per GTFS on ``service_date``.  ``geometry_cache`` (trip_id -> matcher and stop
    progress) lets a caller that builds the same trip for many days skip the shape work; it
    must only be shared between calls against the same static feed.  Raises ``ValueError`` for an unknown trip,
    a missing/short shape, or when schedule and geometry disagree about the stops.
    """
    from stoptimes.core.spatial.polyline import assign_stops_monotonic, build_polyline

    schedule = static.trip_schedule(trip_id)
    cached = geometry_cache.get(trip_id) if geometry_cache is not None else None
    if cached is None:
        shape_points, stops = static.trip_geometry(trip_id)
        if len(shape_points) < 2:
            raise ValueError(f"trip {trip_id!r} has no usable shape")
        polyline = build_polyline(shape_points)
        placed = assign_stops_monotonic(list(stops), polyline)
        if tuple(s.stop_id for s in placed) != schedule.stop_ids or tuple(
            s.stop_sequence for s in placed
        ) != schedule.stop_sequences:
            raise ValueError(f"trip {trip_id!r}: stops of the shape geometry and the schedule do not align")
        cached = (ForwardMatcher(polyline), tuple(float(s.progress_m) for s in placed))
        if geometry_cache is not None:
            geometry_cache[trip_id] = cached
    matcher, progress = cached
    ctx = TripContext(
        trip_id=str(trip_id),
        route_id=str(route_id),
        service_date=service_date,
        tz=tz,
        stop_ids=schedule.stop_ids,
        stop_sequences=schedule.stop_sequences,
        stop_progress_m=progress,
        sched_epochs=tuple(float(v) for v in scheduled_epochs(schedule, service_date, tz)),
        holiday_region=holiday_region,
    )
    return ctx, matcher


# ---------------------------------------------------------------------------
# Clock features (shared by both paths)
# ---------------------------------------------------------------------------


def minute_of_day(t: np.ndarray | float, service_date: str, tz: str) -> np.ndarray:
    """Minutes since local midnight, using the UTC offset at local noon of the service date."""
    off = local_offset_s(service_date, tz)
    return ((np.asarray(t, dtype=float) + off) % 86400.0) / 60.0


def temporal_columns(t: np.ndarray | Sequence[float], tz: str, holiday_region: str | None) -> pd.DataFrame:
    """The nine registry temporal features at epochs ``t`` (float columns), via the registry.

    Every temporal feature depends only on the local date and hour, so the registry runs
    once per distinct local hour and the result is broadcast back; the values equal the
    per-timestamp registry output exactly.
    """
    from stoptimes.core.features.registry import build as build_registry

    t = np.asarray(t, dtype=float)
    if t.size == 0:
        return pd.DataFrame({c: np.empty(0) for c in TEMPORAL_FEATURES})
    stamps = pd.to_datetime(t, unit="s", utc=True)
    local = stamps.tz_convert(tz)
    key = local.strftime("%Y%m%d%H")
    codes, uniq_idx = pd.factorize(pd.Series(key))
    first = pd.Series(np.arange(len(key))).groupby(codes).first().to_numpy()
    probe = pd.DataFrame({"_ts": [stamps[i] for i in first]})
    built = build_registry(
        list(TEMPORAL_FEATURES), probe, {"timestamp_column": "_ts", "tz": tz, "holiday_region": holiday_region}
    )
    small = built[list(TEMPORAL_FEATURES)].astype(float).to_numpy()
    return pd.DataFrame(small[codes], columns=list(TEMPORAL_FEATURES))


def _check_features(features: Sequence[str], assets: LiveAssets) -> None:
    unknown = [f for f in features if f not in LIVE_FEATURES]
    if unknown:
        raise ValueError(f"not live features: {unknown}; known: {list(LIVE_FEATURES)}")
    if "prior_rem" in features and assets.segment_stats is None:
        raise MissingAssetError("feature 'prior_rem' needs the segment_stats asset")
    if "route_code" in features and assets.route_vocabulary is None:
        raise MissingAssetError("feature 'route_code' needs the route_vocabulary asset")


def _empty_frame(features: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame({c: np.empty(0) for c in (*IDENTITY_COLUMNS, *features)})


def _route_code(assets: LiveAssets, route_id: str) -> float:
    assert assets.route_vocabulary is not None
    return float(assets.route_vocabulary.encode([route_id])[0])


# ---------------------------------------------------------------------------
# Serving path: one prediction time
# ---------------------------------------------------------------------------


def _latest_valid(state: TripState, t: float) -> tuple[float, float] | None:
    for epoch, progress, ok in reversed(state.pings):
        if ok and epoch <= t:
            return epoch, progress
    return None


def assemble_live_features(
    ctx: TripContext,
    assets: LiveAssets,
    state: TripState,
    t: float,
    *,
    features: Sequence[str] = LIVE_FEATURES,
) -> pd.DataFrame:
    """Feature rows for every downstream stop of one trip at prediction time ``t``.

    ``state`` holds the pings seen so far (those with epoch > ``t`` are ignored).  An empty
    frame comes back when the trip has no valid ping yet or no stop remains ahead.
    """
    features = tuple(features)
    _check_features(features, assets)
    if len(state.stop_progress_m) != ctx.n_stops:
        raise ValueError("trip state and trip context disagree on the number of stops")
    latest = _latest_valid(state, t)
    if latest is None:
        return _empty_frame(features)
    ping_ts, progress = latest
    sprog = np.asarray(ctx.stop_progress_m, dtype=float)
    sched = np.asarray(ctx.sched_epochs, dtype=float)
    j0 = int(np.searchsorted(sprog, progress, side="right"))
    idx = np.arange(j0, ctx.n_stops)
    if idx.size == 0:
        return _empty_frame(features)
    k = idx.size
    out: dict[str, Any] = {
        "trip_id": ctx.trip_id,
        "route_id": ctx.route_id,
        "service_date": ctx.service_date,
        "stop_id": np.asarray(ctx.stop_ids, dtype=object)[idx],
        "stop_idx": idx,
        "t": np.full(k, float(t)),
        "ping_ts": np.full(k, ping_ts),
    }
    sf = schedule_features(sprog, sched, progress, ping_ts, float(t), idx)
    cols: dict[str, np.ndarray] = {
        "distance_to_stop": sprog[idx] - progress,
        "stops_ahead": (idx - j0).astype(float),
        "minute_of_day": minute_of_day(np.full(k, float(t)), ctx.service_date, ctx.tz),
        "stop_sequence": np.asarray(ctx.stop_sequences, dtype=float)[idx],
        "delay_now": sf["delay_now"],
        "sched_rem": sf["sched_rem"],
        "sched_rem_t": sf["sched_rem_t"],
        "delay_last": np.full(k, delay_last(state, sched, progress, ping_ts)),
    }
    if "route_code" in features:
        cols["route_code"] = np.full(k, _route_code(assets, ctx.route_id))
    if "prior_rem" in features:
        assert assets.segment_stats is not None
        cols["prior_rem"] = prior_remaining(
            assets.segment_stats,
            ctx.route_id,
            ctx.stop_ids,
            sprog,
            progress,
            float(t),
            ctx.tz,
            ctx.service_date,
            idx,
        )
    if any(f in TEMPORAL_FEATURES for f in features):
        temporal = temporal_columns(np.full(k, float(t)), ctx.tz, ctx.holiday_region)
        for name in TEMPORAL_FEATURES:
            cols[name] = temporal[name].to_numpy()
    frame = pd.DataFrame(out)
    for name in features:
        frame[name] = cols[name]
    return frame


# ---------------------------------------------------------------------------
# Training path: many prediction times of one trip
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Instants:
    """Per-instant quantities of a batch (one entry per kept prediction time)."""

    t: np.ndarray
    ping_ts: np.ndarray
    progress: np.ndarray
    j0: np.ndarray
    keep: np.ndarray = field(repr=False)  # indices into the requested times that were kept


def _resolve_instants(ctx: TripContext, state: TripState, times: np.ndarray) -> _Instants:
    valid = [(e, p) for e, p, ok in state.pings if ok]
    if not valid:
        z = np.empty(0)
        return _Instants(z, z, z, z.astype(int), z.astype(int))
    eps = np.array([v[0] for v in valid])
    prog = np.array([v[1] for v in valid])
    pos = np.searchsorted(eps, times, side="right") - 1
    keep = np.flatnonzero(pos >= 0)
    pos = pos[keep]
    sprog = np.asarray(ctx.stop_progress_m, dtype=float)
    j0 = np.searchsorted(sprog, prog[pos], side="right")
    has_rows = j0 < ctx.n_stops
    keep, pos, j0 = keep[has_rows], pos[has_rows], j0[has_rows]
    return _Instants(times[keep], eps[pos], prog[pos], j0.astype(int), keep)


def _delay_last_batch(ctx: TripContext, state: TripState, inst: _Instants) -> np.ndarray:
    arr_epoch, detect, _ = state.arrival_arrays()
    sched = np.asarray(ctx.sched_epochs, dtype=float)
    n = ctx.n_stops
    ks = np.arange(n)[None, :]
    ok = (ks < inst.j0[:, None]) & np.isfinite(arr_epoch)[None, :] & (detect[None, :] <= inst.ping_ts[:, None])
    has = ok.any(axis=1)
    last = n - 1 - np.argmax(ok[:, ::-1], axis=1)
    return np.where(has, arr_epoch[last] - sched[last], np.nan)


def _prior_batch(ctx: TripContext, stats: SegmentStats, inst: _Instants) -> np.ndarray:
    """``prior_remaining`` for many instants of one trip; same arithmetic, vectorised over instants."""
    sprog = np.asarray(ctx.stop_progress_m, dtype=float)
    n = ctx.n_stops
    n_inst = inst.t.size
    eta = np.full((n_inst, n), np.nan)
    if n < 2 or n_inst == 0:
        return eta
    off = local_offset_s(ctx.service_date, ctx.tz)
    dtp = daytype_of(ctx.service_date)
    route = str(ctx.route_id)
    seg_ids = stats.segment_ids(
        np.full(n - 1, route), np.asarray(ctx.stop_ids, dtype=str)[:-1], np.asarray(ctx.stop_ids, dtype=str)[1:]
    )
    seg_len = np.diff(sprog)

    def seg_time(k: int, when: np.ndarray) -> np.ndarray:
        return stats.lookup_ids(seg_ids[k], dtp, local_hour(when, off), seg_len[k], route)

    s = inst.j0 - 1
    cur = np.zeros(n_inst)
    has_cur = s >= 0
    if has_cur.any():
        sc = s[has_cur]
        span = np.maximum(sprog[sc + 1] - sprog[sc], 1.0)
        frac = np.clip((inst.progress[has_cur] - sprog[sc]) / span, 0.0, 1.0)
        h = np.empty(n_inst)
        for k in np.unique(sc):
            m = has_cur & (s == k)
            h[m] = seg_time(int(k), inst.t[m])
        cur[has_cur] = (1.0 - frac) * h[has_cur]
    rows = np.arange(n_inst)
    eta[rows, inst.j0] = cur
    for j in range(1, n):
        g = inst.j0 < j
        if not g.any():
            continue
        prev = eta[g, j - 1]
        eta[g, j] = prev + seg_time(j - 1, inst.t[g] + prev)
    return eta


def assemble_live_batch(
    ctx: TripContext,
    assets: LiveAssets,
    state: TripState,
    times: Sequence[float] | np.ndarray,
    *,
    features: Sequence[str] = LIVE_FEATURES,
) -> pd.DataFrame:
    """Feature rows at many prediction times of one trip, from the trip's full history.

    Row-for-row (and bit-for-bit) what :func:`assemble_live_features` returns when the pings
    are replayed one at a time and asked at each ``t`` - causality is enforced the same way:
    an instant sees only valid pings with epoch <= ``t`` and arrivals detected by then.
    Times with no valid ping yet, or past the last stop, contribute no rows.  Rows are
    ordered by instant (input order), then stop.
    """
    features = tuple(features)
    _check_features(features, assets)
    if len(state.stop_progress_m) != ctx.n_stops:
        raise ValueError("trip state and trip context disagree on the number of stops")
    times = np.asarray(times, dtype=float)
    inst = _resolve_instants(ctx, state, times)
    if inst.t.size == 0:
        return _empty_frame(features)
    n = ctx.n_stops
    sprog = np.asarray(ctx.stop_progress_m, dtype=float)
    sched = np.asarray(ctx.sched_epochs, dtype=float)
    counts = n - inst.j0
    rep = np.repeat(np.arange(inst.t.size), counts)
    starts = np.cumsum(counts) - counts
    stop_idx = inst.j0[rep] + (np.arange(rep.size) - starts[rep])
    t_row = inst.t[rep]
    progress_row = inst.progress[rep]
    sched_now = np.asarray(scheduled_at_progress(sprog, sched, inst.progress))
    out: dict[str, Any] = {
        "trip_id": ctx.trip_id,
        "route_id": ctx.route_id,
        "service_date": ctx.service_date,
        "stop_id": np.asarray(ctx.stop_ids, dtype=object)[stop_idx],
        "stop_idx": stop_idx,
        "t": t_row,
        "ping_ts": inst.ping_ts[rep],
    }
    cols: dict[str, np.ndarray] = {
        "distance_to_stop": sprog[stop_idx] - progress_row,
        "stops_ahead": (stop_idx - inst.j0[rep]).astype(float),
        "minute_of_day": minute_of_day(t_row, ctx.service_date, ctx.tz),
        "stop_sequence": np.asarray(ctx.stop_sequences, dtype=float)[stop_idx],
        "delay_now": (inst.ping_ts - sched_now)[rep],
        "sched_rem": sched[stop_idx] - sched_now[rep],
        "sched_rem_t": sched[stop_idx] - t_row,
        "delay_last": _delay_last_batch(ctx, state, inst)[rep],
    }
    if "route_code" in features:
        cols["route_code"] = np.full(rep.size, _route_code(assets, ctx.route_id))
    if "prior_rem" in features:
        assert assets.segment_stats is not None
        cols["prior_rem"] = _prior_batch(ctx, assets.segment_stats, inst)[rep, stop_idx]
    if any(f in TEMPORAL_FEATURES for f in features):
        temporal = temporal_columns(t_row, ctx.tz, ctx.holiday_region)
        for name in TEMPORAL_FEATURES:
            cols[name] = temporal[name].to_numpy()
    frame = pd.DataFrame(out)
    for name in features:
        frame[name] = cols[name]
    return frame
