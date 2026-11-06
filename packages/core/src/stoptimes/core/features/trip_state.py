"""Per-trip memory for live features (track F3).

A live poll only shows the latest vehicle position; features such as
``delay_last`` need the trip's history.  This module is that memory.  It is
causal by construction: a state at time ``t`` is a function of the pings with
epoch <= ``t`` only, and each arrival carries the ``detect_epoch`` at which it
first became knowable.

Behaviour follows the study's ground-truth method (forward-only windowed map
matching; arrival = linear interpolation of the time progress first reaches a
stop, rejected if the bracketing valid-ping gap exceeds 60 s) so that online
replay and the batch function give identical arrivals.

Public API: ``Ping``, ``TripState``, ``StopArrival``, ``ForwardMatcher``,
``update_trip_state``, ``arrivals_from_pings``, ``delay_last``,
``TripStateStore``, ``InMemoryTripStateStore``.  All state updates return new
objects.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from stoptimes.core.spatial.polyline import PolylinePoint

M_PER_DEG: float = 6371000.0 * math.pi / 180.0
XT_MAX_M: float = 100.0  # farther than this from the allowed window: off-route, ignored
BACK_TOL_M: float = 60.0  # allowed backward GPS jitter
MAX_GAP_S: float = 60.0  # bracketing valid-ping gap above which an arrival is rejected
TIE_M: float = 20.0  # among near-equal cross-track candidates prefer earliest progress
WINDOW_SPEED_MPS: float = 35.0  # forward window grows with elapsed time at this speed...
WINDOW_BASE_M: float = 400.0  # ...from this base...
WINDOW_MAX_M: float = 3000.0  # ...up to this cap
MID_TRIP_TOL_M: float = 150.0  # first valid progress beyond first stop by more than this


@dataclass(frozen=True)
class Ping:
    """One vehicle position report."""

    epoch: float
    lat: float
    lon: float
    speed: float | None = None
    status: int | str | None = None
    current_stop_sequence: int | None = None


@dataclass(frozen=True)
class StopArrival:
    """Interpolated arrival at a stop and when it became knowable."""

    epoch: float
    detect_epoch: float  # epoch of the first ping after the crossing
    gap_s: float  # bracketing valid-ping gap (<= MAX_GAP_S)


# (epoch, progress_m, valid); progress is NaN when the ping was ignored.
MatchedPing = tuple[float, float, bool]


@dataclass(frozen=True)
class TripState:
    """Everything remembered about one trip instance."""

    trip_key: Hashable
    stop_progress_m: tuple[float, ...]
    pings: tuple[MatchedPing, ...]
    arrivals: tuple[StopArrival | None, ...]
    started_mid_trip: bool
    last_update_epoch: float

    def last_valid(self) -> MatchedPing | None:
        for m in reversed(self.pings):
            if m[2]:
                return m
        return None

    def latest_progress_m(self) -> float:
        m = self.last_valid()
        return math.nan if m is None else m[1]

    def finished_epoch(self) -> float | None:
        """Epoch of the first valid ping at/after the last stop, if any."""
        if not self.stop_progress_m:
            return None
        end = self.stop_progress_m[-1]
        for e, p, ok in self.pings:
            if ok and p >= end:
                return e
        return None

    def arrival_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(arrival_epoch, detect_epoch, gap_s)`` per stop, NaN where unknown."""
        n = len(self.arrivals)
        a, d, g = (np.full(n, np.nan) for _ in range(3))
        for i, x in enumerate(self.arrivals):
            if x is not None:
                a[i], d[i], g[i] = x.epoch, x.detect_epoch, x.gap_s
        return a, d, g


# ---------------------------------------------------------------------------
# Forward-only map matching
# ---------------------------------------------------------------------------


class ForwardMatcher:
    """Projects pings onto a polyline, constrained to a window ahead of the previous progress.

    Immutable once built; pass one instance around instead of the raw polyline
    to avoid rebuilding the numpy arrays on every ping.
    """

    def __init__(self, polyline: Sequence[PolylinePoint]) -> None:
        if len(polyline) < 2:
            raise ValueError("polyline must contain at least 2 points")
        self._lat = np.array([p.lat for p in polyline], dtype=float)
        self._lon = np.array([p.lon for p in polyline], dtype=float)
        self._cum = np.array([p.cum_m for p in polyline], dtype=float)
        self._nseg = len(polyline) - 1

    def _project(self, plat: float, plon: float, k0: int, k1: int) -> tuple[np.ndarray, np.ndarray]:
        lat, lon, cum = self._lat, self._lon, self._cum
        c = math.cos(math.radians(plat))
        ax = (lon[k0:k1] - plon) * c * M_PER_DEG
        ay = (lat[k0:k1] - plat) * M_PER_DEG
        bx = (lon[k0 + 1 : k1 + 1] - plon) * c * M_PER_DEG
        by = (lat[k0 + 1 : k1 + 1] - plat) * M_PER_DEG
        dx, dy = bx - ax, by - ay
        l2 = dx * dx + dy * dy
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(l2 > 0, -(ax * dx + ay * dy) / l2, 0.0)
        t = np.clip(t, 0.0, 1.0)
        xt = np.hypot(ax + t * dx, ay + t * dy)
        prog = cum[k0:k1] + t * (cum[k0 + 1 : k1 + 1] - cum[k0:k1])
        return prog, xt

    def _window(self, prev: float | None, dt_s: float, lost: bool) -> tuple[int, int]:
        if prev is None:
            lo_m, hi_m = 0.0, math.inf
        elif lost:
            lo_m, hi_m = prev - BACK_TOL_M, math.inf
        else:
            lo_m = prev - BACK_TOL_M
            hi_m = prev + min(WINDOW_SPEED_MPS * max(dt_s, 0.0) + WINDOW_BASE_M, WINDOW_MAX_M)
        k0 = max(int(np.searchsorted(self._cum, lo_m, side="right")) - 1, 0)
        k1 = (
            self._nseg
            if not math.isfinite(hi_m)
            else int(np.searchsorted(self._cum, hi_m, side="left"))
        )
        return k0, min(max(k1, k0 + 1), self._nseg)

    def match(
        self,
        prev_progress_m: float | None,
        lat: float,
        lon: float,
        *,
        dt_s: float = 0.0,
        lost: bool = True,
    ) -> float | None:
        """Progress in metres, or ``None`` if the ping is off-route / jumped backwards.

        ``prev_progress_m=None`` (first ping) projects globally.  ``lost`` means the
        previous ping was ignored, so the forward window is unbounded.
        ``dt_s`` is the time since the last valid ping.
        """
        k0, k1 = self._window(prev_progress_m, dt_s, lost)
        prog, xt = self._project(lat, lon, k0, k1)
        j = int(np.argmin(xt))
        if xt[j] > XT_MAX_M:
            return None
        if lost or prev_progress_m is None:
            j = int(np.flatnonzero(xt <= xt[j] + TIE_M)[0])
        if prev_progress_m is not None and prog[j] < prev_progress_m - BACK_TOL_M:
            return None
        return float(max(prog[j], prev_progress_m if prev_progress_m is not None else 0.0))


def _as_matcher(polyline: ForwardMatcher | Sequence[PolylinePoint]) -> ForwardMatcher:
    return polyline if isinstance(polyline, ForwardMatcher) else ForwardMatcher(polyline)


def _match_next(matcher: ForwardMatcher, pings: Sequence[MatchedPing], ping: Ping) -> float | None:
    last = next((m for m in reversed(pings) if m[2]), None)
    lost = not pings or not pings[-1][2]
    if last is None:
        return matcher.match(None, ping.lat, ping.lon)
    return matcher.match(last[1], ping.lat, ping.lon, dt_s=ping.epoch - last[0], lost=lost)


# ---------------------------------------------------------------------------
# Online update
# ---------------------------------------------------------------------------


def _empty_state(trip_key: Hashable, stops: tuple[float, ...]) -> TripState:
    return TripState(trip_key, stops, (), (None,) * len(stops), False, -math.inf)


def _crossed_arrivals(
    stops: tuple[float, ...], t0: float, p0: float, t1: float, p1: float
) -> dict[int, StopArrival]:
    """Arrivals for stops with p0 < S <= p1 between two consecutive valid pings."""
    gap = t1 - t0
    if gap > MAX_GAP_S or p1 - p0 <= 0:
        return {}
    lo = int(np.searchsorted(stops, p0, side="right"))
    hi = int(np.searchsorted(stops, p1, side="right"))
    return {j: StopArrival(t0 + (stops[j] - p0) / (p1 - p0) * gap, t1, gap) for j in range(lo, hi)}


def update_trip_state(
    state: TripState | None,
    ping: Ping,
    *,
    polyline: ForwardMatcher | Sequence[PolylinePoint],
    stop_progress_m: Sequence[float],
    trip_key: Hashable,
) -> TripState:
    """Return the state after ``ping`` (duplicate / out-of-order: ``state`` unchanged)."""
    stops = tuple(float(s) for s in stop_progress_m)
    cur = state if state is not None else _empty_state(trip_key, stops)
    if ping.epoch <= cur.last_update_epoch:
        return cur
    progress = _match_next(_as_matcher(polyline), cur.pings, ping)
    matched: MatchedPing = (
        ping.epoch,
        math.nan if progress is None else progress,
        progress is not None,
    )
    prev = cur.last_valid()
    arrivals = cur.arrivals
    mid = cur.started_mid_trip
    if progress is not None and prev is None:
        mid = bool(stops) and progress > stops[0] + MID_TRIP_TOL_M
    elif progress is not None and prev is not None:
        new = _crossed_arrivals(stops, prev[0], prev[1], ping.epoch, progress)
        arrivals = tuple(new.get(j, a) for j, a in enumerate(arrivals))
    return TripState(
        cur.trip_key, cur.stop_progress_m, cur.pings + (matched,), arrivals, mid, ping.epoch
    )


# ---------------------------------------------------------------------------
# Batch (training)
# ---------------------------------------------------------------------------


def _causal_pings(pings: Sequence[Ping]) -> list[Ping]:
    """Drop duplicate / out-of-order pings exactly as the online path does."""
    if not pings:
        return []
    ts = np.array([p.epoch for p in pings], dtype=float)
    prior_max = np.concatenate([[-np.inf], np.maximum.accumulate(ts)[:-1]])
    return [p for p, keep in zip(pings, ts > prior_max, strict=True) if keep]


def arrivals_from_pings(
    pings: Sequence[Ping],
    polyline: ForwardMatcher | Sequence[PolylinePoint],
    stop_progress_m: Sequence[float],
    *,
    trip_key: Hashable = "",
) -> TripState:
    """Whole-trip equivalent of replaying ``pings`` through :func:`update_trip_state`.

    Matching is sequential; arrival interpolation is vectorised with
    ``searchsorted`` over the valid pings.  Returns the final :class:`TripState`
    (use :meth:`TripState.arrival_arrays` for numpy output).
    """
    stops = tuple(float(s) for s in stop_progress_m)
    matcher = _as_matcher(polyline)
    kept = _causal_pings(pings)
    matched: list[MatchedPing] = []
    for p in kept:
        progress = _match_next(matcher, matched, p)
        matched.append((p.epoch, math.nan if progress is None else progress, progress is not None))
    return TripState(
        trip_key,
        stops,
        tuple(matched),
        _batch_arrivals(matched, stops),
        _batch_mid_trip(matched, stops),
        kept[-1].epoch if kept else -math.inf,
    )


def _batch_mid_trip(matched: Sequence[MatchedPing], stops: tuple[float, ...]) -> bool:
    first = next((m for m in matched if m[2]), None)
    return first is not None and bool(stops) and first[1] > stops[0] + MID_TRIP_TOL_M


def _batch_arrivals(
    matched: Sequence[MatchedPing], stops: tuple[float, ...]
) -> tuple[StopArrival | None, ...]:
    valid = [m for m in matched if m[2]]
    out: list[StopArrival | None] = [None] * len(stops)
    if len(valid) < 2 or not stops:
        return tuple(out)
    ts = np.array([m[0] for m in valid])
    prog = np.array([m[1] for m in valid])
    sp = np.array(stops)
    idx = np.searchsorted(prog, sp, side="left")  # first valid ping at/after each stop
    for j in np.flatnonzero((idx > 0) & (idx < len(prog))):
        k = int(idx[j])
        gap = ts[k] - ts[k - 1]
        dp = prog[k] - prog[k - 1]
        if gap <= MAX_GAP_S and dp > 0:
            out[j] = StopArrival(
                float(ts[k - 1] + (sp[j] - prog[k - 1]) / dp * gap), float(ts[k]), float(gap)
            )
    return tuple(out)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def delay_last(
    state: TripState, sched_epochs: Sequence[float], progress_m: float, ping_epoch: float
) -> float:
    """``arrival - scheduled`` at the last passed stop detectable by ``ping_epoch``.

    "Passed" means stop progress <= ``progress_m``.  NaN when there is none.
    """
    j0 = int(np.searchsorted(state.stop_progress_m, progress_m, side="right"))
    for k in range(min(j0, len(state.arrivals)) - 1, -1, -1):
        a = state.arrivals[k]
        if a is not None and a.detect_epoch <= ping_epoch:
            return a.epoch - float(sched_epochs[k])
    return math.nan


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


@runtime_checkable
class TripStateStore(Protocol):
    """Keyed memory of trip states (in-process here; Redis later)."""

    def get(self, key: Hashable) -> TripState | None: ...

    def put(self, state: TripState) -> None: ...

    def evict(self, now_epoch: float) -> int: ...

    def __len__(self) -> int: ...


class InMemoryTripStateStore:
    """Dict-backed store.

    A trip is evicted when it has had no update for ``ttl_s``, or has been past
    its last stop for more than ``finished_grace_s``.
    """

    def __init__(self, *, ttl_s: float = 1800.0, finished_grace_s: float = 600.0) -> None:
        self._ttl_s = ttl_s
        self._grace_s = finished_grace_s
        self._states: dict[Hashable, TripState] = {}

    def get(self, key: Hashable) -> TripState | None:
        return self._states.get(key)

    def put(self, state: TripState) -> None:
        self._states = {**self._states, state.trip_key: state}

    def _expired(self, s: TripState, now: float) -> bool:
        if now - s.last_update_epoch > self._ttl_s:
            return True
        fin = s.finished_epoch()
        return fin is not None and now - fin > self._grace_s

    def evict(self, now_epoch: float) -> int:
        keep = {k: s for k, s in self._states.items() if not self._expired(s, now_epoch)}
        removed = len(self._states) - len(keep)
        self._states = keep
        return removed

    def __len__(self) -> int:
        return len(self._states)
