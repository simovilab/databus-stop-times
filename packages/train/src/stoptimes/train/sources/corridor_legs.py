"""Leg segmentation for a back-and-forth shuttle on one corridor: turning points, not dwell gaps.

``bucr_trip_inference`` splits a vehicle-day at stationary waits of >= 90 s. On bUCR's
semester data that loses each trip's first leg: buses turn around at Educación/Artes in
30 s - 3 min, so a quick turnaround is no boundary, the new trip's first ~300 m are
glued to the previous trip and discarded, and with the origin stop gone the variant
(Educación vs Artes origin) becomes a coin flip (44% agreement with the timetable).

This module segments by the corridor geometry instead:

1. every ping is projected onto a *reference* shape running from the low-end terminals
   (Educación/Artes) to the high end (Odontología): corridor coordinate ``c``;
   off-corridor pings (the milla detour) keep the last on-corridor value;
2. the zigzag of ``c`` (extrema at least ``hysteresis_m`` apart) gives the turnarounds;
3. inside each turnaround zone the cut is the *departure* from the longest stationary
   wait near a terminal (no wait: the extremum), so a leg runs from leaving one
   terminal to leaving the next, its arrival and wait included;
4. a leg going up the corridor is the low->high direction; its variant is the low-end
   terminal it left (nearest to its first ping) plus whether it visited a detour stop;
   a leg going down ends at the low-end terminal nearest its last ping.

The variant is only a *prior*: :mod:`schedule_match` lets the timetable overrule it.
Everything shape-specific (terminals, detour stops, which shape is which variant) is
derived from the static GTFS by :func:`corridor_layout`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from stoptimes.core.spatial.polyline import haversine_m, project_point_to_polyline
from stoptimes.train.sources.bucr_gtfs import RouteDirectionCandidate

__all__ = ["LegParams", "CorridorLayout", "Leg", "corridor_layout", "segment_legs"]


@dataclass(frozen=True)
class LegParams:
    """Thresholds of the leg segmentation (metres / seconds)."""

    hysteresis_m: float = 1500.0
    """Turnarounds are extrema of the corridor coordinate at least this far apart (bUCR: a leg
    spans ~4.4 km, the Educación/Artes loop ~350 m)."""
    zone_m: float = 700.0
    """A turnaround zone is the stretch within this of the corridor's low or high end."""
    terminal_radius_m: float = 40.0
    off_corridor_m: float = 120.0
    """Pings further than this from the reference shape (a detour) do not move ``c``."""
    detour_radius_m: float = 40.0
    max_gap_s: float = 900.0
    """A leg with a ping gap longer than this is dropped (device offline mid-trip)."""
    layover_s: float = 120.0
    """In a turnaround zone the leg starts at the departure from the *last* wait at least this
    long (a bus that waits at Educación and again at Artes leaves from Artes); with none, from
    the longest wait (bUCR buses also stop 30-90 s at Artes on their way out of Educación)."""
    out_of_service_wait_s: float = 300.0
    """A stationary wait this long away from a terminal (bUCR: a break at the depot, which
    lies on the inbound route) ends the leg where the wait starts - otherwise every row
    before it would carry the break in its target."""
    min_pings: int = 5


@dataclass(frozen=True)
class CorridorLayout:
    """What the segmentation needs to know about the network, derived from the GTFS."""

    reference: RouteDirectionCandidate
    up_direction: int
    """``direction_id`` of shapes running low end -> high end."""
    low_terminals: Mapping[str, tuple[float, float]]
    high_terminals: Mapping[str, tuple[float, float]]
    detour_stops: Mapping[str, tuple[float, float]]
    variants: Mapping[tuple[int, str, bool], str]
    """``(direction_id, low-end terminal stop_id, took detour) -> shape_id``."""
    candidates: Mapping[str, RouteDirectionCandidate] = field(repr=False)


@dataclass(frozen=True)
class Leg:
    """One segmented leg: ``[start, end]`` row indices into the vehicle-day frame."""

    start: int
    end: int
    direction_id: int
    shape_id: str


def _endpoints(c: RouteDirectionCandidate) -> tuple[str, str]:
    return c.stops[0].stop_id, c.stops[-1].stop_id


def corridor_layout(candidates: Sequence[RouteDirectionCandidate]) -> CorridorLayout:
    """Terminals, detour stops and variant lookup of a two-terminal-zone shuttle network.

    Terminals are shape endpoints that are never an intermediate stop. The up direction
    is the one whose shapes start at a terminal and end at a terminal of a different
    zone; detour stops are the intermediate stops that differ between shapes of one
    direction sharing both endpoints (bUCR: the milla stops).
    """
    by_id = {c.shape_id: c for c in candidates}
    loc = {s.stop_id: (s.lat, s.lon) for c in candidates for s in c.stops}
    inner = {s.stop_id for c in candidates for s in c.stops[1:-1]}
    ends = {sid for c in candidates for sid in _endpoints(c)}
    terminals = ends - inner
    directions = sorted({c.direction_id for c in candidates})
    up = next(
        d
        for d in directions
        if all(_endpoints(c)[0] in terminals for c in candidates if c.direction_id == d)
    )
    ups = [c for c in candidates if c.direction_id == up]
    low = {_endpoints(c)[0] for c in ups}
    high = {_endpoints(c)[1] for c in ups if _endpoints(c)[1] in terminals}
    high |= {
        _endpoints(c)[0]
        for c in candidates
        if c.direction_id != up and _endpoints(c)[0] in terminals
    }
    detour: set[str] = set()
    groups: dict[tuple[int, str, str], list[RouteDirectionCandidate]] = {}
    for c in candidates:
        groups.setdefault((c.direction_id, *_endpoints(c)), []).append(c)
    for members in groups.values():
        if len(members) > 1:
            sets = [{s.stop_id for s in m.stops[1:-1]} for m in members]
            detour |= set.union(*sets) - set.intersection(*sets)
    reference = min(
        (c for c in ups if not ({s.stop_id for s in c.stops} & detour)),
        key=lambda c: c.polyline[-1].cum_m,
        default=ups[0],
    )

    def low_high(c: RouteDirectionCandidate) -> tuple[str, str]:
        first, last = _endpoints(c)
        return (first, last) if c.direction_id == up else (last, first)

    variants: dict[tuple[int, str, bool], str] = {}
    # a shape whose high end is a real terminal wins its key (bUCR: Odontología->Educación
    # over the once-a-day EDUFI->Educación; the timetable still picks the latter at 21:20)
    for c in sorted(candidates, key=lambda c: (low_high(c)[1] not in high, c.shape_id)):
        low_end = low_high(c)[0]
        if low_end in low:
            key = (c.direction_id, low_end, bool({s.stop_id for s in c.stops} & detour))
            variants.setdefault(key, c.shape_id)
    return CorridorLayout(
        reference=reference,
        up_direction=up,
        low_terminals={s: loc[s] for s in sorted(low)},
        high_terminals={s: loc[s] for s in sorted(high)},
        detour_stops={s: loc[s] for s in sorted(detour)},
        variants=variants,
        candidates=by_id,
    )


def _near(
    lat: np.ndarray, lon: np.ndarray, where: tuple[float, float], radius: float
) -> np.ndarray:
    return np.array(
        [haversine_m(a, b, *where) <= radius for a, b in zip(lat, lon, strict=True)], dtype=bool
    )


def _zigzag(c: np.ndarray, hysteresis: float) -> list[int]:
    """Indices of alternating extrema of ``c``, consecutive ones at least ``hysteresis`` apart."""
    ext, mode, lo, hi = [0], 0, 0, 0
    for i in range(1, len(c)):
        hi = i if c[i] > c[hi] else hi
        lo = i if c[i] < c[lo] else lo
        if mode >= 0 and c[hi] - c[i] >= hysteresis:
            ext.append(hi)
            mode, lo = -1, i
        elif mode <= 0 and c[i] - c[lo] >= hysteresis:
            ext.append(lo)
            mode, hi = 1, i
    ext.append(len(c) - 1)
    return sorted(set(ext))


def _departure(
    t: np.ndarray, stopped: np.ndarray, near: np.ndarray, lo: int, hi: int, layover_s: float
) -> int | None:
    """Departure (last index) from the last stationary run in ``near[lo:hi]`` lasting
    ``layover_s`` or more, else from the longest one (None: no wait at all)."""
    best: tuple[float, int] | None = None
    last_layover: int | None = None
    i = lo
    while i < hi:
        if near[i] and stopped[i]:
            j = i
            while j + 1 < hi and near[j + 1] and stopped[j + 1]:
                j += 1
            dur = float((t[j] - t[i]) / np.timedelta64(1, "s"))
            if best is None or dur > best[0]:
                best = (dur, j)
            if dur >= layover_s:
                last_layover = j
            i = j + 1
        else:
            i += 1
    if last_layover is not None:
        return last_layover
    return None if best is None else best[1]


def _cuts(
    c: np.ndarray,
    t: np.ndarray,
    stopped: np.ndarray,
    near_low: np.ndarray,
    near_high: np.ndarray,
    total: float,
    params: LegParams,
) -> list[int]:
    ext = _zigzag(c, params.hysteresis_m)
    cuts = [0]
    for e in ext[1:-1]:
        is_low = c[e] < total / 2
        zone = c <= params.zone_m if is_low else c >= total - params.zone_m
        lo, hi = e, e
        while lo > 0 and zone[lo - 1]:
            lo -= 1
        while hi + 1 < len(c) and zone[hi + 1]:
            hi += 1
        dep = _departure(
            t, stopped, near_low if is_low else near_high, lo, hi + 1, params.layover_s
        )
        cuts.append(e if dep is None else dep)
    cuts.append(len(c) - 1)
    return sorted(set(cuts))


def _out_of_service_start(
    t: np.ndarray, stopped: np.ndarray, at_terminal: np.ndarray, a: int, b: int, min_wait_s: float
) -> int | None:
    """Start index of the first stationary run in ``(a, b]`` lasting ``min_wait_s`` or more
    with no ping at a terminal (None: there is none)."""
    i = a + 1
    while i <= b:
        if stopped[i]:
            j = i
            while j + 1 <= b and stopped[j + 1]:
                j += 1
            long_enough = (t[j] - t[i]) / np.timedelta64(1, "s") >= min_wait_s
            if long_enough and not at_terminal[i : j + 1].any():
                return i
            i = j + 1
        else:
            i += 1
    return None


def _nearest(lat: float, lon: float, places: Mapping[str, tuple[float, float]]) -> str:
    return min(places, key=lambda s: haversine_m(lat, lon, *places[s]))


def segment_legs(
    day: pd.DataFrame, layout: CorridorLayout, params: LegParams | None = None
) -> list[Leg]:
    """Legs of one vehicle's service day.

    ``day`` holds one vehicle's pings in time order with ``ts`` (tz-aware), ``lat``,
    ``lon`` and ``stopped`` (bool: the device reported the bus stationary).
    """
    params = params or LegParams()
    if len(day) < params.min_pings:
        return []
    lat, lon = day["lat"].to_numpy(float), day["lon"].to_numpy(float)
    t = day["ts"].to_numpy()
    ref = layout.reference.polyline
    proj = [project_point_to_polyline(a, b, ref) for a, b in zip(lat, lon, strict=True)]
    c = np.array([p.progress_m for p in proj])
    on = np.array([p.cross_track_m for p in proj]) <= params.off_corridor_m
    c = pd.Series(np.where(on, c, np.nan)).ffill().bfill().to_numpy()
    if np.isnan(c).all():
        return []
    stopped = day["stopped"].to_numpy(bool)
    near_low = np.zeros(len(day), bool)
    for where in layout.low_terminals.values():
        near_low |= _near(lat, lon, where, params.terminal_radius_m)
    near_high = np.zeros(len(day), bool)
    for where in layout.high_terminals.values():
        near_high |= _near(lat, lon, where, params.terminal_radius_m)
    cuts = _cuts(c, t, stopped, near_low, near_high, ref[-1].cum_m, params)
    legs = []
    for a, b in zip(cuts[:-1], cuts[1:], strict=True):
        if abs(c[b] - c[a]) < params.hysteresis_m:
            continue  # a stub between two cuts in the same zone
        up = bool(c[b] > c[a])
        end = b  # where the leg reaches its terminal: decides a down leg's variant
        brk = _out_of_service_start(
            t, stopped, near_low | near_high, a, b, params.out_of_service_wait_s
        )
        if brk is not None:
            b = brk  # the trip is unobserved after the bus went out of service
        if b - a + 1 < params.min_pings or abs(c[b] - c[a]) < params.hysteresis_m:
            continue  # too short to be a trip
        if (
            b > a
            and float(np.max(np.diff(t[a : b + 1]) / np.timedelta64(1, "s"))) > params.max_gap_s
        ):
            continue
        direction = layout.up_direction if up else _other(layout)
        low_end = (
            _nearest(lat[a], lon[a], layout.low_terminals)
            if up
            else _nearest(lat[end], lon[end], layout.low_terminals)
        )
        detour = any(
            _near(lat[a : end + 1], lon[a : end + 1], w, params.detour_radius_m).any()
            for w in layout.detour_stops.values()
        )
        shape = layout.variants.get((direction, low_end, detour)) or layout.variants.get(
            (direction, low_end, False)
        )
        if shape is not None:
            legs.append(Leg(a, b, direction, shape))
    return legs


def _other(layout: CorridorLayout) -> int:
    return next(
        c.direction_id for c in layout.candidates.values() if c.direction_id != layout.up_direction
    )
