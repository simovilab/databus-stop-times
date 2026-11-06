"""Scheduled-time features from GTFS static (agency-agnostic).

Pure functions; no I/O. GTFS anchoring: a service day's origin is local noon
minus 12 h on the service date (NOT local midnight on DST-change days), and
stop times are seconds from that origin (>= 24:00:00 allowed).

Behavioral reference: ``_trip_block`` / ``build_features`` of the MBTA study
(``sched_now = interp(prog, cummax(stop_progress), sched)``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np

_HALF_DAY_S = 12 * 3600


@dataclass(frozen=True)
class TripSchedule:
    """Scheduled stop times of one trip, sorted by stop_sequence."""

    trip_id: str
    stop_ids: tuple[str, ...]
    stop_sequences: tuple[int, ...]
    offsets_s: tuple[float, ...]  # seconds from the service-day origin; NaN if unknown
    interpolated: tuple[bool, ...]  # True where the time was filled, not published


# --- time parsing and anchoring ----------------------------------------------


def parse_gtfs_time(text: object) -> float:
    """``HH:MM:SS`` (hours may be >= 24) to seconds; blank/None/NaN to NaN."""
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return math.nan
    raw = str(text).strip()
    if raw == "" or raw.lower() == "nan":
        return math.nan
    parts = raw.split(":")
    try:
        if len(parts) != 3:
            raise ValueError
        h, m, s = (int(p) for p in parts)
    except ValueError:
        raise ValueError(f"invalid GTFS time {raw!r}") from None
    return float(h * 3600 + m * 60 + s)


def _zone(tz: str) -> ZoneInfo:
    try:
        return ZoneInfo(tz)
    except (ZoneInfoNotFoundError, ValueError, OSError):
        raise ValueError(f"unknown timezone {tz!r}") from None


def service_day_origin(start_date: str, tz: str) -> float:
    """Epoch seconds of (local noon on ``YYYYMMDD``) minus 12 h, per the GTFS spec."""
    try:
        day = datetime.strptime(start_date, "%Y%m%d")
    except ValueError:
        raise ValueError(f"start_date must be YYYYMMDD, got {start_date!r}") from None
    noon = day.replace(hour=12, tzinfo=_zone(tz))
    return noon.timestamp() - _HALF_DAY_S


def resolve_timezone(static_tz: str | None, config_tz: str | None) -> str:
    """agency.txt timezone first, configured fallback second; error if neither."""
    for candidate in (static_tz, config_tz):
        if candidate and candidate.strip():
            tz = candidate.strip()
            _zone(tz)
            return tz
    raise ValueError(
        "no timezone: agency.txt has no agency_timezone and none is configured"
    )


# --- building a TripSchedule ---------------------------------------------------


def _usable_x(dist: Sequence[float] | None, n: int) -> np.ndarray:
    """Interpolation abscissa: shape_dist_traveled if complete and monotone, else index."""
    index = np.arange(n, dtype=float)
    if dist is None:
        return index
    x = np.asarray(dist, dtype=float)
    if x.shape != (n,) or np.isnan(x).any() or (np.diff(x) < 0).any() or x[-1] <= x[0]:
        return index
    return x


def _fill_blanks(offsets: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Linearly fill NaNs lying between known times; return (offsets, filled mask)."""
    known = ~np.isnan(offsets)
    if known.sum() < 2:
        return offsets, np.zeros(offsets.shape, dtype=bool)
    lo, hi = np.flatnonzero(known)[[0, -1]]
    inside = np.zeros(offsets.shape, dtype=bool)
    inside[lo : hi + 1] = True
    filled = inside & ~known
    out = offsets.copy()
    out[filled] = np.interp(x[filled], x[known], offsets[known])
    return out, filled


def build_trip_schedule(
    trip_id: str,
    stop_ids: Sequence[str],
    stop_sequences: Sequence[int],
    arrival_times: Sequence[object],
    departure_times: Sequence[object],
    shape_dist_traveled: Sequence[float] | None = None,
) -> TripSchedule:
    """Sort by stop_sequence, take arrival (else departure), fill blanks, freeze."""
    n = len(stop_ids)
    lengths = {len(stop_sequences), len(arrival_times), len(departure_times)}
    if shape_dist_traveled is not None:
        lengths.add(len(shape_dist_traveled))
    if lengths != {n}:
        raise ValueError("stop_times columns have different lengths")
    order = np.argsort(np.asarray(stop_sequences, dtype=float), kind="stable")
    arr = np.array([parse_gtfs_time(arrival_times[i]) for i in order])
    dep = np.array([parse_gtfs_time(departure_times[i]) for i in order])
    raw = np.where(np.isnan(arr), dep, arr)
    dist = None if shape_dist_traveled is None else [shape_dist_traveled[i] for i in order]
    offsets, filled = _fill_blanks(raw, _usable_x(dist, n))
    return TripSchedule(
        trip_id=trip_id,
        stop_ids=tuple(str(stop_ids[i]) for i in order),
        stop_sequences=tuple(int(stop_sequences[i]) for i in order),
        offsets_s=tuple(float(v) for v in offsets),
        interpolated=tuple(bool(f) for f in filled),
    )


def scheduled_epochs(schedule: TripSchedule, start_date: str, tz: str) -> np.ndarray:
    """Scheduled epoch seconds per stop for the service day ``start_date``."""
    return service_day_origin(start_date, tz) + np.asarray(schedule.offsets_s, dtype=float)


# --- features -------------------------------------------------------------------


def scheduled_at_progress(
    stop_progress_m: np.ndarray, sched_epochs: np.ndarray, progress_m: float | np.ndarray
) -> float | np.ndarray:
    """Scheduled time at a path position; cummax progress, clamped at both ends.

    Stops whose schedule is missing are skipped; all missing gives NaN.
    """
    prog = np.maximum.accumulate(np.asarray(stop_progress_m, dtype=float))
    sched = np.asarray(sched_epochs, dtype=float)
    ok = ~np.isnan(sched) & ~np.isnan(prog)
    if not ok.any():
        return math.nan if np.ndim(progress_m) == 0 else np.full(np.shape(progress_m), np.nan)
    out = np.interp(progress_m, prog[ok], sched[ok])
    return float(out) if np.ndim(progress_m) == 0 else out


def schedule_features(
    stop_progress_m: np.ndarray,
    sched_epochs: np.ndarray,
    progress_m: float,
    ping_ts: float,
    t: float,
    target_idx: np.ndarray,
) -> dict[str, np.ndarray]:
    """``delay_now``, ``sched_rem``, ``sched_rem_t`` for one vehicle and many targets."""
    idx = np.asarray(target_idx, dtype=int)
    sched = np.asarray(sched_epochs, dtype=float)
    sched_now = float(scheduled_at_progress(stop_progress_m, sched, progress_m))
    return {
        "delay_now": np.full(idx.shape, ping_ts - sched_now, dtype=float),
        "sched_rem": sched[idx] - sched_now,
        "sched_rem_t": sched[idx] - t,
    }
