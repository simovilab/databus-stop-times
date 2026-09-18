"""Derive actual stop arrivals from vehicle positions, agency-agnostic.

Phase 2, track T2 (docs/PLAN.md). This is the ground truth everything else
(training targets, validation) is scored against, so every method is honest
about its evidence: `reported` arrivals are copied from the feed verbatim,
`map_matched` arrivals are backed by an actual shape-crossing interpolation,
and `interpolated` arrivals are clearly marked as filled gaps with a lower
`confidence`. Nothing here ever guesses a stop no observation supports.

Public entry point
-------------------
``derive_arrivals(observations, static, *, source, params) -> pd.DataFrame``
conforming to ``stoptimes.core.contracts.data.ARRIVAL_SCHEMA``. The CLI's
``stoptimes data arrivals`` command is wired to exactly this signature — see
docs/notes/p2-t2.md.

``observations`` is a DataFrame of ``Observation``-shaped rows (T1 produces
these from readers; tests build their own). ``static`` is anything exposing
``trip_geometry(trip_id) -> (shape_points, ordered_stops)`` — in production
``stoptimes.core.gtfs.static.GtfsStaticLoader`` (T1); tests use a small fake.

Algorithm summary (see each helper's docstring for detail):

1. **reported** — TripUpdate observations that already carry an
   ``arrival_time`` (or, lacking that, a ``departure_time``) are taken
   directly; the most recently observed one per (trip, start_date,
   stop_sequence) wins.
2. **map_matched** — VehiclePosition observations for a trip are projected
   onto the trip's shape (``core.spatial.polyline``); a stop "arrival" is
   the moment the vehicle's distance to the stop crosses
   ``params.arrival_radius_m`` (optionally gated by a continuous
   ``params.dwell_s`` and a shape-fit check via ``params.shape_threshold_m``),
   with the crossing **time linearly interpolated** between the two
   bracketing samples rather than snapped to the nearest one.
3. **interpolated** — an unobserved scheduled stop between two map-matched
   neighbors is filled by interpolating time proportionally to along-shape
   distance, only when the neighbors are close enough
   (``params.max_interpolation_gap_s`` / ``_m``); confidence is lowered.
4. Deduplicate by (trip_id, start_date, stop_sequence) — not just
   ``stop_id``, so a loop route visiting the same stop twice keeps both
   visits — preferring reported > map_matched > interpolated, then higher
   confidence. Finally drop physically impossible sequences: arrival times
   that regress against ``stop_sequence``, and any hop between consecutive
   kept arrivals implying a speed above ``params.max_teleport_speed_mps``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Protocol

import numpy as np
import pandas as pd

from stoptimes.core.contracts.data import (
    ARRIVAL_SCHEMA,
    ArrivalMethod,
    ObservationKind,
    coerce_utc_ns,
)
from stoptimes.core.contracts.source import GroundTruthDetector, Source
from stoptimes.core.domain.models import Stop
from stoptimes.core.spatial.polyline import (
    assign_stops_monotonic,
    build_polyline,
    haversine_m,
    project_point_to_polyline,
)

__all__ = [
    "ArrivalParams",
    "StaticGeometry",
    "derive_arrivals",
]


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArrivalParams:
    """Tunable knobs for :func:`derive_arrivals`.

    Defaults for ``arrival_radius_m``/``dwell_s``/``shape_threshold_m`` match
    ``contracts.source.GroundTruthDetector`` exactly (same field names,
    same defaults) — construct with :meth:`from_detector` to inherit a
    specific ``Source``'s tuned values, or override anything explicitly.
    """

    arrival_radius_m: float = 40.0
    """Great-circle distance (m) inside which the vehicle counts as having
    reached a stop (see ``GroundTruthDetector.arrival_radius_m``)."""

    dwell_s: float = 0.0
    """If > 0, the vehicle must stay continuously within ``arrival_radius_m``
    for at least this many seconds before the arrival is confirmed (see
    ``GroundTruthDetector.dwell_s``)."""

    shape_threshold_m: float = 30.0
    """Maximum cross-track distance (m) from the trip's shape at the
    crossing sample for a map-matched arrival to be trusted (see
    ``GroundTruthDetector.shape_threshold_m``). Guards against a radius
    overlap with a *different* stop that happens to sit nearby but off the
    trip's own shape."""

    max_interpolation_gap_s: float = 300.0
    """An unobserved intermediate stop is only filled by interpolation when
    the two bracketing map-matched arrivals are at most this many seconds
    apart."""

    max_interpolation_gap_m: float = 600.0
    """...and at most this many along-shape metres apart."""

    sampling_tolerance_s: float = 2.0
    """Two VehiclePosition samples at or below this time delta are treated
    as duplicates (same instant reported by more than one feed) and
    collapsed to one before matching."""

    max_teleport_speed_mps: float = 45.0
    """A hop between two consecutive *kept* arrivals implying a speed above
    this (~162 km/h) is treated as physically impossible and the later
    arrival of the pair is dropped."""

    @classmethod
    def from_detector(cls, detector: GroundTruthDetector, **overrides: float) -> "ArrivalParams":
        """Build params inheriting a ``Source``'s ``GroundTruthDetector`` values."""
        base = cls(
            arrival_radius_m=detector.arrival_radius_m,
            dwell_s=detector.dwell_s,
            shape_threshold_m=detector.shape_threshold_m,
        )
        return replace(base, **overrides) if overrides else base


# ---------------------------------------------------------------------------
# Static geometry duck type
# ---------------------------------------------------------------------------


class StaticGeometry(Protocol):
    """What `derive_arrivals` needs from a loaded static feed.

    ``stoptimes.core.gtfs.static.GtfsStaticLoader`` (T1) satisfies this
    already; tests may pass any object with the same method.
    """

    def trip_geometry(
        self, trip_id: str
    ) -> tuple[list[tuple[float, float, float | None]], list[Stop]]: ...


class _TripGeometry:
    """Resolved, ready-to-use geometry for one trip: polyline + ordered stops."""

    __slots__ = ("stops", "polyline", "shape_total_m")

    def __init__(self, stops: list[Stop], polyline: list | None) -> None:
        self.polyline = polyline
        if polyline and len(polyline) >= 2:
            self.stops = assign_stops_monotonic(stops, polyline)
            self.shape_total_m = polyline[-1].cum_m
        else:
            self.stops = stops
            self.shape_total_m = None


def _resolve_trip_geometry(static: StaticGeometry, trip_id: str) -> "_TripGeometry | None":
    try:
        shape_points, stops = static.trip_geometry(trip_id)
    except (ValueError, KeyError):
        return None
    if not stops:
        return None
    polyline = build_polyline(shape_points) if shape_points else None
    return _TripGeometry(stops, polyline)


# ---------------------------------------------------------------------------
# Output frame plumbing
# ---------------------------------------------------------------------------

_ROW_COLUMNS = ARRIVAL_SCHEMA.column_names


def _empty_frame() -> pd.DataFrame:
    dtypes = {
        "source_id": "string",
        "trip_id": "string",
        "start_date": "string",
        "route_id": "string",
        "vehicle_id": "string",
        "stop_id": "string",
        "stop_sequence": "int64",
        "arrival_time": "datetime64[ns, UTC]",
        "method": "string",
        "confidence": "float64",
    }
    return pd.DataFrame({c: pd.Series(dtype=dtypes[c]) for c in _ROW_COLUMNS})


def _rows_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return _empty_frame()
    df = pd.DataFrame(rows)
    df = df.reindex(columns=_ROW_COLUMNS)
    df["arrival_time"] = coerce_utc_ns(df["arrival_time"])
    for c in ("source_id", "trip_id", "start_date", "route_id", "vehicle_id", "stop_id", "method"):
        df[c] = df[c].astype("string")
    df["stop_sequence"] = df["stop_sequence"].astype("int64")
    df["confidence"] = df["confidence"].astype("float64")
    return df


def _service_date(ts: pd.Timestamp, tz: str) -> str:
    """Fallback service-day string (YYYYMMDD) when an observation lacks ``start_date``."""
    local = ts.tz_convert(tz) if ts.tzinfo is not None else ts.tz_localize("UTC").tz_convert(tz)
    return local.strftime("%Y%m%d")


# ---------------------------------------------------------------------------
# 1. reported
# ---------------------------------------------------------------------------

_SKIPPED_RELATIONSHIPS = {"SKIPPED", "CANCELED", "DELETED"}


def _reported_arrivals(observations: pd.DataFrame, source: Source) -> list[dict[str, Any]]:
    """TripUpdate observations that already publish an actual arrival time."""
    if observations.empty:
        return []
    tu = observations[observations["kind"] == ObservationKind.TRIP_UPDATE.value].copy()
    if tu.empty:
        return []

    if "schedule_relationship" in tu.columns:
        tu = tu[~tu["schedule_relationship"].isin(_SKIPPED_RELATIONSHIPS)]
    if tu.empty:
        return []

    time_col = tu["arrival_time"] if "arrival_time" in tu.columns else pd.Series(index=tu.index, dtype="object")
    if "departure_time" in tu.columns:
        time_col = time_col.fillna(tu["departure_time"])
    tu = tu.assign(_actual_time=time_col)
    tu = tu[tu["_actual_time"].notna()]
    if tu.empty:
        return []

    tu["_actual_time"] = pd.to_datetime(tu["_actual_time"], utc=True)
    tu["start_date"] = tu.apply(
        lambda r: r["start_date"] if pd.notna(r.get("start_date")) else _service_date(r["_actual_time"], source.timezone),
        axis=1,
    )
    # Latest report wins per (trip, start_date, stop_sequence): a later
    # TripUpdate supersedes an earlier prediction/estimate for the same stop.
    sort_key = tu["ingested_at"] if "ingested_at" in tu.columns and tu["ingested_at"].notna().any() else tu["ts"]
    tu = tu.assign(_sort_key=sort_key)
    tu = tu.sort_values("_sort_key")
    tu = tu.drop_duplicates(subset=["trip_id", "start_date", "stop_sequence"], keep="last")

    rows: list[dict[str, Any]] = []
    for _, r in tu.iterrows():
        rows.append(
            {
                "source_id": source.id,
                "trip_id": r["trip_id"],
                "start_date": r["start_date"],
                "route_id": r.get("route_id"),
                "vehicle_id": r.get("vehicle_id"),
                "stop_id": r["stop_id"],
                "stop_sequence": int(r["stop_sequence"]),
                "arrival_time": r["_actual_time"],
                "method": ArrivalMethod.REPORTED.value,
                "confidence": None,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# 2. map_matched
# ---------------------------------------------------------------------------


def _vp_trip_groups(
    observations: pd.DataFrame, source: Source, params: ArrivalParams
) -> Iterable[tuple[str, str, pd.DataFrame]]:
    """Yield (trip_id, start_date, group) for each VehiclePosition trip instance."""
    if observations.empty:
        return
    vp = observations[observations["kind"] == ObservationKind.VEHICLE_POSITION.value].copy()
    # Tolerate an observations frame that never carries VP columns at all
    # (e.g. a TripUpdate-only batch) — backfill as NA rather than KeyError.
    for col in ("lat", "lon", "ts", "trip_id", "start_date", "route_id", "vehicle_id"):
        if col not in vp.columns:
            vp[col] = pd.NA
    if vp.empty:
        return
    vp = vp[vp["lat"].notna() & vp["lon"].notna() & vp["ts"].notna() & vp["trip_id"].notna()]
    if vp.empty:
        return
    vp["ts"] = pd.to_datetime(vp["ts"], utc=True)
    vp["start_date"] = vp.apply(
        lambda r: r["start_date"] if pd.notna(r.get("start_date")) else _service_date(r["ts"], source.timezone),
        axis=1,
    )
    vp = vp.sort_values("ts")
    vp = vp.drop_duplicates(subset=["trip_id", "start_date", "lat", "lon", "ts"])

    for (trip_id, start_date), group in vp.groupby(["trip_id", "start_date"], sort=True):
        group = group.sort_values("ts").reset_index(drop=True)
        # Collapse near-duplicate timestamps (same instant from >1 feed).
        keep = [0]
        last_ts = group["ts"].iloc[0]
        for i in range(1, len(group)):
            gap = (group["ts"].iloc[i] - last_ts).total_seconds()
            if gap > params.sampling_tolerance_s:
                keep.append(i)
                last_ts = group["ts"].iloc[i]
        group = group.iloc[keep].reset_index(drop=True)
        yield str(trip_id), str(start_date), group


def _crossing_time(
    ts: np.ndarray, distances: np.ndarray, entry_idx: int, radius_m: float
) -> pd.Timestamp:
    """Interpolate the instant `distances` crosses below `radius_m`, bracketing at `entry_idx`."""
    if entry_idx == 0:
        return pd.Timestamp(ts[0])
    prev, cur = entry_idx - 1, entry_idx
    d0, d1 = distances[prev], distances[cur]
    if d0 == d1:
        return pd.Timestamp(ts[cur])
    frac = (d0 - radius_m) / (d0 - d1)
    frac = min(max(frac, 0.0), 1.0)
    t0, t1 = ts[prev], ts[cur]
    delta_ns = (t1 - t0) / np.timedelta64(1, "ns")
    return pd.Timestamp(t0 + np.timedelta64(int(round(delta_ns * frac)), "ns"))


def _find_confirmed_entry(
    distances: np.ndarray, ts: np.ndarray, radius_m: float, dwell_s: float
) -> int | None:
    """First index where the vehicle enters `radius_m` and (if `dwell_s` > 0)
    stays continuously inside it for at least `dwell_s`. Mirrors
    `val.groundtruth.position_radius.PositionRadiusDetector`'s reset rule.
    """
    n = len(distances)
    first_entry: int | None = None
    for i in range(n):
        if distances[i] <= radius_m:
            if first_entry is None:
                first_entry = i
            span = (ts[i] - ts[first_entry]) / np.timedelta64(1, "s")
            if dwell_s <= 0.0 or span >= dwell_s:
                return first_entry
        else:
            first_entry = None
    return None


def _map_match_trip(
    trip_id: str,
    start_date: str,
    group: pd.DataFrame,
    geometry: _TripGeometry,
    source: Source,
    params: ArrivalParams,
) -> list[dict[str, Any]]:
    lats = group["lat"].to_numpy(dtype=float)
    lons = group["lon"].to_numpy(dtype=float)
    ts = group["ts"].to_numpy()
    route_id = group["route_id"].iloc[0] if "route_id" in group.columns else None
    vehicle_id = group["vehicle_id"].iloc[0] if "vehicle_id" in group.columns else None

    cross_track: np.ndarray | None = None
    if geometry.polyline is not None:
        cross_track = np.array(
            [
                project_point_to_polyline(lats[i], lons[i], geometry.polyline).cross_track_m
                for i in range(len(lats))
            ]
        )

    rows: list[dict[str, Any]] = []
    for stop in geometry.stops:
        distances = np.array([haversine_m(lats[i], lons[i], stop.lat, stop.lon) for i in range(len(lats))])
        entry_idx = _find_confirmed_entry(distances, ts, params.arrival_radius_m, params.dwell_s)
        if entry_idx is None:
            continue

        if cross_track is not None and cross_track[entry_idx] > params.shape_threshold_m:
            # Close to this stop's coordinates but off the trip's own shape —
            # likely a different nearby route; not evidence for this stop.
            continue

        crossing_ts = _crossing_time(ts, distances, entry_idx, params.arrival_radius_m)

        if cross_track is not None:
            confidence = float(
                min(1.0, max(0.5, 1.0 - cross_track[entry_idx] / max(params.shape_threshold_m, 1e-9)))
            )
        else:
            # No shape available for this trip — still map-matched (derived
            # from positions), but with no cross-track validation possible.
            confidence = 0.6

        rows.append(
            {
                "source_id": source.id,
                "trip_id": trip_id,
                "start_date": start_date,
                "route_id": route_id,
                "vehicle_id": vehicle_id,
                "stop_id": stop.stop_id,
                "stop_sequence": int(stop.stop_sequence),
                "arrival_time": crossing_ts,
                "method": ArrivalMethod.MAP_MATCHED.value,
                "confidence": confidence,
            }
        )
    return rows


def _map_matched_arrivals(
    observations: pd.DataFrame,
    static: StaticGeometry,
    source: Source,
    params: ArrivalParams,
    geometry_cache: dict[str, "_TripGeometry | None"],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trip_id, start_date, group in _vp_trip_groups(observations, source, params):
        if trip_id not in geometry_cache:
            geometry_cache[trip_id] = _resolve_trip_geometry(static, trip_id)
        geometry = geometry_cache[trip_id]
        if geometry is None or len(group) == 0:
            continue
        rows.extend(_map_match_trip(trip_id, start_date, group, geometry, source, params))
    return rows


# ---------------------------------------------------------------------------
# 3. interpolated — fill small gaps between map-matched neighbors
# ---------------------------------------------------------------------------


def _interpolate_trip_gaps(
    trip_id: str,
    start_date: str,
    matched_rows: list[dict[str, Any]],
    geometry: _TripGeometry,
    source: Source,
    params: ArrivalParams,
) -> list[dict[str, Any]]:
    if geometry.polyline is None or geometry.shape_total_m is None:
        return []
    ordered_stops = sorted(geometry.stops, key=lambda s: s.stop_sequence)
    if len(ordered_stops) < 3:
        return []
    matched_by_seq = {r["stop_sequence"]: r for r in matched_rows}
    seq_to_stop = {s.stop_sequence: s for s in ordered_stops}

    matched_seqs = sorted(matched_by_seq)
    rows: list[dict[str, Any]] = []
    for left_seq, right_seq in zip(matched_seqs, matched_seqs[1:]):
        between = [seq for seq in seq_to_stop if left_seq < seq < right_seq]
        if not between:
            continue
        left, right = matched_by_seq[left_seq], matched_by_seq[right_seq]
        left_t, right_t = pd.Timestamp(left["arrival_time"]), pd.Timestamp(right["arrival_time"])
        gap_s = (right_t - left_t).total_seconds()
        left_stop, right_stop = seq_to_stop[left_seq], seq_to_stop[right_seq]
        gap_m = (
            (right_stop.progress_m - left_stop.progress_m)
            if left_stop.progress_m is not None and right_stop.progress_m is not None
            else None
        )
        if gap_s < 0 or gap_s > params.max_interpolation_gap_s:
            continue
        if gap_m is not None and gap_m > params.max_interpolation_gap_m:
            continue

        for seq in between:
            stop = seq_to_stop[seq]
            if stop.progress_m is None or left_stop.progress_m is None or right_stop.progress_m is None:
                continue
            span = right_stop.progress_m - left_stop.progress_m
            frac = 0.5 if span <= 0 else (stop.progress_m - left_stop.progress_m) / span
            frac = min(max(frac, 0.0), 1.0)
            interp_ts = left_t + (right_t - left_t) * frac
            base_confidence = min(left.get("confidence") or 0.6, right.get("confidence") or 0.6)
            confidence = float(max(0.2, base_confidence * 0.5))
            rows.append(
                {
                    "source_id": source.id,
                    "trip_id": trip_id,
                    "start_date": start_date,
                    "route_id": left.get("route_id"),
                    "vehicle_id": left.get("vehicle_id"),
                    "stop_id": stop.stop_id,
                    "stop_sequence": int(seq),
                    "arrival_time": interp_ts,
                    "method": ArrivalMethod.INTERPOLATED.value,
                    "confidence": confidence,
                }
            )
    return rows


def _interpolated_arrivals(
    matched_rows: list[dict[str, Any]],
    geometry_cache: dict[str, "_TripGeometry | None"],
    source: Source,
    params: ArrivalParams,
) -> list[dict[str, Any]]:
    if not matched_rows:
        return []
    rows: list[dict[str, Any]] = []
    df = pd.DataFrame(matched_rows)
    for (trip_id, start_date), group in df.groupby(["trip_id", "start_date"], sort=True):
        geometry = geometry_cache.get(trip_id)
        if geometry is None:
            continue
        rows.extend(
            _interpolate_trip_gaps(
                trip_id, start_date, group.to_dict("records"), geometry, source, params
            )
        )
    return rows


# ---------------------------------------------------------------------------
# 4. Dedup + physical-plausibility filtering
# ---------------------------------------------------------------------------

_METHOD_RANK = {
    ArrivalMethod.REPORTED.value: 3,
    ArrivalMethod.MAP_MATCHED.value: 2,
    ArrivalMethod.INTERPOLATED.value: 1,
}


def _dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (trip_id, start_date, stop_sequence) — a loop stop visited
    twice keeps both visits since each has its own stop_sequence. Prefers
    reported > map_matched > interpolated, then higher confidence.
    """
    if df.empty:
        return df
    df = df.copy()
    df["_rank"] = df["method"].map(_METHOD_RANK).fillna(0)
    df["_conf_sort"] = df["confidence"].fillna(1.0)
    df = df.sort_values(["_rank", "_conf_sort"])
    df = df.drop_duplicates(subset=["trip_id", "start_date", "stop_sequence"], keep="last")
    return df.drop(columns=["_rank", "_conf_sort"])


def _drop_impossible(
    df: pd.DataFrame, geometry_cache: dict[str, "_TripGeometry | None"], params: ArrivalParams
) -> pd.DataFrame:
    """Drop non-monotonic arrivals and teleports within each trip instance."""
    if df.empty:
        return df
    kept_frames: list[pd.DataFrame] = []
    for (trip_id, start_date), group in df.groupby(["trip_id", "start_date"], sort=False):
        group = group.sort_values("stop_sequence").reset_index(drop=True)
        geometry = geometry_cache.get(trip_id)
        stop_lookup = {s.stop_id: s for s in geometry.stops} if geometry else {}

        keep_rows: list[int] = []
        last_time: pd.Timestamp | None = None
        last_stop_id: str | None = None
        for i in range(len(group)):
            row = group.iloc[i]
            t = pd.Timestamp(row["arrival_time"])
            if last_time is not None and t < last_time:
                continue  # non-monotonic against stop_sequence order
            if last_time is not None and last_stop_id is not None and stop_lookup:
                s_from, s_to = stop_lookup.get(last_stop_id), stop_lookup.get(row["stop_id"])
                if s_from is not None and s_to is not None:
                    dist_m = haversine_m(s_from.lat, s_from.lon, s_to.lat, s_to.lon)
                    dt_s = (t - last_time).total_seconds()
                    if dt_s > 0 and (dist_m / dt_s) > params.max_teleport_speed_mps:
                        continue  # implausible speed — teleport
            keep_rows.append(i)
            last_time = t
            last_stop_id = row["stop_id"]
        kept_frames.append(group.iloc[keep_rows])

    return pd.concat(kept_frames, ignore_index=True) if kept_frames else df.iloc[0:0]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def derive_arrivals(
    observations: pd.DataFrame,
    static: StaticGeometry,
    *,
    source: Source,
    params: ArrivalParams | None = None,
) -> pd.DataFrame:
    """Derive an `Arrival` table (contracts.data.ARRIVAL_SCHEMA) from raw
    `Observation` rows plus the trip's static geometry.

    Args:
        observations: DataFrame of `Observation`-shaped rows (any subset of
            `OBSERVATION_VP_SCHEMA`/`OBSERVATION_TU_SCHEMA` columns is
            tolerated; missing optional columns are treated as absent).
        static: Anything exposing `trip_geometry(trip_id) -> (shape_points,
            ordered_stops)` — `stoptimes.core.gtfs.static.GtfsStaticLoader`
            in production.
        source: The `Source` these observations came from — supplies
            `source_id`, `timezone` (for a missing `start_date` fallback),
            and (via `params` defaulting) the `GroundTruthDetector`.
        params: Derivation knobs; defaults to
            `ArrivalParams.from_detector(source.detector)`.

    Returns:
        A DataFrame with exactly `ARRIVAL_SCHEMA.column_names`, one row per
        (trip_id, start_date, stop_sequence), sorted by
        (trip_id, start_date, stop_sequence).
    """
    if params is None:
        params = ArrivalParams.from_detector(source.detector)

    geometry_cache: dict[str, "_TripGeometry | None"] = {}

    reported_rows = _reported_arrivals(observations, source)
    matched_rows = _map_matched_arrivals(observations, static, source, params, geometry_cache)
    interpolated_rows = _interpolated_arrivals(matched_rows, geometry_cache, source, params)

    all_rows = reported_rows + matched_rows + interpolated_rows
    frame = _rows_to_frame(all_rows)
    frame = _dedupe(frame)
    frame = _drop_impossible(frame, geometry_cache, params)
    frame = frame.sort_values(["trip_id", "start_date", "stop_sequence"]).reset_index(drop=True)
    return frame.reindex(columns=_ROW_COLUMNS)
