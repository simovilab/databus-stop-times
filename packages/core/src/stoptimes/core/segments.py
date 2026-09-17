"""Shared core for the segment ETA dataset builders (roadmap 3.1).

Both agency builders -- :mod:`stoptimes.train.datasets.segment_dataset_builder`
(bUCR, inferred trips) and :mod:`stoptimes.train.datasets.mbta_segments` (MBTA, real
GTFS trips) -- reduce to the *same* problem once each has resolved its raw feed
into **trip instances**: given, per instance, an ordered VP frame plus the
instance's ordered stops (with along-shape ``progress_m``) and shape polyline,
emit one row per traversed stop-to-stop segment with a position-derived target.

Only the *resolver* differs between agencies (bUCR parses a synthetic trip id
back to a route-direction candidate; MBTA maps a real ``trip_id`` through the
static GTFS and groups by service date). Everything downstream of a
:class:`TripInstance` -- arrival detection, the per-segment feature/target
emission, the cyclical encodings, and the final schema projection -- is
identical and lives here, so the two builders share one tested implementation
instead of duplicating it.

Pure/deterministic: no I/O, no network, no mutation of inputs. Spatial math is
reused from ``stoptimes.core.spatial.polyline``; temporal features from
``stoptimes.core.features.temporal``. Runs under ``uv run --group bucr``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, degrees, radians, sin
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from stoptimes.core.spatial.polyline import haversine_m, project_point_to_polyline

from stoptimes.core.config import AGENCY_TEMPORAL_DEFAULTS
from stoptimes.core.features.temporal import extract_temporal_features

__all__ = [
    "ARRIVAL_RADIUS_M",
    "CLOSEST_APPROACH_FALLBACK_M",
    "MAX_SEGMENT_SECONDS",
    "OUTPUT_COLUMNS",
    "TripInstance",
    "resolve_agency_temporal",
    "build_segments_from_instances",
]

# First VP within this many metres of a stop = "arrived" (same as
# ``dataset_builder``'s ``distance_threshold`` default).
ARRIVAL_RADIUS_M: float = 50.0

# If a stop is never approached within ``ARRIVAL_RADIUS_M``, fall back to the
# closest approach only when it is at least this near (same 200 m rule as
# ``dataset_builder.find_actual_arrival_time``). Sparse campus-shuttle pings
# routinely skip past a stop without landing a fix inside 50 m.
CLOSEST_APPROACH_FALLBACK_M: float = 200.0

# A single stop-to-stop segment taking longer than this (seconds) is treated as
# a spurious traversal (an undetected layover between the two arrivals, or a
# mis-paired arrival) and dropped -- campus segments are short; 30 min is
# already far beyond any legitimate stop-to-stop leg.
MAX_SEGMENT_SECONDS: float = 1800.0

# Emitted columns: exactly the schema ``dataset_builder.build_vp_training_dataset``
# produces (so ``ETADataset``/``train_all_models`` consume this unchanged), plus
# two segment-provenance columns (``from_stop_id``, ``segment_index``) that the
# model feature groups simply ignore.
OUTPUT_COLUMNS: List[str] = [
    # identifiers
    "trip_id", "route_id", "vehicle_id", "stop_id", "stop_sequence",
    # segment provenance (ignored by the model feature groups)
    "from_stop_id", "segment_index",
    # position
    "vp_ts", "vp_lat", "vp_lon", "vp_bearing", "stop_lat", "stop_lon",
    # spatial
    "distance_to_stop", "distance_to_next_stop", "shape_distance_to_stop",
    "shape_progress", "cross_track_error", "progress_ratio", "stops_ahead",
    # kinematic
    "current_speed_kmh", "bearing_to_stop", "bearing_diff",
    # status
    "is_at_stop",
    # target (position-derived)
    "actual_arrival", "time_to_arrival_seconds",
    # temporal
    "hour", "day_of_week", "is_weekend", "is_holiday", "is_peak_hour",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]


@dataclass(frozen=True)
class TripInstance:
    """One resolved trip instance ready for segment emission.

    Attributes:
        trip_id: Identifier emitted verbatim in the ``trip_id`` column.
        route_id: Route identifier emitted verbatim.
        vehicle_id: Vehicle identifier emitted verbatim (resolved differently
            per agency -- from the synthetic id for bUCR, from the VP frame for
            MBTA -- but consumed identically here).
        stops: Ordered stops for the instance; each must expose ``lat``,
            ``lon``, ``stop_id``, ``stop_sequence`` and along-shape
            ``progress_m`` (already filtered to non-null ``progress_m`` by the
            resolver, but re-filtered defensively here).
        polyline: The instance's shape polyline (list of ``PolylinePoint`` with
            cumulative ``cum_m``) or ``None`` when unavailable -- used for the
            cross-track feature and the shape-total progress denominator.
        trip_vps: The instance's raw VP frame (must carry ``ts``/``lat``/``lon``
            and optionally ``bearing``/``speed``/``current_status``); sorted by
            ``ts`` here, so the resolver need not pre-sort.
    """

    trip_id: str
    route_id: str
    vehicle_id: object
    stops: list
    polyline: Optional[list]
    trip_vps: pd.DataFrame


def _initial_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial great-circle bearing point1->point2, degrees [0, 360)."""
    p1, p2 = radians(lat1), radians(lat2)
    dl = radians(lon2 - lon1)
    x = sin(dl) * cos(p2)
    y = cos(p1) * sin(p2) - sin(p1) * cos(p2) * cos(dl)
    return (degrees(atan2(x, y)) + 360.0) % 360.0


def _angle_diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """Smallest absolute difference between two bearings, degrees [0, 180]."""
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return None
    d = abs(float(a) - float(b)) % 360.0
    return d if d <= 180.0 else 360.0 - d


def _detect_arrival(
    vp_lat: np.ndarray,
    vp_lon: np.ndarray,
    stop_lat: float,
    stop_lon: float,
    radius_m: float,
    fallback_m: float,
) -> int:
    """Index of the VP where the vehicle arrived at (stop_lat, stop_lon), or -1.

    Faithful to ``dataset_builder.find_actual_arrival_time``'s rule, but returns
    the *index* (callers need the arriving VP's kinematics, not just its ts):
    first VP within ``radius_m``; else the closest-approach VP when that
    approach is within ``fallback_m``; else -1. ``vp_lat``/``vp_lon`` must be
    ordered by ascending ts (so "first" is chronological).
    """
    n = len(vp_lat)
    if n == 0:
        return -1
    dists = np.array(
        [haversine_m(float(vp_lat[i]), float(vp_lon[i]), stop_lat, stop_lon) for i in range(n)]
    )
    within = np.flatnonzero(dists <= radius_m)
    if within.size:
        return int(within[0])
    closest = int(dists.argmin())
    if dists[closest] <= fallback_m:
        return closest
    return -1


def resolve_agency_temporal(
    agency: str, tz_for_temporal: Optional[str]
) -> Tuple[str, str]:
    """Validate ``agency`` and resolve (timezone, region) for temporal features.

    Raises ``ValueError`` for an unknown agency. ``tz_for_temporal`` overrides
    the agency's default timezone when given; the region is always the agency's.
    """
    if agency not in AGENCY_TEMPORAL_DEFAULTS:
        raise ValueError(f"unknown agency {agency!r}; known: {sorted(AGENCY_TEMPORAL_DEFAULTS)}")
    tz = tz_for_temporal or AGENCY_TEMPORAL_DEFAULTS[agency]["timezone"]
    region = AGENCY_TEMPORAL_DEFAULTS[agency]["region"]
    return tz, region


def _emit_rows_for_instance(
    inst: TripInstance,
    arrival_radius_m: float,
    closest_fallback_m: float,
    max_segment_seconds: float,
    temporal_tz: str,
    temporal_region: str,
) -> List[dict]:
    """Emit the segment rows for one trip instance (see module docstring)."""
    stops = [s for s in inst.stops if s.progress_m is not None]
    if len(stops) < 2:
        return []
    n_stops = len(stops)
    polyline = inst.polyline
    shape_total_m = polyline[-1].cum_m if polyline else None

    trip_vps = inst.trip_vps.sort_values("ts", kind="stable").reset_index(drop=True)
    vp_lat = trip_vps["lat"].to_numpy(dtype=float)
    vp_lon = trip_vps["lon"].to_numpy(dtype=float)
    vp_ts = trip_vps["ts"].to_numpy()
    has_bearing = "bearing" in trip_vps.columns
    has_speed = "speed" in trip_vps.columns
    has_status = "current_status" in trip_vps.columns

    arrival_idx = np.array(
        [
            _detect_arrival(
                vp_lat, vp_lon, float(s.lat), float(s.lon),
                arrival_radius_m, closest_fallback_m,
            )
            for s in stops
        ]
    )

    rows: List[dict] = []
    for k in range(n_stops - 1):
        i_from = int(arrival_idx[k])
        i_to = int(arrival_idx[k + 1])
        if i_from == -1 or i_to == -1:
            continue
        t_from = vp_ts[i_from]
        t_to = vp_ts[i_to]
        traversal_s = (t_to - t_from) / np.timedelta64(1, "s")
        if not (0.0 < traversal_s <= max_segment_seconds):
            continue

        from_stop = stops[k]
        to_stop = stops[k + 1]

        seg_len_m = float(to_stop.progress_m - from_stop.progress_m)
        next_len_m = (
            float(stops[k + 2].progress_m - to_stop.progress_m)
            if k + 2 < n_stops
            else np.nan
        )

        entry_lat = float(vp_lat[i_from])
        entry_lon = float(vp_lon[i_from])
        entry_bearing = trip_vps["bearing"].iloc[i_from] if has_bearing else np.nan
        entry_speed = trip_vps["speed"].iloc[i_from] if has_speed else np.nan
        entry_status = trip_vps["current_status"].iloc[i_from] if has_status else None

        cross_track = np.nan
        if polyline:
            proj = project_point_to_polyline(entry_lat, entry_lon, polyline)
            cross_track = float(proj.cross_track_m)

        bearing_to_stop = _initial_bearing(
            entry_lat, entry_lon, float(to_stop.lat), float(to_stop.lon)
        )

        temporal = extract_temporal_features(
            pd.Timestamp(t_from).to_pydatetime(),
            tz=temporal_tz,
            region=temporal_region,
        )
        hour = temporal.get("hour")
        dow = temporal.get("day_of_week")

        rows.append(
            {
                "trip_id": inst.trip_id,
                "route_id": inst.route_id,
                "vehicle_id": inst.vehicle_id,
                "stop_id": to_stop.stop_id,
                "stop_sequence": to_stop.stop_sequence,
                "from_stop_id": from_stop.stop_id,
                "segment_index": k,
                "vp_ts": t_from,
                "vp_lat": entry_lat,
                "vp_lon": entry_lon,
                "vp_bearing": entry_bearing,
                "stop_lat": float(to_stop.lat),
                "stop_lon": float(to_stop.lon),
                "distance_to_stop": seg_len_m,
                "distance_to_next_stop": next_len_m,
                "shape_distance_to_stop": seg_len_m,
                "shape_progress": (
                    float(from_stop.progress_m / shape_total_m)
                    if shape_total_m
                    else np.nan
                ),
                "cross_track_error": cross_track,
                "progress_ratio": k / (n_stops - 1),
                "stops_ahead": (n_stops - 1) - (k + 1),
                "current_speed_kmh": (
                    float(entry_speed) * 3.6 if pd.notna(entry_speed) else np.nan
                ),
                "bearing_to_stop": bearing_to_stop,
                "bearing_diff": _angle_diff(entry_bearing, bearing_to_stop),
                "is_at_stop": bool(entry_status == "STOPPED_AT"),
                "actual_arrival": t_to,
                "time_to_arrival_seconds": float(traversal_s),
                "hour": hour,
                "day_of_week": dow,
                "is_weekend": temporal.get("is_weekend"),
                "is_holiday": temporal.get("is_holiday"),
                "is_peak_hour": temporal.get("is_peak_hour"),
            }
        )
    return rows


def build_segments_from_instances(
    instances: Iterable[TripInstance],
    *,
    arrival_radius_m: float,
    closest_fallback_m: float,
    max_segment_seconds: float,
    temporal_tz: str,
    temporal_region: str,
) -> pd.DataFrame:
    """Emit the segment dataset for an ordered iterable of trip instances.

    Processes instances in the order given (the resolver is responsible for a
    stable order), returning a frame with exactly :data:`OUTPUT_COLUMNS`; empty
    (with those columns) when no segment qualifies.
    """
    rows: List[dict] = []
    for inst in instances:
        rows.extend(
            _emit_rows_for_instance(
                inst, arrival_radius_m, closest_fallback_m,
                max_segment_seconds, temporal_tz, temporal_region,
            )
        )

    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    out = pd.DataFrame(rows)
    out["vp_ts"] = pd.to_datetime(out["vp_ts"], utc=True)
    out["actual_arrival"] = pd.to_datetime(out["actual_arrival"], utc=True)

    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7.0)

    for c in OUTPUT_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan

    return out[OUTPUT_COLUMNS].reset_index(drop=True)
