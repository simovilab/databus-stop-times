"""Segment ETA training-dataset builder for inferred BUCR trips (roadmap Step 5, 3.1).

Turns the *inferred* BUCR vehicle-position frame (the output of
``bucr_trip_inference.infer_bucr_trips`` -- one row per assigned, non-anomalous
point, carrying ``trip_id``/``route_id``/``stop_id``/``current_stop_sequence``)
into a **segment** training dataset: one row per
(trip instance x stop-to-stop segment), where the target is the *observed*
seconds a vehicle took to traverse from its arrival at stop k to its arrival
at stop k+1.

Why segment-based (vs. the stop-level ``dataset_builder.build_vp_training_dataset``):
a segment's traversal time is a stable, additive quantity -- stop-level ETA at
any horizon is just the sum of the predicted segment times ahead. See
``docs/BUCR_DATASET_PROMPT.md`` step 5 / roadmap 3.1.

Schema parity (IMPORTANT): the emitted frame carries exactly the columns
``models.common.data.ETADataset`` and ``models/train_all_models.py`` consume
(the same ``FEATURE_GROUPS`` the MBTA stop-level dataset uses), with the
segment traversal time mapped onto ``time_to_arrival_seconds`` -- so the model
families train on this frame **unchanged**. Each stop-level feature is
reinterpreted for a segment (documented per-column in
:func:`build_bucr_segment_dataset`): e.g. ``distance_to_stop`` is the segment's
along-shape length, ``stops_ahead`` is stops-remaining after the segment's
destination, ``is_at_stop`` is whether the entry VP reported ``STOPPED_AT``.

Arrival detection: reuses the same **50 m first-within-threshold, else
closest-approach<=200 m** rule as ``dataset_builder.find_actual_arrival_time``
so segment targets are position-derived (never schedule-derived -- BUCR's
offset timetable does not poison labels). That helper is re-implemented here
rather than imported because ``feature_engineering.dataset_builder`` imports
``django`` and ``feature_engineering.rt_source`` (-> ``rt_pipeline.storage``)
at module load, neither of which is installed under the ``bucr`` uv group --
the same reason ``bucr_trip_inference`` re-declares ``OUTPUT_COLUMNS`` instead
of importing it. The logic is a faithful copy; see :func:`_detect_arrival`.

Pure/deterministic: no I/O, no network, no mutation of inputs. Spatial math
(cross-track projection, haversine) is reused from ``etaval.spatial.polyline``;
temporal features from ``feature_engineering.temporal`` (``agency="bucr"`` ->
America/Costa_Rica / region CR via ``core.config.AGENCY_TEMPORAL_DEFAULTS``).
Runs under ``uv run --group bucr``.
"""

from __future__ import annotations

from math import atan2, cos, degrees, radians, sin
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from etaval.spatial.polyline import haversine_m, project_point_to_polyline

from core.config import AGENCY_TEMPORAL_DEFAULTS
from feature_engineering.bucr_gtfs import RouteDirectionCandidate
from feature_engineering.temporal import extract_temporal_features

__all__ = [
    "OUTPUT_COLUMNS",
    "ARRIVAL_RADIUS_M",
    "CLOSEST_APPROACH_FALLBACK_M",
    "MAX_SEGMENT_SECONDS",
    "build_bucr_segment_dataset",
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
# mis-paired arrival) and dropped -- BUCR campus segments are short; 30 min is
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


def _initial_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial great-circle bearing point1->point2, degrees [0, 360). Copy of
    ``dataset_builder._initial_bearing`` (see module note on the import chain)."""
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

    Faithful copy of ``dataset_builder.find_actual_arrival_time``'s rule, but
    returns the *index* (callers need the arriving VP's kinematics, not just
    its ts): first VP within ``radius_m``; else the closest-approach VP when
    that approach is within ``fallback_m``; else -1. ``vp_lat``/``vp_lon`` must
    be ordered by ascending ts (so "first" is chronological).
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


def _parse_trip_id(trip_id: str) -> Optional[Tuple[str, str, int, str]]:
    """Recover ``(vehicle_id, route_id, direction_id, shape_id)`` from a trip id.

    Inverse of ``bucr_trip_inference._make_trip_id``'s recipe
    ``bucr:{vehicle}:{route}:{direction}:{shape}:{start}`` -- none of vehicle
    (plate), route, or BUCR shape ids contain ``:``, so a plain 6-field split
    is unambiguous. Returns ``None`` for any id not matching the recipe (so an
    unexpected id skips the trip rather than crashing the build).
    """
    parts = str(trip_id).split(":")
    if len(parts) != 6 or parts[0] != "bucr":
        return None
    _, vehicle_id, route_id, direction_str, shape_id, _start = parts
    try:
        direction_id = int(direction_str)
    except ValueError:
        return None
    return vehicle_id, route_id, direction_id, shape_id


def _candidate_index(
    candidates: Sequence[RouteDirectionCandidate],
) -> Dict[Tuple[str, int, str], RouteDirectionCandidate]:
    """Index candidates by ``(route_id, direction_id, shape_id)`` for O(1) lookup."""
    return {(c.route_id, c.direction_id, c.shape_id): c for c in candidates}


def build_bucr_segment_dataset(
    inferred_df: pd.DataFrame,
    candidates: Sequence[RouteDirectionCandidate],
    *,
    arrival_radius_m: float = ARRIVAL_RADIUS_M,
    closest_fallback_m: float = CLOSEST_APPROACH_FALLBACK_M,
    max_segment_seconds: float = MAX_SEGMENT_SECONDS,
    agency: str = "bucr",
    tz_for_temporal: Optional[str] = None,
) -> pd.DataFrame:
    """Build the segment training dataset from an inferred BUCR VP frame.

    Args:
        inferred_df: Output of ``bucr_trip_inference.infer_bucr_trips`` -- must
            carry ``trip_id``, ``vehicle_id``, ``ts``, ``lat``, ``lon``,
            ``speed``, ``current_status`` (and optionally ``bearing``). Each
            distinct ``trip_id`` is one trip instance.
        candidates: The same ``RouteDirectionCandidate`` list inference ran
            against; used to recover each trip's ordered stops (with
            ``progress_m`` along-shape) and its polyline (for cross-track).
        arrival_radius_m: "arrived" radius for the position-derived target.
        closest_fallback_m: closest-approach fallback when no fix lands inside
            ``arrival_radius_m``.
        max_segment_seconds: drop segments whose observed traversal exceeds this
            (a hidden layover / mis-paired arrival, not a real leg).
        agency: temporal-feature agency key (default ``"bucr"`` ->
            America/Costa_Rica / region CR).
        tz_for_temporal: explicit timezone override; defaults to the agency's.

    Returns:
        A DataFrame with :data:`OUTPUT_COLUMNS`. One row per
        (trip instance x traversed segment k->k+1) where arrivals at BOTH
        endpoints were detected, are chronologically ordered, and the traversal
        is in ``(0, max_segment_seconds]``. Empty (with the right columns) when
        no segment qualifies. Column semantics per segment k->k+1 (entry =
        arrival at stop k, destination = stop k+1):
          * ``stop_id``/``stop_sequence``/``stop_lat``/``stop_lon`` -> destination k+1
          * ``from_stop_id`` -> entry stop k
          * ``vp_ts``/``vp_lat``/``vp_lon``/``vp_bearing``/``vp_speed`` -> entry VP
          * ``distance_to_stop`` / ``shape_distance_to_stop`` -> segment along-shape length
          * ``distance_to_next_stop`` -> next segment's along-shape length (NaN if last)
          * ``shape_progress`` -> progress_m(k) / shape total (fraction done at entry)
          * ``progress_ratio`` -> k / (n_stops-1) (stop-index fraction)
          * ``cross_track_error`` -> entry VP's cross-track to the assigned shape
          * ``stops_ahead`` -> stops remaining after the destination (stops-remaining)
          * ``is_at_stop`` -> entry VP reported ``STOPPED_AT``
          * ``time_to_arrival_seconds`` -> observed arrival(k+1) - arrival(k) seconds
          * ``actual_arrival`` -> arrival(k+1) ts

    Raises:
        TypeError: ``inferred_df`` is not a DataFrame or ``candidates`` wrong type.
        ValueError: required columns missing, or ``agency`` unknown.
    """
    if not isinstance(inferred_df, pd.DataFrame):
        raise TypeError(f"inferred_df must be a DataFrame, got {type(inferred_df)!r}")
    if not isinstance(candidates, (list, tuple)):
        raise TypeError(f"candidates must be a list/tuple, got {type(candidates)!r}")

    required = {"trip_id", "vehicle_id", "ts", "lat", "lon"}
    missing = [c for c in required if c not in inferred_df.columns]
    if missing:
        raise ValueError(f"inferred_df missing required columns: {missing}")
    if agency not in AGENCY_TEMPORAL_DEFAULTS:
        raise ValueError(f"unknown agency {agency!r}; known: {sorted(AGENCY_TEMPORAL_DEFAULTS)}")

    temporal_tz = tz_for_temporal or AGENCY_TEMPORAL_DEFAULTS[agency]["timezone"]
    temporal_region = AGENCY_TEMPORAL_DEFAULTS[agency]["region"]

    if inferred_df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    cand_by_key = _candidate_index(candidates)

    df = inferred_df.copy(deep=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")

    rows: List[dict] = []
    for trip_id, trip_vps in df.groupby("trip_id", sort=True):
        parsed = _parse_trip_id(str(trip_id))
        if parsed is None:
            continue
        vehicle_id, route_id, direction_id, shape_id = parsed
        candidate = cand_by_key.get((route_id, direction_id, shape_id))
        if candidate is None:
            continue

        stops = [s for s in candidate.stops if s.progress_m is not None]
        if len(stops) < 2:
            continue
        n_stops = len(stops)
        shape_total_m = candidate.polyline[-1].cum_m if candidate.polyline else None

        trip_vps = trip_vps.sort_values("ts", kind="stable").reset_index(drop=True)
        vp_lat = trip_vps["lat"].to_numpy(dtype=float)
        vp_lon = trip_vps["lon"].to_numpy(dtype=float)
        vp_ts = trip_vps["ts"].to_numpy()
        has_bearing = "bearing" in trip_vps.columns
        has_speed = "speed" in trip_vps.columns
        has_status = "current_status" in trip_vps.columns

        # Arrival VP index per ordered stop (or -1 if never reached).
        arrival_idx = np.array(
            [
                _detect_arrival(
                    vp_lat, vp_lon, float(s.lat), float(s.lon),
                    arrival_radius_m, closest_fallback_m,
                )
                for s in stops
            ]
        )

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
            entry_bearing = (
                trip_vps["bearing"].iloc[i_from] if has_bearing else np.nan
            )
            entry_speed = trip_vps["speed"].iloc[i_from] if has_speed else np.nan
            entry_status = (
                trip_vps["current_status"].iloc[i_from] if has_status else None
            )

            cross_track = np.nan
            if candidate.polyline:
                proj = project_point_to_polyline(entry_lat, entry_lon, candidate.polyline)
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
                    "trip_id": trip_id,
                    "route_id": route_id,
                    "vehicle_id": vehicle_id,
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

    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    out = pd.DataFrame(rows)
    out["vp_ts"] = pd.to_datetime(out["vp_ts"], utc=True)
    out["actual_arrival"] = pd.to_datetime(out["actual_arrival"], utc=True)

    # Cyclical encodings (match dataset_builder exactly).
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7.0)

    for c in OUTPUT_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan

    return out[OUTPUT_COLUMNS].reset_index(drop=True)
