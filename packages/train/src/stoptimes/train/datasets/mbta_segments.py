"""Segment ETA training-dataset builder for MBTA vehicle positions (roadmap 3.1,
generalized to MBTA).

MBTA's raw ``VehiclePositions`` already carry REAL GTFS ids (``trip_id``,
``route_id``, ``current_stop_sequence``, ``stop_id``) -- unlike bUCR's navsat
AVL feed, no trip inference / map-matching is needed to recover which route,
direction, and stop pattern a position belongs to. A ``trip_id`` maps
directly through the static GTFS ``trips`` -> ``shape_id`` and through
``stop_times`` -> the ordered ``(stop_id, stop_sequence)`` list.

Emits the **same schema** (:data:`feature_engineering.segment_dataset_builder.
OUTPUT_COLUMNS`) as :func:`feature_engineering.segment_dataset_builder.
build_bucr_segment_dataset`, using the identical arrival-detection rule,
drop rules, and per-segment feature/target semantics -- see that module's
docstring for the full column-by-column contract. This lets both agencies'
segment datasets feed the same trainers / leakage probe / metrics unchanged.

Shared pure helpers (``_detect_arrival``, ``_initial_bearing``, ``_angle_diff``)
are duplicated here rather than imported from ``segment_dataset_builder`` --
they are small, and duplicating avoids any risk of perturbing that module's
(tested, deployed) behavior. ``OUTPUT_COLUMNS`` and the shared threshold
constants ARE imported directly (they are inert data, not behavior).

Pure/deterministic: no I/O, no network, no mutation of inputs. Spatial math
is reused from ``etaval.spatial.polyline``; GTFS parsing from
``feature_engineering.bucr_gtfs`` (a generic loader despite its name -- it
reads any standard GTFS zip/dir, MBTA's included); temporal features from
``feature_engineering.temporal``. Runs under ``uv run --group bucr`` (needs
``etaval``).
"""

from __future__ import annotations

from math import atan2, cos, degrees, radians, sin
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from etaval.spatial.polyline import PolylinePoint, assign_stops_monotonic, haversine_m, project_point_to_polyline

from core.config import AGENCY_TEMPORAL_DEFAULTS
from feature_engineering.bucr_gtfs import BucrGtfs, BucrGtfsError, build_shape_polyline, stops_for_trip
from feature_engineering.segment_dataset_builder import (
    ARRIVAL_RADIUS_M,
    CLOSEST_APPROACH_FALLBACK_M,
    MAX_SEGMENT_SECONDS,
    OUTPUT_COLUMNS,
)
from feature_engineering.temporal import extract_temporal_features

__all__ = ["build_mbta_segment_dataset"]


def _initial_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial great-circle bearing point1->point2, degrees [0, 360).

    Duplicate of ``segment_dataset_builder._initial_bearing`` -- see module
    docstring for why this is duplicated rather than imported.
    """
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

    Duplicate of ``segment_dataset_builder._detect_arrival`` (first VP within
    ``radius_m``; else closest-approach within ``fallback_m``; else -1).
    ``vp_lat``/``vp_lon`` must be ordered by ascending ts.
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


def build_mbta_segment_dataset(
    vps_df: pd.DataFrame,
    gtfs: BucrGtfs,
    *,
    arrival_radius_m: float = ARRIVAL_RADIUS_M,
    closest_fallback_m: float = CLOSEST_APPROACH_FALLBACK_M,
    max_segment_seconds: float = MAX_SEGMENT_SECONDS,
    agency: str = "mbta",
    tz_for_temporal: Optional[str] = None,
) -> pd.DataFrame:
    """Build the segment training dataset from raw MBTA VehiclePositions.

    Args:
        vps_df: Raw MBTA VP frame (e.g. from
            ``rt_pipeline.storage.s3_writer.read_vehicle_positions``) -- must
            carry ``trip_id``, ``vehicle_id``, ``ts``, ``lat``, ``lon`` (and
            optionally ``bearing``, ``speed``, ``current_status``). Each
            distinct real GTFS ``trip_id`` is one trip instance; rows with a
            null/missing ``trip_id`` (unassigned VPs) are dropped. NO trip
            inference is performed -- ``trip_id`` maps straight through
            ``gtfs``.
        gtfs: A :class:`~feature_engineering.bucr_gtfs.BucrGtfs` loaded via
            ``feature_engineering.bucr_gtfs.load_gtfs`` from the static GTFS
            snapshot in effect for this VP window.
        arrival_radius_m: "arrived" radius for the position-derived target.
        closest_fallback_m: closest-approach fallback when no fix lands
            inside ``arrival_radius_m``.
        max_segment_seconds: drop segments whose observed traversal exceeds
            this (a hidden layover / mis-paired arrival, not a real leg).
        agency: temporal-feature agency key (default ``"mbta"`` ->
            America/New_York / region US_MA).
        tz_for_temporal: explicit timezone override; defaults to the
            agency's.

    Returns:
        A DataFrame with :data:`segment_dataset_builder.OUTPUT_COLUMNS`. One
        row per (trip instance x traversed segment k->k+1) where arrivals at
        both endpoints were detected, chronologically ordered, and the
        traversal is in ``(0, max_segment_seconds]``. Empty (with the right
        columns) when no segment qualifies. Column semantics mirror
        ``build_bucr_segment_dataset`` exactly (see that function's
        docstring); the only difference is provenance: ``route_id`` and
        ``shape_id`` (used internally, not emitted) come directly from the
        GTFS ``trips`` table keyed by the real ``trip_id``, not from parsing
        a synthetic id.

    Raises:
        TypeError: ``vps_df`` is not a DataFrame or ``gtfs`` wrong type.
        ValueError: required columns missing, or ``agency`` unknown.
    """
    if not isinstance(vps_df, pd.DataFrame):
        raise TypeError(f"vps_df must be a DataFrame, got {type(vps_df)!r}")
    if not isinstance(gtfs, BucrGtfs):
        raise TypeError(f"gtfs must be a BucrGtfs, got {type(gtfs)!r}")

    required = {"trip_id", "vehicle_id", "ts", "lat", "lon"}
    missing = [c for c in required if c not in vps_df.columns]
    if missing:
        raise ValueError(f"vps_df missing required columns: {missing}")
    if agency not in AGENCY_TEMPORAL_DEFAULTS:
        raise ValueError(f"unknown agency {agency!r}; known: {sorted(AGENCY_TEMPORAL_DEFAULTS)}")

    temporal_tz = tz_for_temporal or AGENCY_TEMPORAL_DEFAULTS[agency]["timezone"]
    temporal_region = AGENCY_TEMPORAL_DEFAULTS[agency]["region"]

    if vps_df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = vps_df.copy(deep=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["trip_id"])
    df["trip_id"] = df["trip_id"].astype(str)
    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    # Real-world GTFS csvs routinely load id columns as ``object`` dtype with
    # a MIX of str/int python values (some rows numeric-looking, pandas'
    # per-chunk type sniffing doesn't unify them) -- MBTA's static feed does
    # this for stop_times.trip_id/stop_id. A raw ``==`` comparison against a
    # string then silently matches nothing. Normalize every id column used
    # for cross-frame joins to str on a local copy so lookups here (and the
    # bucr_gtfs helpers called on this copy) are dtype-safe; the caller's
    # `gtfs` object itself is never mutated.
    gtfs = BucrGtfs(
        routes=gtfs.routes,
        trips=gtfs.trips.assign(
            trip_id=gtfs.trips["trip_id"].astype(str),
            route_id=gtfs.trips["route_id"].astype(str),
            shape_id=gtfs.trips["shape_id"].astype(str),
        ),
        stops=gtfs.stops.assign(stop_id=gtfs.stops["stop_id"].astype(str)),
        stop_times=gtfs.stop_times.assign(
            trip_id=gtfs.stop_times["trip_id"].astype(str),
            stop_id=gtfs.stop_times["stop_id"].astype(str),
        ),
        shapes=gtfs.shapes.assign(shape_id=gtfs.shapes["shape_id"].astype(str)),
    )

    # Index by string trip_id for O(1) lookup against vps_df's trip_id.
    trips = gtfs.trips.drop_duplicates(subset="trip_id").set_index("trip_id")

    polyline_cache: Dict[str, Optional[List[PolylinePoint]]] = {}

    rows: List[dict] = []
    for trip_id, trip_vps in df.groupby("trip_id", sort=True):
        trip_id_str = str(trip_id)
        if trip_id_str not in trips.index:
            continue
        trip_row = trips.loc[trip_id_str]
        shape_id = str(trip_row["shape_id"])
        route_id = str(trip_row["route_id"])

        if shape_id not in polyline_cache:
            try:
                polyline_cache[shape_id] = build_shape_polyline(gtfs, shape_id)
            except BucrGtfsError:
                polyline_cache[shape_id] = None
        polyline = polyline_cache[shape_id]
        if not polyline:
            continue

        try:
            raw_stops = stops_for_trip(gtfs, trip_id_str)
        except BucrGtfsError:
            continue
        stops = assign_stops_monotonic(raw_stops, polyline)
        stops = [s for s in stops if s.progress_m is not None]
        if len(stops) < 2:
            continue
        n_stops = len(stops)
        shape_total_m = polyline[-1].cum_m if polyline else None

        trip_vps = trip_vps.sort_values("ts", kind="stable").reset_index(drop=True)
        vp_lat = trip_vps["lat"].to_numpy(dtype=float)
        vp_lon = trip_vps["lon"].to_numpy(dtype=float)
        vp_ts = trip_vps["ts"].to_numpy()
        has_bearing = "bearing" in trip_vps.columns
        has_speed = "speed" in trip_vps.columns
        has_status = "current_status" in trip_vps.columns
        vehicle_id = trip_vps["vehicle_id"].iloc[0]

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
                    "trip_id": trip_id_str,
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

    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7.0)

    for c in OUTPUT_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan

    return out[OUTPUT_COLUMNS].reset_index(drop=True)
