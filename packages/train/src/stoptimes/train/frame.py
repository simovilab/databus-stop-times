"""Build the training frame: observations + arrivals + static GTFS -> one row
per (vehicle position x upcoming stop), with spatial/kinematic/temporal-ready
features and the time-to-arrival target.

Phase 2, track T7 (docs/PLAN.md). This is the missing step between T2's
`core.arrivals.derive_arrivals` (the `Arrival` table -- ground truth) and the
model families / feature registry (T3), which is documented (in
`core.features.registry`) to consume exactly the column shape this module
produces.

Public entry point
-------------------
``build_training_frame(observations, arrivals, static, *, source, params) ->
pd.DataFrame``. Pure pandas/numpy: no Django, no database, no network, no
mutation of its inputs.

Behavioral reference: ``stoptimes.train.datasets.dataset_builder.
build_vp_training_dataset`` -- the only existing implementation, but
Django/PostgreSQL/S3-shaped and MBTA-specific. This module reproduces its
column names and spatial-feature semantics (reusing
``stoptimes.core.spatial.geo.calculate_distance_features_with_shape`` and
``ShapePolyline`` directly, not reimplementing the geometry) but drives them
from the source-agnostic `Observation`/`Arrival` contracts and a
`trip_geometry`-shaped static feed instead of the Django ORM, and joins the
target from an already-derived `Arrival` table rather than re-deriving
"first VP within N meters" itself.

Where this deliberately differs from the legacy builder, see
docs/notes/p2-t7.md.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from math import atan2, cos, degrees, radians, sin
from typing import Any, Protocol

import numpy as np
import pandas as pd

from stoptimes.core.contracts.data import ObservationKind, coerce_utc_ns
from stoptimes.core.contracts.source import Source
from stoptimes.core.domain.models import Stop
from stoptimes.core.spatial.geo import ShapePolyline, calculate_distance_features_with_shape

__all__ = ["FrameParams", "StaticGeometry", "build_training_frame"]

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Static geometry duck type (same protocol `core.arrivals.StaticGeometry` uses)
# ---------------------------------------------------------------------------


class StaticGeometry(Protocol):
    """What `build_training_frame` needs from a loaded static feed.

    `stoptimes.core.gtfs.static.GtfsStaticLoader` (T1) satisfies this
    already; tests may pass any object with the same method.
    """

    def trip_geometry(
        self, trip_id: str
    ) -> tuple[list[tuple[float, float, float | None]], list[Stop]]: ...


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameParams:
    """Tunable knobs for :func:`build_training_frame`."""

    max_stops_ahead: int = 5
    """Emit at most this many upcoming stops per vehicle-position row."""

    keep_unlabeled: bool = False
    """When False (default), rows whose target can't be established (no
    matching `Arrival` row for that trip/day/stop) are dropped. When True,
    they are kept with `actual_arrival`/`time_to_arrival_seconds` as NaT/NaN
    so a caller can inspect what's missing."""


# ---------------------------------------------------------------------------
# Small geometry/time helpers (kept local: not "reimplementing" the shape
# projection -- that's `calculate_distance_features_with_shape` -- just the
# plain trig/haversine bits every builder in this codebase already repeats
# locally, e.g. `train.datasets.dataset_builder`/`core.segments`).
# ---------------------------------------------------------------------------

_EARTH_R_M = 6_371_000.0


def _haversine_matrix(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Vectorized haversine distance (meters) between every (lat1,lon1) x
    (lat2,lon2) pair, via numpy broadcasting. Returns an (len(lat1), len(lat2)) matrix."""
    lat1r = np.radians(lat1)[:, None]
    lon1r = np.radians(lon1)[:, None]
    lat2r = np.radians(lat2)[None, :]
    lon2r = np.radians(lon2)[None, :]
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return c * _EARTH_R_M


def _initial_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial great-circle bearing from point 1 to point 2, in degrees [0, 360)."""
    phi1, phi2 = radians(lat1), radians(lat2)
    dlambda = radians(lon2 - lon1)
    x = sin(dlambda) * cos(phi2)
    y = cos(phi1) * sin(phi2) - sin(phi1) * cos(phi2) * cos(dlambda)
    return (degrees(atan2(x, y)) + 360.0) % 360.0


def _angle_diff(a: float | None, b: float | None) -> float | None:
    """Smallest absolute difference between two bearings, in degrees [0, 180]."""
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return None
    d = abs(float(a) - float(b)) % 360.0
    return d if d <= 180.0 else 360.0 - d


def _service_date(ts: pd.Timestamp, tz: str) -> str:
    """Fallback service-day string (YYYYMMDD) for a VP with no `start_date`.

    Mirrors `core.arrivals._service_date`'s rule so a frame built here joins
    cleanly against an `Arrival` table whose rows used the same fallback for
    the same source.
    """
    local = ts.tz_convert(tz) if ts.tzinfo is not None else ts.tz_localize("UTC").tz_convert(tz)
    return local.strftime("%Y%m%d")


_OUTPUT_COLUMNS: list[str] = [
    # identifiers
    "trip_id", "route_id", "vehicle_id", "stop_id", "stop_sequence",
    # position
    "vp_ts", "vp_lat", "vp_lon", "vp_bearing", "stop_lat", "stop_lon",
    # spatial
    "distance_to_stop", "distance_to_next_stop", "shape_distance_to_stop",
    "shape_progress", "cross_track_error", "progress_ratio", "stops_ahead",
    # kinematic
    "current_speed_kmh", "bearing_to_stop", "bearing_diff",
    # status
    "is_at_stop",
    # target
    "actual_arrival", "time_to_arrival_seconds",
]


def _empty_frame() -> pd.DataFrame:
    dtypes: dict[str, Any] = {
        "trip_id": "string",
        "route_id": "string",
        "vehicle_id": "string",
        "stop_id": "string",
        "stop_sequence": "int64",
        "vp_ts": "datetime64[ns, UTC]",
        "vp_lat": "float64",
        "vp_lon": "float64",
        "vp_bearing": "float64",
        "stop_lat": "float64",
        "stop_lon": "float64",
        "distance_to_stop": "float64",
        "distance_to_next_stop": "float64",
        "shape_distance_to_stop": "float64",
        "shape_progress": "float64",
        "cross_track_error": "float64",
        "progress_ratio": "float64",
        "stops_ahead": "int64",
        "current_speed_kmh": "float64",
        "bearing_to_stop": "float64",
        "bearing_diff": "float64",
        "is_at_stop": "bool",
        "actual_arrival": "datetime64[ns, UTC]",
        "time_to_arrival_seconds": "float64",
    }
    return pd.DataFrame({c: pd.Series(dtype=dtypes[c]) for c in _OUTPUT_COLUMNS})


def _resolve_trip_geometry(
    static: StaticGeometry, trip_id: str
) -> tuple[list[Stop], "ShapePolyline | None"] | None:
    try:
        shape_points, stops = static.trip_geometry(trip_id)
    except (ValueError, KeyError):
        return None
    if not stops:
        return None
    ordered_stops = sorted(stops, key=lambda s: s.stop_sequence)
    shape: ShapePolyline | None = None
    if shape_points and len(shape_points) >= 2:
        shape = ShapePolyline([(pt[0], pt[1]) for pt in shape_points])
    return ordered_stops, shape


def _vp_groups(observations: pd.DataFrame, source: Source) -> pd.DataFrame:
    """Vehicle-position observations only, cleaned and with a resolved
    `start_date`, sorted for stable per-(trip, day) grouping."""
    if observations.empty:
        return observations
    vp = observations[observations["kind"] == ObservationKind.VEHICLE_POSITION.value].copy()
    for col in ("lat", "lon", "ts", "trip_id", "route_id", "vehicle_id", "bearing", "speed",
                "current_status", "start_date"):
        if col not in vp.columns:
            vp[col] = pd.NA
    vp = vp[vp["lat"].notna() & vp["lon"].notna() & vp["ts"].notna() & vp["trip_id"].notna()]
    if vp.empty:
        return vp
    vp["ts"] = coerce_utc_ns(vp["ts"])
    vp["start_date"] = vp.apply(
        lambda r: r["start_date"] if pd.notna(r.get("start_date")) else _service_date(r["ts"], source.timezone),
        axis=1,
    )
    return vp.sort_values(["trip_id", "start_date", "ts"], kind="stable").reset_index(drop=True)


def _arrival_lookup(arrivals: pd.DataFrame, source: Source) -> dict[tuple[str, str, int], pd.Timestamp]:
    """(trip_id, start_date, stop_sequence) -> arrival_time, filtered to this source."""
    if arrivals is None or arrivals.empty:
        return {}
    df = arrivals
    if "source_id" in df.columns:
        df = df[df["source_id"] == source.id]
    if df.empty:
        return {}
    return {
        (str(r.trip_id), str(r.start_date), int(r.stop_sequence)): pd.Timestamp(r.arrival_time)
        for r in df.itertuples(index=False)
    }


def build_training_frame(
    observations: pd.DataFrame,
    arrivals: pd.DataFrame,
    static: StaticGeometry,
    *,
    source: Source,
    params: FrameParams | None = None,
) -> pd.DataFrame:
    """Build the training frame: one row per (vehicle position, upcoming stop).

    Args:
        observations: `Observation`-shaped rows (`contracts.data.
            OBSERVATION_VP_SCHEMA`); only `vehicle_position` kind rows are
            used, `trip_update` rows are ignored (they carry no lat/lon).
        arrivals: the `Arrival` table (`contracts.data.ARRIVAL_SCHEMA`,
            e.g. from `core.arrivals.derive_arrivals`) -- the target source.
        static: anything exposing `trip_geometry(trip_id) -> (shape_points,
            ordered_stops)`, e.g. `stoptimes.core.gtfs.static.GtfsStaticLoader`.
        source: the `Source` these observations/arrivals came from -- supplies
            `source_id` (to filter `arrivals`) and `timezone` (for a missing
            `start_date` fallback, matching `core.arrivals`'s rule).
        params: tuning knobs; defaults to `FrameParams()`.

    Returns:
        A DataFrame with exactly the columns `dataset_builder.
        build_vp_training_dataset` documents for its non-temporal groups
        (identifiers, position, spatial, kinematic, status, target) --
        temporal features are the feature registry's job, not this
        function's; it only needs to leave `vp_ts` as a UTC timestamp for
        the registry to derive them from.

        Rows whose target can't be established (no matching `Arrival` row)
        are dropped unless `params.keep_unlabeled=True`. A trip with no
        static geometry (unknown `trip_id`, or no stops) contributes no
        rows. A trip with stops but no shape gets `shape_*`/`cross_track_error`
        as NaN rather than a fabricated value.
    """
    if params is None:
        params = FrameParams()

    vp = _vp_groups(observations, source)
    if vp.empty:
        _log.info("build_training_frame: no vehicle_position observations to process")
        return _empty_frame()

    arrival_lookup = _arrival_lookup(arrivals, source)

    geometry_cache: dict[str, tuple[list[Stop], "ShapePolyline | None"] | None] = {}
    rows: list[dict[str, Any]] = []
    n_no_geometry = 0
    n_no_target = 0
    n_candidates = 0

    for (trip_id, start_date), group in vp.groupby(["trip_id", "start_date"], sort=False):
        trip_id = str(trip_id)
        start_date = str(start_date)
        if trip_id not in geometry_cache:
            geometry_cache[trip_id] = _resolve_trip_geometry(static, trip_id)
        geometry = geometry_cache[trip_id]
        if geometry is None:
            n_no_geometry += len(group)
            continue
        stops, shape = geometry
        n_stops = len(stops)
        if n_stops == 0:
            continue
        trip_total_segments = max(n_stops - 1, 1)

        group = group.sort_values("ts", kind="stable").reset_index(drop=True)
        n_vps = len(group)

        vp_lat_arr = group["lat"].to_numpy(dtype=float)
        vp_lon_arr = group["lon"].to_numpy(dtype=float)
        vp_ts_arr = group["ts"].to_numpy()
        vp_bearing_arr = group["bearing"].to_numpy()
        vp_speed_arr = group["speed"].to_numpy()
        vp_status_arr = group["current_status"].to_numpy()
        vp_route_id = group["route_id"].iloc[0] if "route_id" in group.columns else None
        vp_vehicle_id = group["vehicle_id"].iloc[0] if "vehicle_id" in group.columns else None

        stop_lat_arr = np.array([s.lat for s in stops], dtype=float)
        stop_lon_arr = np.array([s.lon for s in stops], dtype=float)

        dist_matrix = _haversine_matrix(vp_lat_arr, vp_lon_arr, stop_lat_arr, stop_lon_arr)
        closest_stop_idx_per_vp = dist_matrix.argmin(axis=1)

        for i in range(n_vps):
            vp_lat = float(vp_lat_arr[i])
            vp_lon = float(vp_lon_arr[i])
            vp_ts = vp_ts_arr[i]
            vp_bearing = vp_bearing_arr[i]
            vp_bearing = None if pd.isna(vp_bearing) else float(vp_bearing)
            vp_speed = vp_speed_arr[i]
            vp_position = {"lat": vp_lat, "lon": vp_lon, "bearing": vp_bearing}

            closest_idx = int(closest_stop_idx_per_vp[i])
            end = min(closest_idx + params.max_stops_ahead, n_stops)

            for j in range(closest_idx, end):
                stop = stops[j]
                next_stop = stops[j + 1] if j + 1 < n_stops else None

                stop_payload = {
                    "stop_id": stop.stop_id,
                    "lat": stop.lat,
                    "lon": stop.lon,
                    "stop_order": j,
                    "total_segments": trip_total_segments,
                }
                next_stop_payload = (
                    {"stop_id": next_stop.stop_id, "lat": next_stop.lat, "lon": next_stop.lon}
                    if next_stop is not None
                    else None
                )

                spatial = calculate_distance_features_with_shape(
                    vp_position,
                    stop_payload,
                    next_stop_payload,
                    shape=shape,
                    vehicle_stop_order=closest_idx,
                    total_segments=trip_total_segments,
                )

                bearing_to_stop = _initial_bearing(vp_lat, vp_lon, stop.lat, stop.lon)

                n_candidates += 1
                key = (trip_id, start_date, int(stop.stop_sequence))
                actual_arrival = arrival_lookup.get(key)
                if actual_arrival is None:
                    n_no_target += 1
                    if not params.keep_unlabeled:
                        continue
                    time_to_arrival = np.nan
                else:
                    time_to_arrival = (actual_arrival - pd.Timestamp(vp_ts)).total_seconds()

                rows.append(
                    {
                        "trip_id": trip_id,
                        "route_id": vp_route_id,
                        "vehicle_id": vp_vehicle_id,
                        "stop_id": stop.stop_id,
                        "stop_sequence": int(stop.stop_sequence),
                        "vp_ts": vp_ts,
                        "vp_lat": vp_lat,
                        "vp_lon": vp_lon,
                        "vp_bearing": vp_bearing,
                        "stop_lat": stop.lat,
                        "stop_lon": stop.lon,
                        "distance_to_stop": spatial.get("distance_to_stop"),
                        "distance_to_next_stop": spatial.get("distance_to_next_stop"),
                        "shape_distance_to_stop": spatial.get("shape_distance_to_stop"),
                        "shape_progress": spatial.get("shape_progress"),
                        "cross_track_error": spatial.get("cross_track_error"),
                        "progress_ratio": spatial.get("progress_ratio"),
                        "stops_ahead": int(j - closest_idx),
                        "current_speed_kmh": float(vp_speed) * 3.6 if pd.notna(vp_speed) else np.nan,
                        "bearing_to_stop": bearing_to_stop,
                        "bearing_diff": _angle_diff(vp_bearing, bearing_to_stop),
                        "is_at_stop": bool(vp_status_arr[i] == "STOPPED_AT"),
                        "actual_arrival": actual_arrival if actual_arrival is not None else pd.NaT,
                        "time_to_arrival_seconds": time_to_arrival,
                    }
                )

    _log.info(
        "build_training_frame: %d candidate (vp, stop) pairs, %d dropped for missing "
        "geometry, %d missing a target arrival, %d rows emitted",
        n_candidates, n_no_geometry, n_no_target, len(rows),
    )

    if not rows:
        return _empty_frame()

    df = pd.DataFrame(rows)
    df["vp_ts"] = coerce_utc_ns(df["vp_ts"])
    df["actual_arrival"] = coerce_utc_ns(df["actual_arrival"])
    for c in ("trip_id", "route_id", "vehicle_id", "stop_id"):
        df[c] = df[c].astype("string")
    df["stop_sequence"] = df["stop_sequence"].astype("int64")
    df["stops_ahead"] = df["stops_ahead"].astype("int64")
    df["is_at_stop"] = df["is_at_stop"].astype("bool")

    df = df.sort_values(["trip_id", "vp_ts", "stop_sequence"], kind="stable")
    return df[_OUTPUT_COLUMNS].reset_index(drop=True)
