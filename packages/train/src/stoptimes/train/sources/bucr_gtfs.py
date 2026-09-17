"""Pure loader: BUCR static GTFS (zip or extracted dir) -> shape/stop structures.

Loads the five GTFS files trip inference needs (``routes.txt``, ``trips.txt``,
``stops.txt``, ``stop_times.txt``, ``shapes.txt``) and exposes them in the
shape ready to hand to ``etaval.spatial.polyline`` -- see
``docs/BUCR_DATASET_PROMPT.md`` step 3. This module does NOT do trip
inference itself (matching a raw GPS trace to a route+direction+trip
instance); it only turns the static feed into candidate polylines and
monotonic-assigned stop sequences that the next step scores traces against.

Pure/deterministic, no network: sourcing (S3 snapshot / direct download) is
a separate concern (see docs/BUCR_DATASET_PROMPT.md), this module only reads
a local path. Mirrors the error-handling/style of
``feature_engineering/navsat_adapter.py``.

``etaval`` (the map-matching library this module reuses) lives in a sibling
repo, not on PyPI -- ``git.no_sync/etaval`` next to this repo's root. It is
wired in as the ``bucr`` dependency group (see ``pyproject.toml``), installed
as a real, non-editable package rather than via ``.pth``/editable mechanisms,
so ``import etaval`` works under a plain ``uv run --group bucr`` without
relying on any ``sys.path`` shim. Callers that need this module must run with
that group active (e.g. ``uv run --group bucr ...``).
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd

from etaval.domain.models import Stop as EtavalStop
from etaval.spatial.polyline import (
    PolylinePoint,
    assign_stops_monotonic,
    build_polyline,
)

# GTFS files this loader reads. All five are required -- a feed missing any
# of them cannot support shape-based trip inference.
REQUIRED_FILES: Tuple[str, ...] = (
    "routes.txt",
    "trips.txt",
    "stops.txt",
    "stop_times.txt",
    "shapes.txt",
)


class BucrGtfsError(ValueError):
    """The feed at the given path is not a usable static GTFS, or a lookup failed."""

__all__ = [
    "BucrGtfsError",
    "BucrGtfs",
    "RouteDirectionCandidate",
    "load_gtfs",
    "route_directions",
    "shape_ids_for_route_direction",
    "shape_points",
    "build_shape_polyline",
    "stops_for_trip",
    "route_direction_candidates",
]


@dataclass(frozen=True)
class BucrGtfs:
    """The subset of a static GTFS feed this loader parses, as plain frames."""

    routes: pd.DataFrame
    trips: pd.DataFrame
    stops: pd.DataFrame
    stop_times: pd.DataFrame
    shapes: pd.DataFrame


@dataclass(frozen=True)
class RouteDirectionCandidate:
    """One candidate shape+stop-pattern for a (route_id, direction_id) pair.

    A route/direction can have more than one distinct ``shape_id`` (e.g. a
    campus shuttle with an optional detour stop) -- trip inference (the next
    step) scores a raw trace against every candidate and picks the best
    match, so this loader hands back one candidate per distinct shape rather
    than collapsing them.
    """

    route_id: str
    direction_id: int
    shape_id: str
    representative_trip_id: str
    polyline: List[PolylinePoint]
    stops: List[EtavalStop]  # progress_m already assigned via assign_stops_monotonic


def _read_member(zf: zipfile.ZipFile, name: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(zf.read(name)))


def load_gtfs(path: Union[str, Path]) -> BucrGtfs:
    """Load a static GTFS feed from a zip file or an already-extracted directory.

    Args:
        path: Path to a ``.zip`` GTFS feed, or a directory already containing
            the extracted ``.txt`` files.

    Returns:
        A :class:`BucrGtfs` with one DataFrame per required GTFS file.

    Raises:
        BucrGtfsError: ``path`` does not exist, is neither a file nor a
            directory, or is missing one of ``REQUIRED_FILES``.
    """
    p = Path(path)
    frames: dict[str, pd.DataFrame] = {}

    if p.is_dir():
        missing = [name for name in REQUIRED_FILES if not (p / name).exists()]
        if missing:
            raise BucrGtfsError(f"{p}: directory missing required GTFS files: {missing}")
        for name in REQUIRED_FILES:
            frames[name] = pd.read_csv(p / name)
    elif p.is_file():
        try:
            zf = zipfile.ZipFile(p)
        except zipfile.BadZipFile as exc:
            raise BucrGtfsError(f"{p}: not a valid zip file") from exc
        with zf:
            names = set(zf.namelist())
            missing = [name for name in REQUIRED_FILES if name not in names]
            if missing:
                raise BucrGtfsError(f"{p}: zip missing required GTFS files: {missing}")
            for name in REQUIRED_FILES:
                frames[name] = _read_member(zf, name)
    else:
        raise BucrGtfsError(f"{p}: not a file or directory")

    return BucrGtfs(
        routes=frames["routes.txt"],
        trips=frames["trips.txt"],
        stops=frames["stops.txt"],
        stop_times=frames["stop_times.txt"],
        shapes=frames["shapes.txt"],
    )


def route_directions(data: BucrGtfs) -> List[Tuple[str, int]]:
    """Distinct (route_id, direction_id) pairs present in ``trips.txt``, in first-seen order."""
    seen: List[Tuple[str, int]] = []
    seen_set = set()
    for row in data.trips.itertuples():
        key = (str(row.route_id), int(row.direction_id))
        if key not in seen_set:
            seen_set.add(key)
            seen.append(key)
    return seen


def shape_ids_for_route_direction(data: BucrGtfs, route_id: str, direction_id: int) -> List[str]:
    """Distinct ``shape_id`` values used by trips of (route_id, direction_id), first-seen order."""
    trips = data.trips
    mask = (trips["route_id"] == route_id) & (trips["direction_id"].astype(int) == int(direction_id))
    seen: List[str] = []
    seen_set = set()
    for shape_id in trips.loc[mask, "shape_id"]:
        if shape_id not in seen_set:
            seen_set.add(shape_id)
            seen.append(shape_id)
    return seen


def shape_points(data: BucrGtfs, shape_id: str) -> List[Tuple[float, float, Optional[float]]]:
    """Ordered ``(lat, lon, shape_dist_traveled)`` for one shape.

    ``shape_dist_traveled`` is ``None`` for every point when the column is
    absent or not fully populated for this shape -- ``build_shape_polyline``
    then falls back to cumulative haversine, matching
    ``etaval.spatial.polyline.build_polyline``'s own fallback rule.

    Raises:
        BucrGtfsError: no rows exist for ``shape_id``.
    """
    sub = data.shapes.loc[data.shapes["shape_id"] == shape_id].sort_values("shape_pt_sequence")
    if sub.empty:
        raise BucrGtfsError(f"unknown shape_id: {shape_id!r}")

    has_dist = "shape_dist_traveled" in sub.columns and sub["shape_dist_traveled"].notna().all()
    return [
        (
            float(row.shape_pt_lat),
            float(row.shape_pt_lon),
            float(row.shape_dist_traveled) if has_dist else None,
        )
        for row in sub.itertuples()
    ]


def build_shape_polyline(data: BucrGtfs, shape_id: str) -> List[PolylinePoint]:
    """Build an ``etaval`` polyline for one shape, ready for map-matching."""
    pts = shape_points(data, shape_id)
    if not pts:
        raise BucrGtfsError(f"shape_id {shape_id!r} has no points")
    dists: Optional[List[float]] = None
    if all(d is not None for _, _, d in pts):
        dists = [d for _, _, d in pts]  # type: ignore[misc]
    return build_polyline([(lat, lon) for lat, lon, _ in pts], shape_dist_traveled=dists)


def stops_for_trip(data: BucrGtfs, trip_id: str) -> List[EtavalStop]:
    """Ordered stops (by ``stop_sequence``) for one trip, ready for ``assign_stops_monotonic``.

    ``progress_m`` is left unset (``None``) here -- it's populated by
    :func:`build_shape_polyline` + ``assign_stops_monotonic``, not by this
    function, since assignment requires a polyline.

    Raises:
        BucrGtfsError: ``trip_id`` has no rows in ``stop_times.txt``, or a
            referenced ``stop_id`` is missing from ``stops.txt``.
    """
    st = data.stop_times.loc[data.stop_times["trip_id"] == trip_id].sort_values("stop_sequence")
    if st.empty:
        raise BucrGtfsError(f"unknown trip_id: {trip_id!r}")

    stops_by_id = data.stops.set_index("stop_id")
    result: List[EtavalStop] = []
    for row in st.itertuples():
        if row.stop_id not in stops_by_id.index:
            raise BucrGtfsError(f"trip {trip_id!r} references unknown stop_id {row.stop_id!r}")
        srow = stops_by_id.loc[row.stop_id]
        result.append(
            EtavalStop(
                stop_id=str(row.stop_id),
                stop_sequence=int(row.stop_sequence),
                lat=float(srow["stop_lat"]),
                lon=float(srow["stop_lon"]),
            )
        )
    return result


def route_direction_candidates(
    data: BucrGtfs, route_id: str, direction_id: int
) -> List[RouteDirectionCandidate]:
    """Candidate polylines + monotonic-assigned stops for (route_id, direction_id).

    One :class:`RouteDirectionCandidate` per distinct ``shape_id`` used by
    trips of this route/direction. Stops come from a representative trip
    (the first trip seen using that shape) -- campus-shuttle-style feeds can
    have more than one stop pattern per shape (e.g. an optional stop), which
    is a trip-inference-time concern, not this loader's.

    Raises:
        BucrGtfsError: no trips match (route_id, direction_id).
    """
    trips = data.trips
    mask = (trips["route_id"] == route_id) & (trips["direction_id"].astype(int) == int(direction_id))
    sub = trips.loc[mask]
    if sub.empty:
        raise BucrGtfsError(
            f"no trips for route_id={route_id!r} direction_id={direction_id!r}"
        )

    candidates: List[RouteDirectionCandidate] = []
    seen_shapes: set = set()
    for row in sub.itertuples():
        shape_id = row.shape_id
        if shape_id in seen_shapes:
            continue
        seen_shapes.add(shape_id)

        polyline = build_shape_polyline(data, shape_id)
        stops = stops_for_trip(data, row.trip_id)
        assigned_stops = assign_stops_monotonic(stops, polyline)

        candidates.append(
            RouteDirectionCandidate(
                route_id=str(route_id),
                direction_id=int(direction_id),
                shape_id=str(shape_id),
                representative_trip_id=str(row.trip_id),
                polyline=polyline,
                stops=assigned_stops,
            )
        )
    return candidates
