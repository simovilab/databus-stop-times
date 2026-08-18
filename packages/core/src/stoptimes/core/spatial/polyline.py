"""Spatial map-matching, implemented fresh (no external geometry dep). Phase P2.

Public API
----------
haversine_m(lat1, lon1, lat2, lon2) -> float
build_polyline(shape_pts, shape_dist_traveled=None) -> list[PolylinePoint]
project_point_to_polyline(lat, lon, polyline) -> Projection
assign_stops_monotonic(stops, polyline) -> list[Stop]
upcoming_stops(progress_m, stops) -> list[Stop]

Data types
----------
PolylinePoint(lat, lon, cum_m)           -- NamedTuple, metres from first point
Projection(progress_m, cross_track_m, segment_idx)  -- NamedTuple
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

from etaval.domain.models import Stop

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EARTH_R_M: float = 6_371_000.0
# Metres per degree of latitude (also longitude at the equator on a sphere).
# Used only for the flat-earth equirectangular segment projection; all
# cumulative polyline distances (cum_m) are computed via haversine.
_METERS_PER_DEG: float = _EARTH_R_M * math.pi / 180.0  # ≈ 111_194.9


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class PolylinePoint(NamedTuple):
    """One point on a cumulative polyline."""

    lat: float
    lon: float
    cum_m: float  # cumulative metres from the first shape point


class Projection(NamedTuple):
    """Result of projecting a geographic point onto a polyline."""

    progress_m: float  # distance along the polyline to the nearest point on it
    cross_track_m: float  # perpendicular distance from the query point to the line
    segment_idx: int  # 0-based index of the nearest segment


# ---------------------------------------------------------------------------
# Haversine
# ---------------------------------------------------------------------------


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two WGS-84 points."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return _EARTH_R_M * 2.0 * math.asin(math.sqrt(min(a, 1.0)))


# ---------------------------------------------------------------------------
# Polyline construction
# ---------------------------------------------------------------------------


def build_polyline(
    shape_pts: Sequence[tuple[float, ...]],
    shape_dist_traveled: list[float] | None = None,
) -> list[PolylinePoint]:
    """Build a cumulative polyline from ordered shape points.

    Args:
        shape_pts: Ordered ``[(lat, lon), …]`` or ``[(lat, lon, raw_dist), …]``.
            Three-element tuples are treated as having an embedded raw distance
            in the third position, which is used when *shape_dist_traveled* is
            ``None``.
        shape_dist_traveled: Per-point distances from GTFS ``shapes.txt``.
            When provided it takes precedence over any embedded third element.
            The values are in the agency's native unit; they are scaled
            proportionally to the total haversine distance so that all ``cum_m``
            values are in metres.

    Returns:
        List of :class:`PolylinePoint` with non-decreasing *cum_m*.
    """
    if not shape_pts:
        return []

    lats: list[float] = [float(p[0]) for p in shape_pts]
    lons: list[float] = [float(p[1]) for p in shape_pts]
    n = len(lats)

    # Resolve raw distances: explicit argument wins, then embedded third element.
    # Real-world shapes.txt sometimes leaves shape_dist_traveled blank for a subset
    # of points within an otherwise-populated shape, so every point must actually
    # have a usable value -- checking only shape_pts[0] and assuming the rest match
    # crashes on `float(None)` when a later point is missing it. Falls through to
    # cumulative haversine below when any point in the shape lacks it.
    raw_dists: list[float] | None = shape_dist_traveled
    if (
        raw_dists is None
        and n > 0
        and all(len(p) >= 3 and p[2] is not None for p in shape_pts)
    ):
        raw_dists = [float(p[2]) for p in shape_pts]

    if raw_dists is not None:
        # Scale raw unit → metres proportionally to cumulative haversine.
        total_hav = sum(
            haversine_m(lats[i], lons[i], lats[i + 1], lons[i + 1])
            for i in range(n - 1)
        )
        raw_span = raw_dists[-1] - raw_dists[0]
        scale = (total_hav / raw_span) if raw_span > 0.0 else 1.0
        origin = raw_dists[0]
        return [
            PolylinePoint(lat=lats[i], lon=lons[i], cum_m=(raw_dists[i] - origin) * scale)
            for i in range(n)
        ]

    # No external distances: cumulative haversine.
    cum: float = 0.0
    result: list[PolylinePoint] = [PolylinePoint(lat=lats[0], lon=lons[0], cum_m=0.0)]
    for i in range(1, n):
        cum += haversine_m(lats[i - 1], lons[i - 1], lats[i], lons[i])
        result.append(PolylinePoint(lat=lats[i], lon=lons[i], cum_m=cum))
    return result


# ---------------------------------------------------------------------------
# Segment projection helpers
# ---------------------------------------------------------------------------


def _seg_project_t_distsq(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> tuple[float, float]:
    """Project local-coord point P onto segment A→B.

    Returns ``(t, dist_sq)`` where *t* is clamped to [0, 1] and *dist_sq* is
    the squared distance (m²) from P to the nearest point on the segment.
    Uses a flat-earth equirectangular approximation — accurate at segment scale.
    """
    dx = bx - ax
    dy = by - ay
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq == 0.0:
        return 0.0, (px - ax) ** 2 + (py - ay) ** 2
    t = ((px - ax) * dx + (py - ay) * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))
    cx = ax + t * dx
    cy = ay + t * dy
    return t, (px - cx) ** 2 + (py - cy) ** 2


def _project_onto_all_segments(
    lat: float, lon: float, polyline: list[PolylinePoint]
) -> list[Projection]:
    """Return one :class:`Projection` per segment (not just the nearest one).

    All coordinates are converted to a local equirectangular frame centred on
    the query point's latitude before the planar projection is applied.
    """
    cos_lat = math.cos(math.radians(lat))
    px = lon * cos_lat * _METERS_PER_DEG
    py = lat * _METERS_PER_DEG

    result: list[Projection] = []
    for i in range(len(polyline) - 1):
        p1, p2 = polyline[i], polyline[i + 1]
        ax = p1.lon * cos_lat * _METERS_PER_DEG
        ay = p1.lat * _METERS_PER_DEG
        bx = p2.lon * cos_lat * _METERS_PER_DEG
        by = p2.lat * _METERS_PER_DEG

        t, dist_sq = _seg_project_t_distsq(px, py, ax, ay, bx, by)
        seg_len_m = p2.cum_m - p1.cum_m
        progress_m = p1.cum_m + t * seg_len_m
        result.append(
            Projection(
                progress_m=progress_m,
                cross_track_m=math.sqrt(dist_sq),
                segment_idx=i,
            )
        )
    return result


# ---------------------------------------------------------------------------
# Public projection function
# ---------------------------------------------------------------------------


def project_point_to_polyline(
    lat: float, lon: float, polyline: list[PolylinePoint]
) -> Projection:
    """Project ``(lat, lon)`` onto *polyline*, returning the globally nearest projection.

    Searches all segments and returns the one with the smallest cross-track
    distance.  Uses a flat-earth equirectangular approximation per segment.

    Args:
        lat: Query latitude in decimal degrees.
        lon: Query longitude in decimal degrees.
        polyline: Built by :func:`build_polyline`.

    Returns:
        :class:`Projection` with ``progress_m``, ``cross_track_m``, and
        ``segment_idx``.

    Raises:
        ValueError: If *polyline* has fewer than 2 points.
    """
    if len(polyline) < 2:
        raise ValueError("polyline must contain at least 2 points")
    projs = _project_onto_all_segments(lat, lon, polyline)
    return min(projs, key=lambda p: p.cross_track_m)


# ---------------------------------------------------------------------------
# Monotonic stop assignment
# ---------------------------------------------------------------------------


def assign_stops_monotonic(
    stops: list[Stop], polyline: list[PolylinePoint]
) -> list[Stop]:
    """Assign ``progress_m`` to each stop using a forward-monotonic DP.

    Stops are processed in ascending *stop_sequence* order.  For each stop all
    segment projections are enumerated; among those whose ``progress_m`` is
    greater than or equal to the previous stop's assigned progress, the one
    with the smallest ``cross_track_m`` is selected.

    This Viterbi-style forward pass prevents a geographically early stop
    (e.g., on the return leg of a loop or out-and-back shape) from snapping
    back to an early segment — it must land on or after the current monotonic
    frontier.  If no segment projection satisfies the monotonic constraint the
    frontier value is reused (ensuring non-decreasing progress even in
    pathological cases).

    Returns new :class:`Stop` instances (via ``model_copy``).  Inputs are
    never mutated.

    Raises:
        ValueError: If *polyline* has fewer than 2 points.
    """
    if not stops:
        return []
    if len(polyline) < 2:
        raise ValueError("polyline must contain at least 2 points")

    ordered = sorted(stops, key=lambda s: s.stop_sequence)
    min_progress: float = 0.0
    result: list[Stop] = []

    for stop in ordered:
        projs = _project_onto_all_segments(stop.lat, stop.lon, polyline)
        valid = [p for p in projs if p.progress_m >= min_progress]

        if valid:
            best = min(valid, key=lambda p: p.cross_track_m)
            assigned = best.progress_m
        else:
            # All projections fall behind the monotonic frontier.
            assigned = min_progress

        min_progress = assigned
        result.append(stop.model_copy(update={"progress_m": assigned}))

    return result


# ---------------------------------------------------------------------------
# Upcoming-stop slice
# ---------------------------------------------------------------------------


def upcoming_stops(progress_m: float, stops: list[Stop]) -> list[Stop]:
    """Return stops not yet passed by the vehicle at *progress_m*.

    A stop is considered upcoming when its ``progress_m >= progress_m``
    (i.e., it has not yet been reached).  Stops without an assigned
    ``progress_m`` (``None``) are excluded.  Input order is preserved.
    """
    return [s for s in stops if s.progress_m is not None and s.progress_m >= progress_m]
