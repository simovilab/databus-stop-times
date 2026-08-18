"""Tests for etaval/spatial/polyline.py — Phase P2.

Coverage
--------
haversine_m           : known great-circle distances within 1 % tolerance
build_polyline        : monotonicity; with/without shape_dist_traveled; single-point edge case
project_point_to_polyline : perpendicular cross-track and along-track progress; endpoint snapping
assign_stops_monotonic : straight shape → monotonic; looped/out-and-back shape → still monotonic;
                         stop_sequence ordering; degenerate (single stop); pre-sorted stops
upcoming_stops        : exact filter at boundary; none passed; all passed; None-progress exclusion
"""

from __future__ import annotations

import math

import pytest

from etaval.domain.models import Stop
from etaval.spatial.polyline import (
    PolylinePoint,
    Projection,
    assign_stops_monotonic,
    build_polyline,
    haversine_m,
    project_point_to_polyline,
    upcoming_stops,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stop(sid: str, seq: int, lat: float, lon: float, pm: float | None = None) -> Stop:
    return Stop(stop_id=sid, stop_sequence=seq, lat=lat, lon=lon, progress_m=pm)


# ---------------------------------------------------------------------------
# haversine_m
# ---------------------------------------------------------------------------


class TestHaversineM:
    def test_one_degree_latitude_approx_111km(self) -> None:
        """1 degree of latitude ≈ 111.2 km (spherical Earth, R=6 371 000 m)."""
        d = haversine_m(0.0, 0.0, 1.0, 0.0)
        # Exact value for a sphere with R=6 371 000: R * pi/180 ≈ 111 194.9 m
        assert abs(d - 111_195.0) < 1_000.0, f"got {d:.1f} m"

    def test_one_degree_longitude_at_equator(self) -> None:
        """1 degree of longitude at the equator equals 1 degree of latitude."""
        lat_dist = haversine_m(0.0, 0.0, 1.0, 0.0)
        lon_dist = haversine_m(0.0, 0.0, 0.0, 1.0)
        assert abs(lat_dist - lon_dist) < 1.0, "should be equal at equator"

    def test_zero_distance(self) -> None:
        assert haversine_m(48.0, -71.0, 48.0, -71.0) == pytest.approx(0.0)

    def test_antipodal_points(self) -> None:
        """Antipodal points should be ~half Earth circumference."""
        d = haversine_m(0.0, 0.0, 0.0, 180.0)
        earth_half_circumference = math.pi * 6_371_000.0
        assert abs(d - earth_half_circumference) < 1_000.0

    def test_known_coordinate_pair(self) -> None:
        """Boston → New York (LAX) rough sanity check > 200 km."""
        d = haversine_m(42.36, -71.06, 40.71, -74.01)
        assert 290_000 < d < 330_000, f"got {d:.0f} m"

    def test_symmetry(self) -> None:
        assert haversine_m(10.0, 20.0, 30.0, 40.0) == pytest.approx(
            haversine_m(30.0, 40.0, 10.0, 20.0)
        )


# ---------------------------------------------------------------------------
# build_polyline
# ---------------------------------------------------------------------------


class TestBuildPolyline:
    def test_empty_input_returns_empty(self) -> None:
        assert build_polyline([]) == []

    def test_single_point(self) -> None:
        pts = build_polyline([(1.0, 2.0)])
        assert len(pts) == 1
        assert pts[0].cum_m == pytest.approx(0.0)

    def test_straight_line_monotonic(self) -> None:
        """Three collinear points → cum_m strictly increasing."""
        pts = build_polyline([(0.0, 0.0), (0.0, 1.0), (0.0, 2.0)])
        assert len(pts) == 3
        assert pts[0].cum_m == pytest.approx(0.0)
        assert pts[1].cum_m > pts[0].cum_m
        assert pts[2].cum_m > pts[1].cum_m

    def test_two_segment_total_distance(self) -> None:
        """cum_m[-1] should equal haversine from first to last via midpoint."""
        pts = build_polyline([(0.0, 0.0), (0.0, 1.0), (0.0, 2.0)])
        expected = haversine_m(0.0, 0.0, 0.0, 1.0) + haversine_m(0.0, 1.0, 0.0, 2.0)
        assert pts[2].cum_m == pytest.approx(expected, rel=1e-9)

    def test_with_shape_dist_traveled_honored(self) -> None:
        """When shape_dist_traveled is provided the mid-point ratio is preserved."""
        shape_pts = [(0.0, 0.0), (0.0, 1.0), (0.0, 2.0)]
        # Raw distances: 0, 100, 200 — midpoint is exactly halfway.
        pts = build_polyline(shape_pts, shape_dist_traveled=[0.0, 100.0, 200.0])
        assert pts[0].cum_m == pytest.approx(0.0)
        # Mid-point should be exactly half of the total
        assert pts[1].cum_m == pytest.approx(pts[2].cum_m / 2.0, rel=1e-9)
        # Total should equal haversine total
        expected_total = haversine_m(0.0, 0.0, 0.0, 1.0) + haversine_m(0.0, 1.0, 0.0, 2.0)
        assert pts[2].cum_m == pytest.approx(expected_total, rel=1e-6)

    def test_shape_dist_traveled_unequal_spacing(self) -> None:
        """Unequal raw dist values produce correct proportional cum_m."""
        shape_pts = [(0.0, 0.0), (0.0, 1.0), (0.0, 2.0)]
        # Raw: first gap = 25, second gap = 75 → first section is 25% of total
        pts = build_polyline(shape_pts, shape_dist_traveled=[0.0, 25.0, 100.0])
        total = pts[2].cum_m
        assert pts[1].cum_m == pytest.approx(total * 0.25, rel=1e-6)

    def test_embedded_third_element_used_as_dist(self) -> None:
        """3-element tuples without explicit shape_dist_traveled → third element used."""
        pts_3 = build_polyline([(0.0, 0.0, 0.0), (0.0, 1.0, 100.0), (0.0, 2.0, 200.0)])
        pts_2 = build_polyline(
            [(0.0, 0.0), (0.0, 1.0), (0.0, 2.0)],
            shape_dist_traveled=[0.0, 100.0, 200.0],
        )
        for a, b in zip(pts_3, pts_2, strict=True):
            assert a.cum_m == pytest.approx(b.cum_m, rel=1e-9)

    def test_lat_lon_preserved(self) -> None:
        pts = build_polyline([(10.0, 20.0), (11.0, 21.0)])
        assert pts[0] == PolylinePoint(lat=10.0, lon=20.0, cum_m=0.0)
        assert pts[1].lat == pytest.approx(11.0)
        assert pts[1].lon == pytest.approx(21.0)

    def test_embedded_third_element_partially_none_falls_back_to_haversine(self) -> None:
        """Real shapes.txt sometimes leaves shape_dist_traveled blank for a subset of
        points within an otherwise-populated shape. The first point having a distance
        must not be trusted as a promise that every point does -- this used to raise
        `TypeError: float() argument must be a string or a real number, not 'NoneType'`.
        """
        shape_pts = [(0.0, 0.0, 0.0), (0.0, 1.0, None), (0.0, 2.0, 200.0)]
        pts = build_polyline(shape_pts)
        expected = haversine_m(0.0, 0.0, 0.0, 1.0) + haversine_m(0.0, 1.0, 0.0, 2.0)
        assert pts[2].cum_m == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# project_point_to_polyline
# ---------------------------------------------------------------------------


class TestProjectPointToPolyline:
    @pytest.fixture()
    def east_west_line(self) -> list[PolylinePoint]:
        """Single segment along the equator from lon=0 to lon=1."""
        return build_polyline([(0.0, 0.0), (0.0, 1.0)])

    def test_midpoint_on_segment_zero_cross_track(
        self, east_west_line: list[PolylinePoint]
    ) -> None:
        """A point exactly on the line → cross_track_m ≈ 0."""
        proj = project_point_to_polyline(0.0, 0.5, east_west_line)
        assert proj.cross_track_m == pytest.approx(0.0, abs=1.0)

    def test_midpoint_progress(self, east_west_line: list[PolylinePoint]) -> None:
        """Point at midpoint → progress_m ≈ half of total polyline length."""
        total = east_west_line[-1].cum_m
        proj = project_point_to_polyline(0.0, 0.5, east_west_line)
        assert proj.progress_m == pytest.approx(total / 2.0, rel=0.01)

    def test_perpendicular_offset_cross_track(
        self, east_west_line: list[PolylinePoint]
    ) -> None:
        """Point offset 0.001° north → cross_track_m ≈ 0.001 * ~111 194 m ≈ 111 m."""
        proj = project_point_to_polyline(0.001, 0.5, east_west_line)
        expected_ct = haversine_m(0.0, 0.5, 0.001, 0.5)
        # Flat-earth approximation vs haversine should agree within 1 %
        assert abs(proj.cross_track_m - expected_ct) < expected_ct * 0.01

    def test_perpendicular_offset_progress(
        self, east_west_line: list[PolylinePoint]
    ) -> None:
        """Progress for a northward-offset midpoint should still be ~half the line."""
        total = east_west_line[-1].cum_m
        proj = project_point_to_polyline(0.001, 0.5, east_west_line)
        assert proj.progress_m == pytest.approx(total / 2.0, rel=0.02)

    def test_point_before_start_snaps_to_segment_start(
        self, east_west_line: list[PolylinePoint]
    ) -> None:
        """Point west of the line snaps to progress_m ≈ 0."""
        proj = project_point_to_polyline(0.0, -1.0, east_west_line)
        assert proj.progress_m == pytest.approx(0.0, abs=1.0)
        assert proj.segment_idx == 0

    def test_point_after_end_snaps_to_segment_end(
        self, east_west_line: list[PolylinePoint]
    ) -> None:
        """Point east of the line snaps to the polyline endpoint."""
        total = east_west_line[-1].cum_m
        proj = project_point_to_polyline(0.0, 2.0, east_west_line)
        assert proj.progress_m == pytest.approx(total, abs=1.0)

    def test_raises_on_single_point_polyline(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            project_point_to_polyline(0.0, 0.0, [PolylinePoint(0.0, 0.0, 0.0)])

    def test_multi_segment_correct_segment_chosen(self) -> None:
        """Three-segment polyline → point near second segment returns segment_idx=1."""
        poly = build_polyline([(0.0, 0.0), (0.0, 1.0), (0.0, 2.0), (0.0, 3.0)])
        # Point slightly above lon=1.5 → second segment (idx 1)
        proj = project_point_to_polyline(0.001, 1.5, poly)
        assert proj.segment_idx == 1


# ---------------------------------------------------------------------------
# assign_stops_monotonic
# ---------------------------------------------------------------------------


class TestAssignStopsMonotonic:
    # ---- straight shape ----

    @pytest.fixture()
    def straight_polyline(self) -> list[PolylinePoint]:
        """Four-point east-west line, each segment 0.1°."""
        return build_polyline([(0.0, 0.0), (0.0, 0.1), (0.0, 0.2), (0.0, 0.3)])

    def test_straight_progress_monotonic(
        self, straight_polyline: list[PolylinePoint]
    ) -> None:
        stops = [
            _stop("A", 1, 0.0, 0.05),
            _stop("B", 2, 0.0, 0.15),
            _stop("C", 3, 0.0, 0.25),
        ]
        result = assign_stops_monotonic(stops, straight_polyline)
        progs = [s.progress_m for s in result]
        # Non-decreasing
        assert all(progs[i] <= progs[i + 1] for i in range(len(progs) - 1))

    def test_straight_progress_values_sensible(
        self, straight_polyline: list[PolylinePoint]
    ) -> None:
        """Stops at known positions → progress_m within 5 % of haversine."""
        stops = [
            _stop("A", 1, 0.0, 0.05),
            _stop("B", 2, 0.0, 0.15),
            _stop("C", 3, 0.0, 0.25),
        ]
        result = assign_stops_monotonic(stops, straight_polyline)
        # Expected: midpoint of each 0.1° segment
        step = haversine_m(0.0, 0.0, 0.0, 0.1)
        expected = [0.5 * step, 1.5 * step, 2.5 * step]
        for s, exp in zip(result, expected, strict=True):
            assert abs(s.progress_m - exp) < exp * 0.05, (
                f"stop {s.stop_id}: got {s.progress_m:.1f} expected ≈{exp:.1f}"
            )

    def test_input_order_does_not_matter(
        self, straight_polyline: list[PolylinePoint]
    ) -> None:
        """Stops supplied in reverse sequence order → same result as forward."""
        stops_fwd = [
            _stop("A", 1, 0.0, 0.05),
            _stop("B", 2, 0.0, 0.15),
            _stop("C", 3, 0.0, 0.25),
        ]
        stops_rev = list(reversed(stops_fwd))
        result_fwd = assign_stops_monotonic(stops_fwd, straight_polyline)
        result_rev = assign_stops_monotonic(stops_rev, straight_polyline)
        for a, b in zip(result_fwd, result_rev, strict=True):
            assert a.progress_m == pytest.approx(b.progress_m)

    def test_inputs_not_mutated(
        self, straight_polyline: list[PolylinePoint]
    ) -> None:
        """Original Stop objects must remain unchanged (frozen + model_copy)."""
        original = _stop("A", 1, 0.0, 0.05)  # progress_m = None
        result = assign_stops_monotonic([original], straight_polyline)
        assert original.progress_m is None
        assert result[0].progress_m is not None

    def test_new_stop_objects_returned(
        self, straight_polyline: list[PolylinePoint]
    ) -> None:
        stop = _stop("X", 1, 0.0, 0.05)
        result = assign_stops_monotonic([stop], straight_polyline)
        assert result[0] is not stop

    def test_single_stop(
        self, straight_polyline: list[PolylinePoint]
    ) -> None:
        result = assign_stops_monotonic([_stop("A", 1, 0.0, 0.15)], straight_polyline)
        assert len(result) == 1
        assert result[0].progress_m is not None

    def test_empty_stops(
        self, straight_polyline: list[PolylinePoint]
    ) -> None:
        assert assign_stops_monotonic([], straight_polyline) == []

    # ---- looped / out-and-back shape ----

    @pytest.fixture()
    def outback_polyline(self) -> list[PolylinePoint]:
        """Out-and-back shape: east 0→0.3°, then back 0.3→0°.

        Segment indices:
          0: (0,0.0)→(0,0.1)
          1: (0,0.1)→(0,0.2)
          2: (0,0.2)→(0,0.3)   ← turnaround
          3: (0,0.3)→(0,0.2)
          4: (0,0.2)→(0,0.1)
          5: (0,0.1)→(0,0.0)
        """
        return build_polyline([
            (0.0, 0.0),
            (0.0, 0.1),
            (0.0, 0.2),
            (0.0, 0.3),
            (0.0, 0.2),
            (0.0, 0.1),
            (0.0, 0.0),
        ])

    def test_looped_shape_progress_monotonic(
        self, outback_polyline: list[PolylinePoint]
    ) -> None:
        """Late stop near a geographically early location must NOT snap backward."""
        stops = [
            _stop("S1", 1, 0.0, 0.05),   # outbound, near start
            _stop("S2", 2, 0.0, 0.25),   # outbound, near turnaround
            _stop("S3", 3, 0.0, 0.05),   # return leg, same geo as S1 but later in sequence
        ]
        result = assign_stops_monotonic(stops, outback_polyline)
        progs = [s.progress_m for s in result]
        assert all(progs[i] <= progs[i + 1] for i in range(len(progs) - 1)), (
            f"expected non-decreasing progress, got {progs}"
        )

    def test_looped_shape_late_stop_on_return_leg(
        self, outback_polyline: list[PolylinePoint]
    ) -> None:
        """S3 (return leg) must have higher progress than S2 (outbound)."""
        stops = [
            _stop("S1", 1, 0.0, 0.05),
            _stop("S2", 2, 0.0, 0.25),
            _stop("S3", 3, 0.0, 0.05),
        ]
        result = assign_stops_monotonic(stops, outback_polyline)
        s2_prog = result[1].progress_m
        s3_prog = result[2].progress_m
        assert s3_prog is not None and s2_prog is not None
        assert s3_prog > s2_prog, (
            f"S3 should be on the return leg (prog > S2={s2_prog:.1f}), got {s3_prog:.1f}"
        )

    def test_looped_shape_s3_on_return_segment(
        self, outback_polyline: list[PolylinePoint]
    ) -> None:
        """S3 progress_m should be well into the second half of the shape."""
        total = outback_polyline[-1].cum_m
        stops = [
            _stop("S1", 1, 0.0, 0.05),
            _stop("S2", 2, 0.0, 0.25),
            _stop("S3", 3, 0.0, 0.05),
        ]
        result = assign_stops_monotonic(stops, outback_polyline)
        s3_prog = result[2].progress_m
        assert s3_prog is not None
        # S3 must be in the second half (return leg) of the shape
        assert s3_prog > total / 2.0, (
            f"S3 should be on the return leg (>{total/2:.0f} m), got {s3_prog:.1f} m"
        )


# ---------------------------------------------------------------------------
# upcoming_stops
# ---------------------------------------------------------------------------


class TestUpcomingStops:
    @pytest.fixture()
    def assigned_stops(self) -> list[Stop]:
        return [
            _stop("A", 1, 0.0, 0.0, pm=0.0),
            _stop("B", 2, 0.0, 0.0, pm=100.0),
            _stop("C", 3, 0.0, 0.0, pm=200.0),
            _stop("D", 4, 0.0, 0.0, pm=300.0),
        ]

    def test_none_passed(self, assigned_stops: list[Stop]) -> None:
        result = upcoming_stops(0.0, assigned_stops)
        assert [s.stop_id for s in result] == ["A", "B", "C", "D"]

    def test_some_passed(self, assigned_stops: list[Stop]) -> None:
        result = upcoming_stops(150.0, assigned_stops)
        assert [s.stop_id for s in result] == ["C", "D"]

    def test_at_stop_boundary_inclusive(self, assigned_stops: list[Stop]) -> None:
        """Vehicle at exactly stop B's progress → B is still upcoming."""
        result = upcoming_stops(100.0, assigned_stops)
        assert [s.stop_id for s in result] == ["B", "C", "D"]

    def test_all_passed(self, assigned_stops: list[Stop]) -> None:
        result = upcoming_stops(300.1, assigned_stops)
        assert result == []

    def test_empty_stops_list(self) -> None:
        assert upcoming_stops(50.0, []) == []

    def test_none_progress_stops_excluded(self) -> None:
        stops = [
            _stop("X", 1, 0.0, 0.0, pm=None),
            _stop("Y", 2, 0.0, 0.0, pm=200.0),
        ]
        result = upcoming_stops(100.0, stops)
        assert [s.stop_id for s in result] == ["Y"]

    def test_all_none_progress(self) -> None:
        stops = [_stop("X", 1, 0.0, 0.0, pm=None)]
        assert upcoming_stops(0.0, stops) == []

    def test_preserves_order(self) -> None:
        """Result preserves the input list order, not sorted by progress_m."""
        stops = [
            _stop("Z", 3, 0.0, 0.0, pm=300.0),
            _stop("Y", 2, 0.0, 0.0, pm=200.0),
            _stop("X", 1, 0.0, 0.0, pm=100.0),
        ]
        result = upcoming_stops(50.0, stops)
        assert [s.stop_id for s in result] == ["Z", "Y", "X"]


# ---------------------------------------------------------------------------
# Type / return-type smoke tests
# ---------------------------------------------------------------------------


class TestReturnTypes:
    def test_polyline_point_is_named_tuple(self) -> None:
        pt = PolylinePoint(lat=1.0, lon=2.0, cum_m=3.0)
        assert isinstance(pt, tuple)
        assert pt.lat == 1.0

    def test_projection_is_named_tuple(self) -> None:
        pr = Projection(progress_m=10.0, cross_track_m=5.0, segment_idx=0)
        assert isinstance(pr, tuple)
        assert pr.segment_idx == 0

    def test_build_polyline_returns_polyline_points(self) -> None:
        pts = build_polyline([(0.0, 0.0), (1.0, 1.0)])
        assert all(isinstance(p, PolylinePoint) for p in pts)

    def test_assign_stops_monotonic_returns_stop_list(self) -> None:
        poly = build_polyline([(0.0, 0.0), (0.0, 1.0)])
        stops = [_stop("A", 1, 0.0, 0.5)]
        result = assign_stops_monotonic(stops, poly)
        assert all(isinstance(s, Stop) for s in result)
