"""Real map-matching in `default_match` (Phase 3, U6).

Before this, `default_match` was a placeholder: pass-through for
`trip_update` observations, a recorded error for every `vehicle_position`
observation. These tests pin the real behavior: a `vehicle_position` is
projected onto its trip's shape (`core.spatial.polyline`) to discover the
stops still ahead and their along-shape distances; a `trip_update` gets the
same distance/`stops_ahead` features attached when static geometry is
available. No network, no CLI -- a real `GtfsStaticLoader` over a small
synthetic GTFS zip built in `tmp_path`, exactly like other core/serve tests.
"""

from __future__ import annotations

import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from stoptimes.core.contracts.data import Observation, ObservationKind
from stoptimes.core.contracts.pipeline import PipelineContext
from stoptimes.core.contracts.source import Source
from stoptimes.core.gtfs.static import GtfsStaticLoader
from stoptimes.serve.pipeline import default_match


def _static_zip(path: Path) -> Path:
    """A one-route, four-stop straight line, one shape, one trip."""
    stops = ["stop_id,stop_name,stop_lat,stop_lon"]
    for i in range(4):
        stops.append(f"s{i + 1},Stop {i + 1},0.0,{i * 0.01:.5f}")

    trips = ["route_id,service_id,trip_id,direction_id,shape_id", "r1,svc,t1,0,sh1"]
    stop_times = ["trip_id,arrival_time,departure_time,stop_id,stop_sequence,shape_dist_traveled"]
    for i in range(4):
        stop_times.append(f"t1,08:{i * 2:02d}:00,08:{i * 2:02d}:00,s{i + 1},{i + 1},{i * 1113}")

    shapes = ["shape_id,shape_pt_lat,shape_pt_lon,shape_pt_sequence,shape_dist_traveled"]
    for i in range(4):
        shapes.append(f"sh1,0.0,{i * 0.01:.5f},{i + 1},{i * 1113}")

    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("stops.txt", "\n".join(stops) + "\n")
        zf.writestr("trips.txt", "\n".join(trips) + "\n")
        zf.writestr("stop_times.txt", "\n".join(stop_times) + "\n")
        zf.writestr("shapes.txt", "\n".join(shapes) + "\n")
        zf.writestr("routes.txt", "route_id,route_short_name,route_long_name\nr1,1,Line One\n")
    return path


def _static_zip_with_tail(path: Path) -> Path:
    """Same line as `_static_zip`, but the shape runs past the last stop so
    a vehicle can be positioned genuinely beyond every scheduled stop."""
    stops = ["stop_id,stop_name,stop_lat,stop_lon"]
    for i in range(4):
        stops.append(f"s{i + 1},Stop {i + 1},0.0,{i * 0.01:.5f}")

    trips = ["route_id,service_id,trip_id,direction_id,shape_id", "r1,svc,t1,0,sh1"]
    stop_times = ["trip_id,arrival_time,departure_time,stop_id,stop_sequence,shape_dist_traveled"]
    for i in range(4):
        stop_times.append(f"t1,08:{i * 2:02d}:00,08:{i * 2:02d}:00,s{i + 1},{i + 1},{i * 1113}")

    shapes = ["shape_id,shape_pt_lat,shape_pt_lon,shape_pt_sequence,shape_dist_traveled"]
    for i in range(6):  # two extra points past stop 4 (index 3)
        shapes.append(f"sh1,0.0,{i * 0.01:.5f},{i + 1},{i * 1113}")

    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("stops.txt", "\n".join(stops) + "\n")
        zf.writestr("trips.txt", "\n".join(trips) + "\n")
        zf.writestr("stop_times.txt", "\n".join(stop_times) + "\n")
        zf.writestr("shapes.txt", "\n".join(shapes) + "\n")
        zf.writestr("routes.txt", "route_id,route_short_name,route_long_name\nr1,1,Line One\n")
    return path


def _source(gtfs_static_url: str = "/dev/null") -> Source:
    return Source(id="testagency", name="Test Agency", gtfs_static_url=gtfs_static_url)


def _ctx(source: Source, *, static_feed=None) -> PipelineContext:
    return PipelineContext(
        source=source,
        static_feed=static_feed,
        model=None,
        now=datetime(2026, 9, 17, 12, 0, 0, tzinfo=timezone.utc),
        config={},
    )


def _vp(lat: float, lon: float, trip_id: str = "t1") -> Observation:
    return Observation(
        kind=ObservationKind.VEHICLE_POSITION,
        source_id="testagency",
        trip_id=trip_id,
        route_id="r1",
        start_date="20260917",
        vehicle_id="v1",
        ts=None,
        lat=lat,
        lon=lon,
    )


def _tu(stop_id: str, stop_sequence: int, trip_id: str = "t1") -> Observation:
    return Observation(
        kind=ObservationKind.TRIP_UPDATE,
        source_id="testagency",
        trip_id=trip_id,
        route_id="r1",
        start_date="20260917",
        stop_id=stop_id,
        stop_sequence=stop_sequence,
    )


def test_vehicle_position_is_matched_to_upcoming_stops_with_real_distances(tmp_path):
    loader = GtfsStaticLoader(_static_zip(tmp_path / "gtfs.zip"))
    ctx = _ctx(_source(), static_feed=loader)

    # Sitting exactly at stop 1 (lon 0.0): every stop is still "ahead"
    # (>= its own progress, and stop 1's progress is 0).
    matched = default_match([_vp(0.0, 0.0)], ctx)

    assert ctx.state.get("errors", []) == []
    assert [m.stop_id for m in matched] == ["s1", "s2", "s3", "s4"]
    assert [m.stop_sequence for m in matched] == [1, 2, 3, 4]
    assert [m.extra["stops_ahead"] for m in matched] == [0.0, 1.0, 2.0, 3.0]

    distances = [m.extra["distance_to_stop"] for m in matched]
    # Monotonically increasing distance the further ahead the stop is.
    assert distances == sorted(distances)
    assert distances[0] == pytest.approx(0.0, abs=1.0)
    assert distances[-1] > distances[0]


def test_vehicle_position_past_the_last_stop_records_an_error(tmp_path):
    loader = GtfsStaticLoader(_static_zip_with_tail(tmp_path / "gtfs.zip"))
    ctx = _ctx(_source(), static_feed=loader)

    # Past stop 4 (lon 0.03), but the shape itself continues to lon 0.05.
    matched = default_match([_vp(0.0, 0.05)], ctx)

    assert matched == []
    errors = ctx.state["errors"]
    assert any(e["stage"] == "match" and e["trip_id"] == "t1" for e in errors)


def test_vehicle_position_without_any_static_geometry_still_records_an_error():
    """No `ctx.static_feed` and a `gtfs_static_url` that isn't a real feed:
    falls back to the historical placeholder behavior instead of crashing."""
    ctx = _ctx(_source(gtfs_static_url="/dev/null"))

    matched = default_match([_vp(0.0, 0.0, trip_id="t3")], ctx)

    assert matched == []
    errors = ctx.state["errors"]
    assert any(e["stage"] == "match" and e["trip_id"] == "t3" for e in errors)


def test_trip_update_gets_real_distance_and_stops_ahead_when_geometry_available(tmp_path):
    loader = GtfsStaticLoader(_static_zip(tmp_path / "gtfs.zip"))
    ctx = _ctx(_source(), static_feed=loader)

    matched = default_match([_tu("s3", 3)], ctx)

    assert ctx.state.get("errors", []) == []
    assert len(matched) == 1
    item = matched[0]
    assert item.stop_id == "s3"
    # stop_sequence 3 has two stops after it (4 total): stops_ahead = 1.
    assert item.extra["stops_ahead"] == 1.0
    # distance_to_stop is the from-stop(s2)-to-this-stop(s3) segment length.
    assert item.extra["distance_to_stop"] == pytest.approx(1113.0, rel=0.05)


def test_trip_update_without_geometry_still_passes_through_unchanged():
    """Historical placeholder behavior: no geometry -> pass-through, no
    spatial extras, never an error (the stop is already identified)."""
    ctx = _ctx(_source(gtfs_static_url="/dev/null"))

    matched = default_match([_tu("s1", 1, trip_id="t1")], ctx)

    assert ctx.state.get("errors", []) == []
    assert len(matched) == 1
    assert matched[0].stop_id == "s1"
    assert matched[0].extra == {}
