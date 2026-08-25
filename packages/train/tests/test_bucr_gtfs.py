"""Tests for the BUCR static GTFS loader (feature_engineering.bucr_gtfs).

Two kinds of coverage:
  * A synthetic in-memory feed (tiny, hand-built) so the suite never depends
    on network or a locally-downloaded feed.
  * Tests against the REAL BUCR feed pulled from the S3 snapshot
    (``s3://transit/feeds/bucr/gtfs_static/2026-08-24.zip``) -- skipped if
    that local copy isn't present, so CI without the snapshot still passes.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import pytest

from feature_engineering.bucr_gtfs import (
    BucrGtfsError,
    load_gtfs,
    route_direction_candidates,
    route_directions,
    shape_ids_for_route_direction,
    stops_for_trip,
)

# Path the sourcing step downloaded the real feed to. Not committed -- lives
# in the session scratchpad. Tests against it are skipped when absent.
_REAL_FEED_ZIP = Path(
    "/private/tmp/claude-501/-Users-dotj-Desktop-SIMOVI-git-no-sync-gtfs-eta"
    "/011eb36e-aa69-42fd-a73f-95fb1c198afb/scratchpad/bucr_gtfs.zip"
)
_REAL_FEED_DIR = Path(
    "/private/tmp/claude-501/-Users-dotj-Desktop-SIMOVI-git-no-sync-gtfs-eta"
    "/011eb36e-aa69-42fd-a73f-95fb1c198afb/scratchpad/bucr_gtfs"
)

requires_real_feed = pytest.mark.skipif(
    not _REAL_FEED_ZIP.exists(), reason="real BUCR GTFS snapshot not present locally"
)


# ---------------------------------------------------------------------------
# Synthetic in-memory feed
# ---------------------------------------------------------------------------

_SYNTH_FILES: Dict[str, List[Dict[str, str]]] = {
    "routes.txt": [
        {
            "route_id": "R1",
            "agency_id": "A1",
            "route_short_name": "1",
            "route_long_name": "Test Loop",
            "route_type": "3",
        }
    ],
    "trips.txt": [
        {"route_id": "R1", "service_id": "WK", "trip_id": "T1", "direction_id": "0", "shape_id": "S1"},
        {"route_id": "R1", "service_id": "WK", "trip_id": "T2", "direction_id": "0", "shape_id": "S1"},
        {"route_id": "R1", "service_id": "WK", "trip_id": "T3", "direction_id": "1", "shape_id": "S2"},
    ],
    "stops.txt": [
        {"stop_id": "A", "stop_name": "Stop A", "stop_lat": "0.0000", "stop_lon": "0.0000"},
        {"stop_id": "B", "stop_name": "Stop B", "stop_lat": "0.0000", "stop_lon": "0.0010"},
        {"stop_id": "C", "stop_name": "Stop C", "stop_lat": "0.0000", "stop_lon": "0.0020"},
    ],
    "stop_times.txt": [
        {"trip_id": "T1", "stop_id": "A", "stop_sequence": "1", "arrival_time": "08:00:00", "departure_time": "08:00:00"},
        {"trip_id": "T1", "stop_id": "B", "stop_sequence": "2", "arrival_time": "08:05:00", "departure_time": "08:05:00"},
        {"trip_id": "T1", "stop_id": "C", "stop_sequence": "3", "arrival_time": "08:10:00", "departure_time": "08:10:00"},
        {"trip_id": "T2", "stop_id": "A", "stop_sequence": "1", "arrival_time": "09:00:00", "departure_time": "09:00:00"},
        {"trip_id": "T2", "stop_id": "B", "stop_sequence": "2", "arrival_time": "09:05:00", "departure_time": "09:05:00"},
        {"trip_id": "T2", "stop_id": "C", "stop_sequence": "3", "arrival_time": "09:10:00", "departure_time": "09:10:00"},
        {"trip_id": "T3", "stop_id": "C", "stop_sequence": "1", "arrival_time": "10:00:00", "departure_time": "10:00:00"},
        {"trip_id": "T3", "stop_id": "A", "stop_sequence": "2", "arrival_time": "10:10:00", "departure_time": "10:10:00"},
    ],
    "shapes.txt": [
        {"shape_id": "S1", "shape_pt_lat": "0.0000", "shape_pt_lon": "0.0000", "shape_pt_sequence": "1"},
        {"shape_id": "S1", "shape_pt_lat": "0.0000", "shape_pt_lon": "0.0010", "shape_pt_sequence": "2"},
        {"shape_id": "S1", "shape_pt_lat": "0.0000", "shape_pt_lon": "0.0020", "shape_pt_sequence": "3"},
        {"shape_id": "S2", "shape_pt_lat": "0.0000", "shape_pt_lon": "0.0020", "shape_pt_sequence": "1"},
        {"shape_id": "S2", "shape_pt_lat": "0.0000", "shape_pt_lon": "0.0000", "shape_pt_sequence": "2"},
    ],
}


def _write_synthetic_feed(tmp_path: Path) -> Path:
    feed_dir = tmp_path / "synthetic_gtfs"
    feed_dir.mkdir()
    for name, rows in _SYNTH_FILES.items():
        fieldnames = list(rows[0].keys())
        with (feed_dir / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return feed_dir


def test_load_gtfs_from_directory_synthetic(tmp_path: Path) -> None:
    feed_dir = _write_synthetic_feed(tmp_path)
    data = load_gtfs(feed_dir)

    assert len(data.routes) == 1
    assert len(data.trips) == 3
    assert len(data.stops) == 3
    assert data.shapes["shape_id"].nunique() == 2
    assert len(data.shapes) == 5


def test_load_gtfs_missing_file_raises(tmp_path: Path) -> None:
    feed_dir = _write_synthetic_feed(tmp_path)
    (feed_dir / "shapes.txt").unlink()

    with pytest.raises(BucrGtfsError, match="missing required GTFS files"):
        load_gtfs(feed_dir)


def test_load_gtfs_nonexistent_path_raises(tmp_path: Path) -> None:
    with pytest.raises(BucrGtfsError, match="not a file or directory"):
        load_gtfs(tmp_path / "does_not_exist")


def test_route_directions_synthetic(tmp_path: Path) -> None:
    data = load_gtfs(_write_synthetic_feed(tmp_path))
    assert route_directions(data) == [("R1", 0), ("R1", 1)]


def test_shape_ids_for_route_direction_synthetic(tmp_path: Path) -> None:
    data = load_gtfs(_write_synthetic_feed(tmp_path))
    assert shape_ids_for_route_direction(data, "R1", 0) == ["S1"]
    assert shape_ids_for_route_direction(data, "R1", 1) == ["S2"]


def test_stops_for_trip_ordered_synthetic(tmp_path: Path) -> None:
    data = load_gtfs(_write_synthetic_feed(tmp_path))
    stops = stops_for_trip(data, "T1")
    assert [s.stop_id for s in stops] == ["A", "B", "C"]
    assert [s.stop_sequence for s in stops] == [1, 2, 3]
    assert all(s.progress_m is None for s in stops)


def test_stops_for_trip_unknown_trip_raises(tmp_path: Path) -> None:
    data = load_gtfs(_write_synthetic_feed(tmp_path))
    with pytest.raises(BucrGtfsError, match="unknown trip_id"):
        stops_for_trip(data, "does-not-exist")


def test_route_direction_candidates_synthetic(tmp_path: Path) -> None:
    data = load_gtfs(_write_synthetic_feed(tmp_path))
    candidates = route_direction_candidates(data, "R1", 0)

    assert len(candidates) == 1
    cand = candidates[0]
    assert cand.shape_id == "S1"
    assert cand.representative_trip_id == "T1"
    assert len(cand.polyline) == 3
    assert [s.stop_id for s in cand.stops] == ["A", "B", "C"]

    progress = [s.progress_m for s in cand.stops]
    assert all(p is not None for p in progress)
    assert progress == sorted(progress)  # monotonic non-decreasing


def test_route_direction_candidates_no_trips_raises(tmp_path: Path) -> None:
    data = load_gtfs(_write_synthetic_feed(tmp_path))
    with pytest.raises(BucrGtfsError, match="no trips for"):
        route_direction_candidates(data, "R1", 2)


# ---------------------------------------------------------------------------
# Real BUCR feed (S3 snapshot 2026-08-24.zip)
# ---------------------------------------------------------------------------


@requires_real_feed
def test_real_feed_load_from_zip() -> None:
    data = load_gtfs(_REAL_FEED_ZIP)
    assert len(data.routes) == 1
    assert len(data.stops) == 22
    assert len(data.trips) == 122
    assert data.shapes["shape_id"].nunique() == 7
    assert len(data.shapes) == 1057


@requires_real_feed
def test_real_feed_load_from_extracted_dir_matches_zip() -> None:
    zip_data = load_gtfs(_REAL_FEED_ZIP)
    dir_data = load_gtfs(_REAL_FEED_DIR)
    assert len(zip_data.trips) == len(dir_data.trips)
    assert len(zip_data.stops) == len(dir_data.stops)
    assert len(zip_data.shapes) == len(dir_data.shapes)


@requires_real_feed
def test_real_feed_costa_rica_geography() -> None:
    data = load_gtfs(_REAL_FEED_ZIP)
    lats = data.stops["stop_lat"].astype(float)
    lons = data.stops["stop_lon"].astype(float)
    assert lats.between(9.0, 11.0).all()
    assert lons.between(-85.0, -84.0).all()


@requires_real_feed
def test_real_feed_route_directions() -> None:
    data = load_gtfs(_REAL_FEED_ZIP)
    assert route_directions(data) == [("bUCR", 0), ("bUCR", 1)]


@requires_real_feed
def test_real_feed_candidates_build_polyline_and_monotonic_stops() -> None:
    data = load_gtfs(_REAL_FEED_ZIP)

    for route_id, direction_id in route_directions(data):
        candidates = route_direction_candidates(data, route_id, direction_id)
        assert candidates, f"no candidates for {route_id}/{direction_id}"
        for cand in candidates:
            assert len(cand.polyline) >= 2
            assert cand.polyline[0].cum_m == 0.0
            # cum_m is non-decreasing along the polyline
            cums = [p.cum_m for p in cand.polyline]
            assert cums == sorted(cums)

            assert cand.stops, f"no stops for shape {cand.shape_id}"
            progress = [s.progress_m for s in cand.stops]
            assert all(p is not None for p in progress)
            assert progress == sorted(progress)


@requires_real_feed
def test_real_feed_shape_dist_traveled_populated() -> None:
    data = load_gtfs(_REAL_FEED_ZIP)
    assert data.shapes["shape_dist_traveled"].notna().all()
    assert data.stop_times["shape_dist_traveled"].notna().all()
