"""Tests for stoptimes.core.api.build_arrivals_dataset (the `data arrivals` verb)."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
import pytest

from stoptimes.core.api import add_dataset, build_arrivals_dataset, get_dataset

SOURCE = "demo"


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    from stoptimes.core.config import reset_config

    monkeypatch.setenv("STOPTIMES_HOME", str(tmp_path / "home"))
    reset_config()
    yield
    reset_config()


def _static_zip(path: Path) -> Path:
    """A one-route, four-stop line, one trip a day."""
    stops = ["stop_id,stop_name,stop_lat,stop_lon"]
    for i in range(4):
        stops.append(f"s{i + 1},Stop {i + 1},0.0,{i * 0.01:.5f}")
    trips = ["route_id,service_id,trip_id,direction_id,shape_id\nr1,svc,t1,0,sh1"]
    stop_times = ["trip_id,arrival_time,departure_time,stop_id,stop_sequence,shape_dist_traveled"]
    for i in range(4):
        clock = f"08:{i * 2:02d}:00"
        stop_times.append(f"t1,{clock},{clock},s{i + 1},{i + 1},{i * 1113}")
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


def _archive(root: Path) -> Path:
    start = pd.Timestamp("2026-09-17T08:00:00Z")
    rows = [
        {
            "source_id": SOURCE,
            "kind": "vehicle_position",
            "trip_id": "t1",
            "route_id": "r1",
            "vehicle_id": "v1",
            "start_date": "20260917",
            "ts": start + pd.Timedelta(seconds=15 * step),
            "lat": 0.0,
            "lon": round(step * 0.00125, 6),
            "bearing": 90.0,
        }
        for step in range(25)
    ]
    partition = root / "year=2026" / "month=09" / "day=17" / "route_id=r1"
    partition.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(partition / "part-0.parquet")
    return root


def _agency_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "agencies"
    directory.mkdir(exist_ok=True)
    (directory / f"{SOURCE}.yaml").write_text(
        f"name: {SOURCE}\n"
        "gtfs_static_url: unused.zip\n"
        "vehicle_positions_url: http://example.invalid/vp.json\n"
        "trip_updates_url: http://example.invalid/tu.json\n"
        "poll_interval_s: 15\n"
        "timezone: UTC\n"
        "arrival_radius_m: 60.0\n"
        "dwell_s: 0.0\n"
        "shape_threshold_m: 50.0\n",
        encoding="utf-8",
    )
    return directory


def test_build_arrivals_dataset_writes_parquet_and_registers(tmp_path):
    static = _static_zip(tmp_path / "gtfs.zip")
    archive_dir = _archive(tmp_path / "archive")
    agencies = _agency_dir(tmp_path)
    add_dataset("positions", str(archive_dir), source=SOURCE, kind="observations")

    out = tmp_path / "arrivals.parquet"
    frame, summary = build_arrivals_dataset(
        observations="positions",
        static=str(static),
        source=SOURCE,
        sources_dir=str(agencies),
        out=str(out),
        register="arrivals",
    )

    assert out.is_file()
    assert summary["arrivals"] == len(frame) > 0
    assert get_dataset("arrivals") is not None
    assert get_dataset("arrivals").row_count == len(frame)


def test_build_arrivals_dataset_refuses_to_overwrite(tmp_path):
    static = _static_zip(tmp_path / "gtfs.zip")
    archive_dir = _archive(tmp_path / "archive")
    agencies = _agency_dir(tmp_path)
    add_dataset("positions", str(archive_dir), source=SOURCE, kind="observations")

    out = tmp_path / "arrivals.parquet"
    build_arrivals_dataset(
        observations="positions", static=str(static), source=SOURCE, sources_dir=str(agencies), out=str(out)
    )
    with pytest.raises(FileExistsError):
        build_arrivals_dataset(
            observations="positions", static=str(static), source=SOURCE, sources_dir=str(agencies), out=str(out)
        )
