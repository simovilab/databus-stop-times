"""`stoptimes data arrivals` — the seam between T1's readers/catalog and T2's
derivation, wired by the orchestrator once both had landed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from stoptimes.cli.main import app

runner = CliRunner()

pytest.importorskip("pandas")


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("STOPTIMES_HOME", str(tmp_path / "home"))


def _write_static_zip(path: Path) -> Path:
    """A one-trip, three-stop feed on a straight west-to-east line."""
    import zipfile

    stops = "stop_id,stop_name,stop_lat,stop_lon\ns1,One,0.0,0.0\ns2,Two,0.0,0.01\ns3,Three,0.0,0.02\n"
    trips = "route_id,service_id,trip_id,direction_id,shape_id\nr1,svc,t1,0,sh1\n"
    stop_times = (
        "trip_id,arrival_time,departure_time,stop_id,stop_sequence,shape_dist_traveled\n"
        "t1,08:00:00,08:00:00,s1,1,0\n"
        "t1,08:02:00,08:02:00,s2,2,1113\n"
        "t1,08:04:00,08:04:00,s3,3,2226\n"
    )
    shapes = (
        "shape_id,shape_pt_lat,shape_pt_lon,shape_pt_sequence,shape_dist_traveled\n"
        "sh1,0.0,0.0,1,0\nsh1,0.0,0.01,2,1113\nsh1,0.0,0.02,3,2226\n"
    )
    routes = "route_id,route_short_name,route_long_name\nr1,1,Line One\n"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("stops.txt", stops)
        zf.writestr("trips.txt", trips)
        zf.writestr("stop_times.txt", stop_times)
        zf.writestr("shapes.txt", shapes)
        zf.writestr("routes.txt", routes)
    return path


def _write_positions(root: Path) -> Path:
    """A vehicle driving that line, sampled every 30 s, in the partitioned
    archive layout the recorder writes."""
    import pandas as pd

    lons = [0.0, 0.002, 0.005, 0.008, 0.01, 0.012, 0.015, 0.018, 0.02]
    rows = [
        {
            "source_id": "demo",
            "kind": "vehicle_position",
            "trip_id": "t1",
            "route_id": "r1",
            "vehicle_id": "v1",
            "start_date": "20260917",
            "ts": pd.Timestamp("2026-09-17T08:00:00Z") + pd.Timedelta(seconds=30 * i),
            "lat": 0.0,
            "lon": lon,
        }
        for i, lon in enumerate(lons)
    ]
    partition = root / "year=2026" / "month=09" / "day=17" / "route_id=r1"
    partition.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(partition / "part-0.parquet")
    return root


def _agency_file(tmp_path: Path) -> Path:
    sources = tmp_path / "agencies"
    sources.mkdir()
    (sources / "demo.yaml").write_text(
        "name: demo\n"
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
    return sources


def test_arrivals_derived_from_positions(tmp_path):
    static = _write_static_zip(tmp_path / "gtfs.zip")
    positions = _write_positions(tmp_path / "archive")
    out = tmp_path / "arrivals.parquet"

    result = runner.invoke(
        app,
        [
            "data", "arrivals",
            "--observations", str(positions),
            "--static", str(static),
            "--source", "demo",
            "--sources-dir", str(_agency_file(tmp_path)),
            "--out", str(out),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    summary = json.loads(result.output)
    assert summary["arrivals"] >= 3
    assert out.is_file()

    import pandas as pd

    from stoptimes.core.contracts.data import ARRIVAL_SCHEMA, validate_frame

    arrivals = pd.read_parquet(out)
    assert validate_frame(arrivals, ARRIVAL_SCHEMA) == []
    assert set(arrivals["stop_id"]) == {"s1", "s2", "s3"}
    assert arrivals["arrival_time"].is_monotonic_increasing


def test_arrivals_registers_in_the_catalog(tmp_path):
    static = _write_static_zip(tmp_path / "gtfs.zip")
    positions = _write_positions(tmp_path / "archive")

    result = runner.invoke(
        app,
        [
            "data", "arrivals",
            "--observations", str(positions),
            "--static", str(static),
            "--source", "demo",
            "--sources-dir", str(_agency_file(tmp_path)),
            "--out", str(tmp_path / "arrivals.parquet"),
            "--register", "demo-arrivals",
        ],
    )
    assert result.exit_code == 0, result.output

    listed = runner.invoke(app, ["data", "list", "--json"])
    entries = json.loads(listed.output)
    assert [e["name"] for e in entries] == ["demo-arrivals"]
    assert entries[0]["kind"] == "arrivals"


def test_missing_static_is_a_clear_error(tmp_path):
    positions = _write_positions(tmp_path / "archive")
    result = runner.invoke(
        app,
        [
            "data", "arrivals",
            "--observations", str(positions),
            "--static", str(tmp_path / "nope.zip"),
            "--source", "demo",
            "--sources-dir", str(_agency_file(tmp_path)),
            "--out", str(tmp_path / "a.parquet"),
        ],
    )
    assert result.exit_code != 0
    assert "nope.zip" in result.output
