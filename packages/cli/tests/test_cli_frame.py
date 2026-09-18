"""`stoptimes data frame` -- the seam between T2's derived Arrival table and
the training frame (T7), following the same conventions as `data arrivals`
(`packages/cli/tests/test_cli_data_arrivals.py`)."""

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
    from stoptimes.core.config import reset_config

    monkeypatch.setenv("STOPTIMES_HOME", str(tmp_path / "home"))
    reset_config()
    yield
    reset_config()


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


def _write_arrivals(path: Path) -> Path:
    import pandas as pd

    rows = [
        {
            "source_id": "demo", "trip_id": "t1", "start_date": "20260917",
            "route_id": "r1", "vehicle_id": "v1", "stop_id": sid, "stop_sequence": seq,
            "arrival_time": ts, "method": "reported", "confidence": None,
        }
        for sid, seq, ts in [
            ("s1", 1, "2026-09-17T08:00:00Z"),
            ("s2", 2, "2026-09-17T08:02:00Z"),
            ("s3", 3, "2026-09-17T08:04:00Z"),
        ]
    ]
    df = pd.DataFrame(rows)
    df["arrival_time"] = pd.to_datetime(df["arrival_time"], utc=True)
    df.to_parquet(path, index=False)
    return path


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


def _invoke_frame(tmp_path, *, out, register=None, extra_args=None):
    static = _write_static_zip(tmp_path / "gtfs.zip")
    positions = _write_positions(tmp_path / "archive")
    arrivals = _write_arrivals(tmp_path / "arrivals.parquet")

    args = [
        "data", "frame",
        "--observations", str(positions),
        "--arrivals", str(arrivals),
        "--static", str(static),
        "--source", "demo",
        "--sources-dir", str(_agency_file(tmp_path)),
        "--out", str(out),
        "--json",
    ]
    if register:
        args += ["--register", register]
    if extra_args:
        args += extra_args
    return runner.invoke(app, args)


def test_frame_built_from_positions_and_arrivals(tmp_path):
    out = tmp_path / "frame.parquet"
    result = _invoke_frame(tmp_path, out=out)

    assert result.exit_code == 0, result.output
    summary = json.loads(result.output)
    assert summary["rows"] > 0
    assert summary["labeled_rows"] == summary["rows"]
    assert out.is_file()

    import pandas as pd

    frame = pd.read_parquet(out)
    assert {"s1", "s2", "s3"} <= set(frame["stop_id"])
    # The first VP (sitting exactly at s1's departure) targets s1 with 0s to go.
    first_row = frame.iloc[0]
    assert first_row["stop_id"] == "s1"
    assert first_row["time_to_arrival_seconds"] == pytest.approx(0.0, abs=1.0)


def test_frame_registers_in_the_catalog(tmp_path):
    out = tmp_path / "frame.parquet"
    result = _invoke_frame(tmp_path, out=out, register="demo-frame")
    assert result.exit_code == 0, result.output

    listed = runner.invoke(app, ["data", "list", "--json"])
    entries = json.loads(listed.output)
    assert [e["name"] for e in entries] == ["demo-frame"]
    assert entries[0]["kind"] == "frame"


def test_frame_max_stops_ahead_option(tmp_path):
    out = tmp_path / "frame.parquet"
    result = _invoke_frame(tmp_path, out=out, extra_args=["--max-stops-ahead", "1"])
    assert result.exit_code == 0, result.output

    import pandas as pd

    frame = pd.read_parquet(out)
    assert (frame["stops_ahead"] == 0).all()


def test_missing_static_is_a_clear_error(tmp_path):
    positions = _write_positions(tmp_path / "archive")
    arrivals = _write_arrivals(tmp_path / "arrivals.parquet")
    result = runner.invoke(
        app,
        [
            "data", "frame",
            "--observations", str(positions),
            "--arrivals", str(arrivals),
            "--static", str(tmp_path / "nope.zip"),
            "--source", "demo",
            "--sources-dir", str(_agency_file(tmp_path)),
            "--out", str(tmp_path / "frame.parquet"),
        ],
    )
    assert result.exit_code != 0
    assert "nope.zip" in result.output


def test_frame_resolves_arrivals_from_the_catalog(tmp_path):
    static = _write_static_zip(tmp_path / "gtfs.zip")
    positions = _write_positions(tmp_path / "archive")
    arrivals_path = _write_arrivals(tmp_path / "arrivals.parquet")

    add_result = runner.invoke(
        app,
        ["data", "add", "demo-arrivals", str(arrivals_path), "--source", "demo", "--kind", "arrivals"],
    )
    assert add_result.exit_code == 0, add_result.output

    result = runner.invoke(
        app,
        [
            "data", "frame",
            "--observations", str(positions),
            "--arrivals", "demo-arrivals",
            "--static", str(static),
            "--source", "demo",
            "--sources-dir", str(_agency_file(tmp_path)),
            "--out", str(tmp_path / "frame.parquet"),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    summary = json.loads(result.output)
    assert summary["rows"] > 0
