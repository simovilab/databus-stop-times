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


# --- data frame --live -----------------------------------------------------------


def _write_live_feed_and_positions(tmp_path: Path) -> tuple[Path, Path]:
    """Eight stops on a line, one trip a day at 08:00 UTC, six service days of pings."""
    import math
    import zipfile

    import pandas as pd

    m_per_deg = 6371000.0 * math.pi / 180.0
    spacing = 600.0

    def lon(progress_m: float) -> float:
        return progress_m / m_per_deg

    n = 8
    stops = "stop_id,stop_name,stop_lat,stop_lon\n" + "".join(f"s{i},S{i},0.0,{lon(i * spacing)}\n" for i in range(n))
    trips = "route_id,service_id,trip_id,direction_id,shape_id\nr1,svc,t1,0,sh1\n"
    rows = "".join(
        f"t1,08:{i:02d}:00,08:{i:02d}:00,s{i},{i + 1},{i * spacing}\n" for i in range(n)
    )
    stop_times = "trip_id,arrival_time,departure_time,stop_id,stop_sequence,shape_dist_traveled\n" + rows
    shapes = "shape_id,shape_pt_lat,shape_pt_lon,shape_pt_sequence,shape_dist_traveled\n" + "".join(
        f"sh1,0.0,{lon(i * 300.0)},{i + 1},{i * 300.0}\n" for i in range(n * 2 + 1)
    )
    static = tmp_path / "live-gtfs.zip"
    with zipfile.ZipFile(static, "w") as zf:
        zf.writestr("stops.txt", stops)
        zf.writestr("trips.txt", trips)
        zf.writestr("stop_times.txt", stop_times)
        zf.writestr("shapes.txt", shapes)
        zf.writestr("routes.txt", "route_id,route_short_name\nr1,1\n")

    root = tmp_path / "live-archive"
    paces = [1.0, 1.4, 0.8, 1.2, 1.1, 0.9]
    for day, pace in zip(range(1, 7), paces):
        t0 = pd.Timestamp(f"2026-09-{day:02d}T08:00:00Z")
        pings = [
            {
                "source_id": "demo", "kind": "vehicle_position", "trip_id": "t1", "route_id": "r1",
                "vehicle_id": "v1", "ts": t0 + pd.Timedelta(seconds=5 * k), "lat": 0.0,
                "lon": lon(min(5 * k * spacing / 60.0 * pace, (n - 1) * spacing + 100)),
            }
            for k in range(0, 100)
        ]
        part = root / "year=2026" / "month=09" / f"day={day:02d}" / "route_id=r1"
        part.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(pings).to_parquet(part / "part-0.parquet")
    return static, root


def test_frame_live_builds_frame_and_assets_then_train_run_makes_an_artifact(tmp_path):
    import pandas as pd

    static, positions = _write_live_feed_and_positions(tmp_path)
    out = tmp_path / "data" / "live-frame.parquet"
    result = runner.invoke(
        app,
        [
            "data", "frame", "--live", "--observations", str(positions), "--static", str(static),
            "--source", "demo", "--sources-dir", str(_agency_file(tmp_path)), "--out", str(out),
            "--train-days", "2026-09-01:2026-09-04", "--val-days", "2026-09-05:2026-09-06",
            "--rows-per-day", "40", "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    summary = json.loads(result.output)
    assert summary["train_days"] == 4 and summary["val_days"] == 2
    assert 0 < summary["train_rows"] <= 160 and 0 < summary["val_rows"] <= 80  # <= 40 rows a day
    assert set(summary["asset_sha256"]) == {"segment_stats", "route_vocabulary"}
    frame = pd.read_parquet(out)
    from stoptimes.core.features.live import LIVE_FEATURES

    assert set(LIVE_FEATURES) <= set(frame.columns)
    assert (out.with_suffix(".assets") / "live.json").is_file()

    experiment = tmp_path / "xgb-live.yaml"
    experiment.write_text(
        "name: cli-live\n"
        "data: {datasets: [live-frame]}\n"
        "target: time_to_arrival_seconds\n"
        "model: {family: xgboost, params: {n_estimators: 20, max_depth: 3, early_stopping_rounds: 5, n_jobs: 2}}\n"
        "features: [" + ", ".join(LIVE_FEATURES) + "]\n"
        "split: {kind: time, train_fraction: 0.9, test_fraction: 0.1}\n"
        "baselines: []\n",
        encoding="utf-8",
    )
    trained = runner.invoke(
        app,
        ["train", "run", str(experiment), "--registry", str(tmp_path / "reg"), "--dataset-dir", str(out.parent), "--json"],
    )
    assert trained.exit_code == 0, trained.output
    assert json.loads(trained.output)["artifact_id"] == "cli-live"
    assert (tmp_path / "reg" / "cli-live" / "segment_stats.bin").is_file()


def test_frame_live_needs_day_ranges(tmp_path):
    static, positions = _write_live_feed_and_positions(tmp_path)
    result = runner.invoke(
        app,
        ["data", "frame", "--live", "--observations", str(positions), "--static", str(static), "--source", "demo",
         "--sources-dir", str(_agency_file(tmp_path)), "--out", str(tmp_path / "f.parquet")],
    )
    assert result.exit_code == 2
