"""The Phase 2 gate: one synthetic agency, the whole chain, through the CLI.

    data add → data arrivals → data frame → train run → serve --once → val

Nothing here is agency-specific and nothing touches the network: the feed is
a three-stop line built in `tmp_path`. If this test passes, a stranger with a
GTFS zip and a position feed can get from raw data to a scored TripUpdates
feed with the documented commands.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from stoptimes.cli.main import app

pytest.importorskip("pandas")

runner = CliRunner()

SOURCE = "demo"
SERVICE_DAY = "20260917"


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Every artifact this test writes stays under tmp_path."""
    from stoptimes.core.config import reset_config

    monkeypatch.setenv("STOPTIMES_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("STOPTIMES_MODEL_REGISTRY_DIR", raising=False)
    monkeypatch.delenv("MODEL_REGISTRY_DIR", raising=False)
    reset_config()
    yield
    reset_config()


def _run(*argv: str):
    result = runner.invoke(app, list(argv))
    assert result.exit_code == 0, f"`stoptimes {' '.join(argv)}` failed:\n{result.output}"
    return result


def _static_zip(path: Path) -> Path:
    """A one-route, four-stop line running west to east, four trips a day."""
    stops = ["stop_id,stop_name,stop_lat,stop_lon"]
    for i in range(4):
        stops.append(f"s{i + 1},Stop {i + 1},0.0,{i * 0.01:.5f}")

    trips = ["route_id,service_id,trip_id,direction_id,shape_id"]
    stop_times = ["trip_id,arrival_time,departure_time,stop_id,stop_sequence,shape_dist_traveled"]
    for t in range(4):
        trip_id = f"t{t + 1}"
        trips.append(f"r1,svc,{trip_id},0,sh1")
        for i in range(4):
            clock = f"{8 + t:02d}:{i * 2:02d}:00"
            stop_times.append(f"{trip_id},{clock},{clock},s{i + 1},{i + 1},{i * 1113}")

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
    """Vehicle positions for those trips, in the layout the recorder writes."""
    import pandas as pd

    rows = []
    for t in range(4):
        start = pd.Timestamp(f"2026-09-17T{8 + t:02d}:00:00Z")
        for step in range(25):  # every 15 s, the whole line
            rows.append(
                {
                    "source_id": SOURCE,
                    "kind": "vehicle_position",
                    "trip_id": f"t{t + 1}",
                    "route_id": "r1",
                    "vehicle_id": f"v{t + 1}",
                    "start_date": SERVICE_DAY,
                    "ts": start + pd.Timedelta(seconds=15 * step),
                    "lat": 0.0,
                    "lon": round(step * 0.00125, 6),
                    "bearing": 90.0,
                }
            )

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


def _trip_updates_feed(path: Path) -> Path:
    """A GTFS-RT TripUpdates feed (JSON) for the day's trips: the input a
    serving deployment would poll from its own agency."""
    entities = []
    for t in range(4):
        stop_time_updates = [
            {
                "stop_sequence": i + 1,
                "stop_id": f"s{i + 1}",
                "arrival": {"time": 1789639200 + t * 3600 + i * 120},
            }
            for i in range(4)
        ]
        entities.append(
            {
                "id": f"e{t + 1}",
                "trip_update": {
                    "trip": {
                        "trip_id": f"t{t + 1}",
                        "route_id": "r1",
                        "direction_id": 0,
                        "start_date": SERVICE_DAY,
                    },
                    "vehicle": {"id": f"v{t + 1}"},
                    "stop_time_update": stop_time_updates,
                    "timestamp": 1789639200 + t * 3600,
                },
            }
        )
    path.write_text(
        json.dumps({"header": {"gtfs_realtime_version": "2.0", "timestamp": 1789639200}, "entity": entities}),
        encoding="utf-8",
    )
    return path


def _pipeline_file(path: Path, feed: Path) -> Path:
    """A user pipeline overriding two stages — the customizable inference
    story from the product page — leaving match/predict/postprocess/publish
    at their defaults."""
    path.write_text(
        f'''
import json
from pathlib import Path

import pandas as pd
from stoptimes.core.contracts.data import Observation, ObservationKind
from stoptimes.serve.pipeline import build_default_pipeline, default_features

pipeline = build_default_pipeline()


@pipeline.stage("ingest")
def ingest_from_file(_payload, ctx):
    feed = json.loads(Path(r"{feed}").read_text(encoding="utf-8"))
    observations = []
    for entity in feed["entity"]:
        tu = entity["trip_update"]
        trip = tu["trip"]
        for stu in tu["stop_time_update"]:
            observations.append(
                Observation(
                    kind=ObservationKind.TRIP_UPDATE,
                    source_id=ctx.source.id,
                    trip_id=trip["trip_id"],
                    route_id=trip.get("route_id"),
                    direction_id=trip.get("direction_id"),
                    start_date=trip.get("start_date"),
                    vehicle_id=(tu.get("vehicle") or {{}}).get("id"),
                    ts=pd.Timestamp(tu["timestamp"], unit="s", tz="UTC"),
                    stop_id=stu["stop_id"],
                    stop_sequence=stu["stop_sequence"],
                    arrival_time=pd.Timestamp(stu["arrival"]["time"], unit="s", tz="UTC"),
                )
            )
    return observations


@pipeline.stage("features")
def features_with_distance(matched, ctx):
    frame = default_features(matched, ctx)
    return frame
''',
        encoding="utf-8",
    )
    return path


def _experiment(path: Path, dataset: str, registry: Path) -> Path:
    path.write_text(
        f"""name: e2e-historical-mean
data:
  datasets: [{dataset}]
target: arrival_delay_s
model:
  family: historical_mean
  params: {{}}
features:
  - distance_to_stop
  - stops_ahead
  - hour
split:
  kind: time
  train_fraction: 0.7
  test_fraction: 0.3
seed: 7
outputs:
  registry_dir: {registry}
  tags: [e2e]
baselines: []
""",
        encoding="utf-8",
    )
    return path


def test_raw_feed_to_scored_predictions(tmp_path):
    static = _static_zip(tmp_path / "gtfs.zip")
    archive = _archive(tmp_path / "archive")
    agencies = _agency_dir(tmp_path)
    registry = tmp_path / "registry"

    # 1. register the recorded positions
    _run("data", "add", "positions", str(archive), "--source", SOURCE, "--kind", "observations")

    # 2. positions -> actual arrivals (the ground truth)
    arrivals_out = tmp_path / "arrivals.parquet"
    arrivals_summary = json.loads(
        _run(
            "data", "arrivals",
            "--observations", "positions",
            "--static", str(static),
            "--source", SOURCE,
            "--sources-dir", str(agencies),
            "--out", str(arrivals_out),
            "--register", "arrivals",
            "--json",
        ).output
    )
    assert arrivals_summary["arrivals"] >= 12, arrivals_summary  # 4 trips x 4 stops, minus misses

    # 3. arrivals + positions -> training frame
    frame_out = tmp_path / "frame.parquet"
    frame_summary = json.loads(
        _run(
            "data", "frame",
            "--observations", "positions",
            "--arrivals", "arrivals",
            "--static", str(static),
            "--source", SOURCE,
            "--sources-dir", str(agencies),
            "--out", str(frame_out),
            "--register", "frame",
            "--json",
        ).output
    )
    assert frame_summary["rows"] > 0, frame_summary

    # 4. train a model from an experiment file
    experiment = _experiment(tmp_path / "experiment.yaml", "frame", registry)
    _run("train", "run", str(experiment))

    listed = json.loads(_run("models", "list", "--registry", str(registry), "--json").output)
    assert len(listed) == 1, listed
    artifact_id = listed[0]["artifact_id"]
    assert (registry / artifact_id / "artifact.json").is_file()
    assert (registry / artifact_id / "card.md").is_file()

    # 5. serve one TripUpdates feed from that model, through a user pipeline
    served = tmp_path / "trip_updates.json"
    pipeline_file = _pipeline_file(
        tmp_path / "my_pipeline.py", _trip_updates_feed(tmp_path / "incoming_tu.json")
    )
    serve_summary = json.loads(
        _run(
            "serve",
            "--source", SOURCE,
            "--sources-dir", str(agencies),
            "--registry", str(registry),
            "--model", artifact_id,
            "--pipeline", f"{pipeline_file}:pipeline",
            "--sink", f"file:{served}",
            "--format", "json",
            "--once",
            "--json",
        ).output
    )
    assert serve_summary["errors"] == [], serve_summary
    assert serve_summary["trip_update_count"] > 0, serve_summary

    feed = json.loads(served.read_text(encoding="utf-8"))
    assert feed["header"]["gtfs_realtime_version"] == "2.0"
    assert feed["entity"], "served feed has no trip updates"
    first = feed["entity"][0]["trip_update"]
    assert first["trip"]["trip_id"].startswith("t")
    assert first["stop_time_update"], "trip update has no stop predictions"

    # 6. score that trained model against the recorded day, through validation
    results = tmp_path / "scores.parquet"
    _run(
        "val", "replay",
        "--source", SOURCE,
        "--sources-dir", str(agencies),
        "--archive", str(archive),
        "--gtfs", str(static),
        "--from", "2026-09-17",
        "--to", "2026-09-18",
        "--predictor", "baseline:schedule",
        "--out", str(results),
    )
    import pandas as pd

    scored = pd.read_parquet(results)
    assert len(scored) > 0, "validation produced no scored predictions"
