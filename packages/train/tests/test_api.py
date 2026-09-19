"""Tests for stoptimes.train.api: build_frame, run_experiment_from_file,
select_model, predict_once -- the training/serving half of the U4 Python API
that `stoptimes.core.api` can't own (docs/PLAN.md §1 rule 1: core must not
import train/serve).

Reuses the same synthetic one-route, four-stop, four-trip fixture as
`tests/e2e/test_end_to_end.py` (kept local here, not shared, so this
package's tests still run standalone).
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest
from stoptimes.core.api import add_dataset, build_arrivals_dataset, resolve_source
from stoptimes.train.api import (
    ArtifactExistsError,
    build_frame,
    predict_once,
    run_experiment_from_file,
    select_model,
)

SOURCE = "demo"
SERVICE_DAY = "20260917"


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    from stoptimes.core.config import reset_config

    monkeypatch.setenv("STOPTIMES_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("STOPTIMES_MODEL_REGISTRY_DIR", raising=False)
    monkeypatch.delenv("MODEL_REGISTRY_DIR", raising=False)
    reset_config()
    yield
    reset_config()


def _static_zip(path: Path) -> Path:
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
    rows = []
    for t in range(4):
        start = pd.Timestamp(f"2026-09-17T{8 + t:02d}:00:00Z")
        for step in range(25):
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


def _experiment(path: Path, dataset: str, registry: Path, *, name: str = "test-experiment") -> Path:
    path.write_text(
        f"""name: {name}
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
  tags: [test]
baselines: []
""",
        encoding="utf-8",
    )
    return path


@pytest.fixture()
def frame_setup(tmp_path):
    static = _static_zip(tmp_path / "gtfs.zip")
    archive = _archive(tmp_path / "archive")
    agencies = _agency_dir(tmp_path)

    add_dataset("positions", str(archive), source=SOURCE, kind="observations")
    build_arrivals_dataset(
        observations="positions",
        static=str(static),
        source=SOURCE,
        sources_dir=str(agencies),
        out=str(tmp_path / "arrivals.parquet"),
        register="arrivals",
    )
    frame, summary = build_frame(
        observations="positions",
        arrivals="arrivals",
        static=str(static),
        source=SOURCE,
        sources_dir=str(agencies),
        out=str(tmp_path / "frame.parquet"),
        register="frame",
    )
    return {"tmp_path": tmp_path, "static": static, "archive": archive, "agencies": agencies, "frame": frame, "summary": summary}


def test_build_frame_writes_and_registers(frame_setup):
    from stoptimes.core.api import get_dataset

    assert frame_setup["summary"]["rows"] > 0
    entry = get_dataset("frame")
    assert entry is not None
    assert entry.row_count == len(frame_setup["frame"])


def test_build_frame_refuses_to_overwrite(frame_setup):
    tmp_path = frame_setup["tmp_path"]
    with pytest.raises(FileExistsError):
        build_frame(
            observations="positions",
            arrivals="arrivals",
            static=str(frame_setup["static"]),
            source=SOURCE,
            sources_dir=str(frame_setup["agencies"]),
            out=str(tmp_path / "frame.parquet"),
        )


def test_run_experiment_from_file_and_select_model(frame_setup):
    tmp_path = frame_setup["tmp_path"]
    registry = tmp_path / "registry"
    experiment = _experiment(tmp_path / "experiment.yaml", "frame", registry)

    meta = run_experiment_from_file(experiment)
    assert meta.family == "historical_mean"
    assert (registry / meta.artifact_id / "artifact.json").is_file()

    with pytest.raises(ArtifactExistsError):
        run_experiment_from_file(experiment)

    src = resolve_source(SOURCE, sources_dir=frame_setup["agencies"])
    picked = select_model(registry=registry, source_id=src.id, artifact_id=meta.artifact_id)
    assert picked is not None
    assert picked.artifact_id == meta.artifact_id

    missing = select_model(registry=registry, source_id=src.id, artifact_id="nonexistent")
    assert missing is None


def _trip_updates_feed(path: Path) -> Path:
    entities = []
    for t in range(4):
        stop_time_updates = [
            {"stop_sequence": i + 1, "stop_id": f"s{i + 1}", "arrival": {"time": 1789639200 + t * 3600 + i * 120}}
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
    frame["distance_to_stop"] = 300.0
    frame["stops_ahead"] = frame["stop_sequence"].astype(float)
    return frame
''',
        encoding="utf-8",
    )
    return path


def test_predict_once_publishes_a_feed(frame_setup):
    tmp_path = frame_setup["tmp_path"]
    registry = tmp_path / "registry"
    experiment = _experiment(tmp_path / "experiment.yaml", "frame", registry)
    meta = run_experiment_from_file(experiment)

    src = resolve_source(SOURCE, sources_dir=frame_setup["agencies"])
    feed_path = _trip_updates_feed(tmp_path / "incoming_tu.json")
    pipeline_file = _pipeline_file(tmp_path / "my_pipeline.py", feed_path)
    served = tmp_path / "trip_updates.json"

    summary = predict_once(
        source=src,
        registry=registry,
        artifact_id=meta.artifact_id,
        pipeline=f"{pipeline_file}:pipeline",
        sink=f"file:{served}",
        format="json",
    )

    assert summary["errors"] == []
    assert summary["trip_update_count"] > 0
    assert summary["artifact_id"] == meta.artifact_id
    fed = json.loads(served.read_text(encoding="utf-8"))
    assert fed["entity"]
