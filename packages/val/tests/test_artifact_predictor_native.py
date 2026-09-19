"""test_artifact_predictor_native — F2: `stoptimes:<artifact>` must score a
*native* artifact (`stoptimes train run` output, no `legacy` migration
metadata), the defect a third-party tester hit (`artifact.py:149` used to
raise `RuntimeError` for every such artifact).

Correctness bar (see docs/PLAN.md's F2 brief): a predictor that returns
*some* number isn't good enough -- these tests train one real artifact and
check that `StoptimesArtifactPredictor` agrees with an independent,
from-scratch run of `stoptimes serve`'s own pipeline
(`serve.pipeline.build_default_pipeline`, including its real `default_match`
map-matching against the same static GTFS) on the same vehicle position.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from stoptimes.cli.main import app as cli_app
from stoptimes.core.contracts.artifact import read_metadata
from stoptimes.core.contracts.data import Observation, ObservationKind
from stoptimes.core.contracts.pipeline import PipelineContext
from stoptimes.core.domain.models import Stop, TripInstance, VehicleState
from stoptimes.core.readers import read_static
from stoptimes.core.spatial.polyline import (
    assign_stops_monotonic,
    build_polyline,
    project_point_to_polyline,
)
from stoptimes.serve.model_loading import load_artifact_model
from stoptimes.serve.pipeline import build_default_pipeline
from stoptimes.val.predictors.artifact import StoptimesArtifactPredictor

runner = CliRunner()

_MODELS_FIXTURES = Path(__file__).resolve().parents[2] / "models" / "tests" / "fixtures"
_MINI_GTFS = Path(__file__).parent / "fixtures" / "mini_gtfs.zip"

TRIP = TripInstance(
    source="test",
    trip_id="trip1",
    vehicle_id="v1",
    route_id="r1",
    direction_id=0,
    shape_id="shape1",
    start_date="20260701",
    started_at=1_704_153_600.0,
)

# A position between s1 (42.3601,-71.0589) and s2 (42.3610,-71.0580) on trip1's shape.
_VEHICLE_LAT = 42.3606
_VEHICLE_LON = -71.0584
_VEHICLE_EPOCH = 1_704_153_600.0


def _train_native_historical_mean_artifact(tmp_path: Path) -> tuple[Path, str]:
    """Train one real, native (non-migrated) `historical_mean` artifact via
    `stoptimes train run`, with the exact feature list the quickstart's
    `--predictor stoptimes:<artifact>` suggestion trains
    (`distance_to_stop`, `stops_ahead`, `hour`)."""
    experiment = {
        "name": "f2-native-test",
        "data": {"datasets": ["sample"]},
        "target": "time_to_arrival_seconds",
        "model": {"family": "historical_mean", "params": {}},
        "features": ["distance_to_stop", "stops_ahead", "hour"],
        "split": {"kind": "time", "train_fraction": 0.7, "test_fraction": 0.3},
        "seed": 7,
        "outputs": {"registry_dir": str(tmp_path / "registry")},
        "baselines": [],
    }
    path = tmp_path / "f2.experiment.yaml"
    path.write_text(yaml.safe_dump(experiment), encoding="utf-8")

    result = runner.invoke(
        cli_app, ["train", "run", str(path), "--dataset-dir", str(_MODELS_FIXTURES)]
    )
    assert result.exit_code == 0, result.output

    registry = tmp_path / "registry"
    artifact_dir = next(d for d in registry.iterdir() if (d / "artifact.json").is_file())

    meta = json.loads((artifact_dir / "artifact.json").read_text())
    assert meta["legacy"] is None, "test fixture must be a native (non-migrated) artifact"

    return registry, artifact_dir.name


def _real_matched_inputs() -> tuple[list[Stop], VehicleState]:
    """Map-match the test vehicle position against mini_gtfs's trip1 shape,
    the same way `val/engine/runtime.py` does, producing `progress_m`-bearing
    `Stop`/`VehicleState` for `StoptimesArtifactPredictor.predict()`."""
    loader, _feed = read_static(str(_MINI_GTFS), "test")
    assert loader is not None
    shape_points, raw_stops = loader.trip_geometry("trip1")
    polyline = build_polyline(shape_points)
    stops_with_progress = assign_stops_monotonic(raw_stops, polyline)

    projection = project_point_to_polyline(_VEHICLE_LAT, _VEHICLE_LON, polyline)
    vehicle_state = VehicleState(
        epoch=_VEHICLE_EPOCH,
        lat=_VEHICLE_LAT,
        lon=_VEHICLE_LON,
        progress_m=projection.progress_m,
        speed=3.0,
        current_status="IN_TRANSIT_TO",
    )

    from stoptimes.core.spatial.polyline import upcoming_stops as _upcoming_stops

    upcoming = _upcoming_stops(projection.progress_m, stops_with_progress)
    return upcoming, vehicle_state


def _serve_pipeline_predictions(registry: Path, artifact_id: str):
    """An independent, from-scratch run of `stoptimes serve`'s own pipeline
    (real `default_match` map-matching against the same static GTFS, real
    `default_features`/`default_predict`) for the same vehicle position --
    the ground truth `StoptimesArtifactPredictor` must agree with."""
    meta = read_metadata(registry / artifact_id)
    model = load_artifact_model(meta, registry)

    pipeline = build_default_pipeline()

    observation = Observation(
        kind=ObservationKind.VEHICLE_POSITION,
        source_id="test",
        trip_id=TRIP.trip_id,
        route_id=TRIP.route_id,
        direction_id=TRIP.direction_id,
        start_date=TRIP.start_date,
        vehicle_id=TRIP.vehicle_id,
        ts=pd.Timestamp(_VEHICLE_EPOCH, unit="s", tz="UTC"),
        lat=_VEHICLE_LAT,
        lon=_VEHICLE_LON,
    )

    @pipeline.stage("ingest")
    def _ingest(_payload, ctx):
        return [observation]

    @pipeline.stage("predict")
    def _predict(features, ctx):
        from stoptimes.serve.pipeline import default_predict

        ctx.model = model
        return default_predict(features, ctx)

    @pipeline.stage("publish")
    def _publish(predictions, ctx):
        # Return the postprocessed predictions directly instead of publishing
        # to a sink -- this test reads the pipeline's own return value.
        return predictions

    loader, _feed = read_static(str(_MINI_GTFS), "test")

    class _Source:
        id = "test"
        gtfs_static_url = str(_MINI_GTFS)
        timezone = "UTC"
        holiday_region = None

    ctx = PipelineContext(
        source=_Source(),
        static_feed=loader,
        model=model,
        now=datetime.fromtimestamp(_VEHICLE_EPOCH, tz=UTC),
    )
    return pipeline.run(ctx)


def test_native_artifact_predictor_agrees_with_serve_pipeline(tmp_path: Path) -> None:
    registry, artifact_id = _train_native_historical_mean_artifact(tmp_path)

    upcoming_stops, vehicle_state = _real_matched_inputs()
    assert upcoming_stops, "vehicle position must resolve at least one upcoming stop"

    predictor = StoptimesArtifactPredictor(artifact_id, registry_dir=registry)
    val_preds = predictor.predict(TRIP, upcoming_stops, vehicle_state, None)
    assert val_preds, "the native artifact bridge must produce predictions, not silently drop them"

    serve_preds = _serve_pipeline_predictions(registry, artifact_id)
    assert serve_preds, "the independent serve.pipeline run must also produce predictions"

    val_by_stop = {p.stop_id: p for p in val_preds}
    serve_by_stop = {p.stop_id: p for p in serve_preds}
    assert set(val_by_stop) == set(serve_by_stop)

    for stop_id, val_pred in val_by_stop.items():
        serve_pred = serve_by_stop[stop_id]
        val_horizon_s = val_pred.predicted_arrival_epoch - val_pred.made_at_epoch
        serve_horizon_s = serve_pred.horizon_s
        assert val_horizon_s == pytest.approx(serve_horizon_s, abs=1e-6), (
            f"stop {stop_id!r}: val predictor horizon {val_horizon_s}s != "
            f"serve.pipeline horizon {serve_horizon_s}s -- validation and serving "
            "disagree on the same artifact/input, which defeats the point of validation"
        )


def test_quickstart_closing_suggestion_via_val_replay_cli(tmp_path: Path, monkeypatch) -> None:
    """CLI-level regression test for the quickstart's closing suggestion:
    `--predictor stoptimes:<artifact>` against a freshly trained artifact
    must produce real predictions through `val replay`, not 103 silent
    errors. Registry is found via `STOPTIMES_MODEL_REGISTRY_DIR` (there is
    no `--registry` flag on `val replay`)."""
    from stoptimes.val.cli.main import app as val_app
    from tests.test_feeds_replay_archive import _base_vp_rows, _write_vp_archive

    registry, artifact_id = _train_native_historical_mean_artifact(tmp_path)
    monkeypatch.setenv("STOPTIMES_MODEL_REGISTRY_DIR", str(registry))

    vp_dir = _write_vp_archive(tmp_path, _base_vp_rows())
    out = tmp_path / "quickstart-results.parquet"

    result = runner.invoke(
        val_app,
        [
            "replay",
            "--source", "mbta",
            "--archive", str(vp_dir),
            "--gtfs", str(_MINI_GTFS),
            "--predictor", f"stoptimes:{artifact_id}",
            "--out", str(out),
            "--seed", "7",
        ],
    )

    assert result.exit_code == 0, result.output
    assert out.exists(), result.output

    scores = pd.read_parquet(out)
    predictor_scores = scores[scores["predictor_id"] == f"stoptimes:{artifact_id}"]
    assert len(predictor_scores) > 0, (
        f"'stoptimes:{artifact_id}' produced no scored predictions via val replay "
        f"-- the quickstart's closing suggestion must actually work. Full output: {result.output}"
    )
