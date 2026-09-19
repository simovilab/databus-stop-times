"""test_pipeline_adapter_cli — `pipeline:<file.py>` predictor, driven through the
real `stoptimes val run` CLI (T6/F1 defect fix).

A third-party newcomer running `--predictor pipeline:my_pipeline.py` (the bare
form the docs use, no `:attr` suffix) hit a TypeError/ValueError deep inside
`stoptimes.serve.pipeline.load_pipeline` -- `pipeline_adapter.py` passed a
`Path` where a `"spec:attr"` string was required, and never appended the
mandatory attribute name at all. A unit test that calls the adapter directly
(as `test_predictors_pipeline_adapter.py` does, with `load_pipeline` mocked
out) would repeat the mistake that let this ship -- `pipeline:` was never
exercised end to end. These tests instead drive the real CLI, on disk, with a
real user pipeline file, exactly the way a newcomer would, and assert
predictions actually come out (not merely that the command exits 0).
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from stoptimes.val.cli.main import app
from tests.test_feeds_replay_archive import _base_vp_rows, _write_vp_archive

runner = CliRunner()

_FIXTURES = Path(__file__).parent / "fixtures"
_MINI_GTFS = _FIXTURES / "mini_gtfs.zip"


# A minimal, real user pipeline: default match/features/postprocess/publish,
# with `ingest` turning the replayed vehicle state into an Observation (the
# same pattern the docs use) and `predict` standing in for a trained model
# with a constant horizon, so the test needs no trained artifact on disk.
_GOOD_PIPELINE = '''
import pandas as pd

from stoptimes.core.contracts.data import Observation, ObservationKind
from stoptimes.core.contracts.prediction import StopTimePrediction
from stoptimes.serve.pipeline import build_default_pipeline

pipeline = build_default_pipeline()


@pipeline.stage("ingest")
def ingest_vehicle_state(_payload, ctx):
    trip = ctx.state["trip"]
    vs = ctx.state["vehicle_state"]
    return [
        Observation(
            kind=ObservationKind.VEHICLE_POSITION,
            source_id=trip.source,
            trip_id=trip.trip_id,
            route_id=trip.route_id,
            direction_id=trip.direction_id,
            start_date=trip.start_date,
            vehicle_id=trip.vehicle_id,
            ts=pd.Timestamp(vs.epoch, unit="s", tz="UTC"),
            lat=vs.lat,
            lon=vs.lon,
        )
    ]


@pipeline.stage("predict")
def predict_constant(features, ctx):
    predictions = []
    for _, row in features.iterrows():
        predictions.append(
            StopTimePrediction(
                trip_id=row.get("trip_id"),
                route_id=row.get("route_id"),
                direction_id=(int(row["direction_id"]) if pd.notna(row.get("direction_id")) else None),
                start_date=row.get("start_date") or ctx.now.strftime("%Y%m%d"),
                stop_id=row.get("stop_id"),
                stop_sequence=int(row.get("stop_sequence")),
                vehicle_id=row.get("vehicle_id"),
                predicted_arrival_time=ctx.now,
                horizon_s=42.0,
                model_id="constant-test",
                artifact_id=None,
                generated_at=ctx.now,
            )
        )
    return predictions


@pipeline.stage("publish")
def publish_return(predictions, ctx):
    # val reads the pipeline's return value instead of a real sink.
    return predictions
'''

_BROKEN_PIPELINE = '''
from stoptimes.serve.pipeline import build_default_pipeline

pipeline = build_default_pipeline()


@pipeline.stage("ingest")
def ingest_boom(_payload, ctx):
    raise RuntimeError("boom: pipeline misconfigured by the user")
'''


def _run_with_predictor(tmp_path: Path, predictor_arg: str, out: Path):
    vp_dir = _write_vp_archive(tmp_path, _base_vp_rows())
    return runner.invoke(
        app,
        [
            "run",
            "--source", "mbta",
            "--replay-vp", str(vp_dir),
            "--replay-gtfs", str(_MINI_GTFS),
            "--predictor", predictor_arg,
            "--out", str(out),
        ],
    )


def test_bare_pipeline_predictor_id_produces_predictions(tmp_path: Path) -> None:
    """`--predictor pipeline:my_pipeline.py` (no `:attr` suffix, the docs' form)
    must actually load and score -- not raise a TypeError/ValueError on every tick."""
    pipeline_file = tmp_path / "my_pipeline.py"
    pipeline_file.write_text(_GOOD_PIPELINE, encoding="utf-8")
    out = tmp_path / "results.parquet"

    result = _run_with_predictor(tmp_path, f"pipeline:{pipeline_file}", out)

    assert result.exit_code == 0, result.output
    assert "Predictor errors" not in result.output, result.output

    summary = json.loads(Path(f"{out}.summary.json").read_text())
    assert summary["samples_saved"] > 0, summary
    assert summary["predictor_errors"] == {}, summary


def test_explicit_attr_pipeline_predictor_id_also_works(tmp_path: Path) -> None:
    """`pipeline:<file.py>:pipeline` (explicit attribute) must keep working too."""
    pipeline_file = tmp_path / "my_pipeline.py"
    pipeline_file.write_text(_GOOD_PIPELINE, encoding="utf-8")
    out = tmp_path / "results.parquet"

    result = _run_with_predictor(tmp_path, f"pipeline:{pipeline_file}:pipeline", out)

    assert result.exit_code == 0, result.output
    summary = json.loads(Path(f"{out}.summary.json").read_text())
    assert summary["samples_saved"] > 0, summary


def test_broken_pipeline_names_the_real_exception(tmp_path: Path) -> None:
    """A predictor that fails on every tick must surface its actual cause --
    on screen and in summary.json -- not just a bare error count."""
    pipeline_file = tmp_path / "broken_pipeline.py"
    pipeline_file.write_text(_BROKEN_PIPELINE, encoding="utf-8")
    out = tmp_path / "results.parquet"

    result = _run_with_predictor(tmp_path, f"pipeline:{pipeline_file}", out)

    assert result.exit_code == 0, result.output
    assert "Predictor errors" in result.output, result.output
    assert "RuntimeError" in result.output, result.output
    assert "boom: pipeline misconfigured by the user" in result.output, result.output
    assert "FAILED EVERY TICK" in result.output, result.output

    summary = json.loads(Path(f"{out}.summary.json").read_text())
    predictor_id = f"pipeline:{pipeline_file.name}"
    assert predictor_id in summary["predictor_errors"]
    assert "RuntimeError" in summary["predictor_first_error"][predictor_id]
    assert "boom: pipeline misconfigured by the user" in summary["predictor_first_error"][predictor_id]


def test_pipeline_predictor_run_does_not_emit_legacy_model_registry_warning(tmp_path: Path, recwarn) -> None:
    """A newcomer using only `pipeline:<file.py>` must never see gtfs-eta's
    legacy MODEL_REGISTRY_DIR RuntimeWarning -- that module is only imported
    (via engine/assembly.py, eagerly, to register the gtfs_eta:*/stoptimes:*
    predictor factories) here, never constructed, so the warning must not fire."""
    pipeline_file = tmp_path / "my_pipeline.py"
    pipeline_file.write_text(_GOOD_PIPELINE, encoding="utf-8")
    out = tmp_path / "results.parquet"

    result = _run_with_predictor(tmp_path, f"pipeline:{pipeline_file}", out)

    assert result.exit_code == 0, result.output
    registry_warnings = [
        w for w in recwarn.list if "MODEL_REGISTRY_DIR" in str(w.message)
    ]
    assert not registry_warnings, [str(w.message) for w in registry_warnings]
    assert "MODEL_REGISTRY_DIR" not in result.output, result.output
