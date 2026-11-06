"""Validation runs are logged to MLflow (docs/MLFLOW_INTEGRATION.md §0 D6), best-effort.

Local sqlite MLflow stores in tmp_path, no network."""

from __future__ import annotations

import asyncio

import pytest
from stoptimes.core.domain.models import ValidationResult
from stoptimes.core.tracking import Tracker
from stoptimes.val.api.runs import RunManager
from stoptimes.val.scoring.metrics import compute_metrics
from stoptimes.val.tracking import log_validation, predictor_key, report_metrics, validation_record

from tests.test_api import _SOURCES_DIR, _replay_deps_builder

pytest.importorskip("mlflow")


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_DISABLE_TELEMETRY", "true")
    monkeypatch.delenv("STOPTIMES_TRACKING", raising=False)
    monkeypatch.chdir(tmp_path)  # default artifact roots land here, never in the repo


def _result(pid: str, error_s: float, horizon_s: float) -> ValidationResult:
    return ValidationResult(
        trip_key="t1",
        predictor_id=pid,
        stop_id="s1",
        stop_sequence=1,
        route_id="15",
        error_s=error_s,
        horizon_s=horizon_s,
        predicted_arrival_epoch=1000.0 + error_s,
        actual_arrival_epoch=1000.0,
        made_at_epoch=1000.0 - horizon_s,
    )


_RESULTS = [
    _result("mlflow:stoptimes-mbta@staging", 60.0, 90.0),
    _result("mlflow:stoptimes-mbta@staging", -20.0, 400.0),
    _result("gtfsrt", 100.0, 90.0),
    _result("gtfsrt", 140.0, 700.0),
]


def _runs(uri: str):
    from mlflow import MlflowClient

    client = MlflowClient(uri)
    exp = client.get_experiment_by_name("stoptimes-val/mbta")
    assert exp is not None
    return client, client.search_runs([exp.experiment_id])


def test_predictor_keys_are_readable():
    assert predictor_key("mlflow:stoptimes-mbta@staging") == "mlflow_stoptimes-mbta_staging"
    assert predictor_key("mlflow:stoptimes-mbta/3") == "mlflow_stoptimes-mbta_v3"
    assert predictor_key("baseline:schedule") == "baseline_schedule"


def test_metrics_cover_every_predictor_and_horizon_bucket():
    metrics = report_metrics(compute_metrics(_RESULTS))
    assert metrics["mlflow_stoptimes-mbta_staging.mae_s"] == 40.0
    assert metrics["gtfsrt.bias_s"] == 120.0
    assert metrics["gtfsrt.n"] == 2.0
    assert metrics["gtfsrt.lt2m.mae_s"] == 100.0
    assert metrics["gtfsrt.gt10m.mae_s"] == 140.0
    assert metrics["mlflow_stoptimes-mbta_staging.5-10m.bias_s"] == -20.0


def test_record_tags_the_model_version_an_alias_resolved_to():
    record = validation_record(
        compute_metrics(_RESULTS),
        source="mbta",
        mode="live",
        run_id="986b90ee-7078",
        predictor_ids=["mlflow:stoptimes-mbta@staging", "gtfsrt"],
        routes=["15", "222"],
        status="error",
        error="ConnectError: dns",
        models={
            "mlflow:stoptimes-mbta@staging": {
                "version": "stoptimes-mbta/1",
                "artifact_id": "xgb-live",
            }
        },
    )
    assert record.experiment == "stoptimes-val/mbta"
    assert record.tags["stoptimes.model.mlflow_stoptimes-mbta_staging"] == "stoptimes-mbta/1"
    assert record.tags["stoptimes.artifact.mlflow_stoptimes-mbta_staging"] == "xgb-live"
    assert record.tags["stoptimes.val.status"] == "error"
    assert record.params["routes"] == "15,222"
    assert "report.json" in record.texts


def test_log_validation_writes_one_run(tmp_path):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    run_id = log_validation(
        Tracker(uri), _RESULTS, source="mbta", mode="replay", run_id="abc", predictor_ids=["gtfsrt"]
    )
    client, [run] = _runs(uri)
    assert run.info.run_id == run_id and run.info.status == "FINISHED"
    assert run.data.metrics["gtfsrt.mae_s"] == 120.0
    assert [a.path for a in client.list_artifacts(run_id)] == ["report.json"]


def test_nothing_is_logged_without_results_or_tracker(tmp_path):
    assert (
        log_validation(None, _RESULTS, source="mbta", mode="live", run_id="x", predictor_ids=[])
        is None
    )
    tracker = Tracker(f"sqlite:///{tmp_path / 'mlflow.db'}")
    assert (
        log_validation(tracker, [], source="mbta", mode="live", run_id="x", predictor_ids=[])
        is None
    )


async def _finish(manager: RunManager, run_id: str) -> None:
    for _ in range(600):
        if manager.get_run(run_id).status in ("done", "error") and not manager._tracking_tasks:
            return
        await asyncio.sleep(0.05)
    raise AssertionError("run or its tracking did not finish")


async def test_api_runs_are_logged_after_they_finish(tmp_path):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    manager = RunManager(
        tmp_path / "data",
        deps_builder=_replay_deps_builder,
        sources_dir=_SOURCES_DIR,
        tracker=Tracker(uri),
    )
    run_id = await manager.start_run(source="mbta", predictor_ids=[], routes=[], duration_s=60.0)
    await _finish(manager, run_id)
    assert manager.get_run(run_id).status == "done"
    _, [run] = _runs(uri)
    assert run.data.tags["stoptimes.val.run_id"] == run_id
    assert any(k.endswith(".mae_s") for k in run.data.metrics)
    events = [e for e in manager.telemetry.events() if e.get("source") == "tracking"]
    assert events and "logged to MLflow" in events[-1]["message"]


async def test_api_tracking_failure_is_only_a_warning_event(tmp_path):
    tracker = Tracker("http://127.0.0.1:9")

    class _Down:
        def get_experiment_by_name(self, name):
            raise ConnectionError("refused")

    tracker._client_obj = _Down()
    manager = RunManager(
        tmp_path / "data",
        deps_builder=_replay_deps_builder,
        sources_dir=_SOURCES_DIR,
        tracker=tracker,
    )
    run_id = await manager.start_run(source="mbta", predictor_ids=[], routes=[], duration_s=60.0)
    await _finish(manager, run_id)
    assert manager.get_run(run_id).status == "done"
    events = [e for e in manager.telemetry.events() if e.get("source") == "tracking"]
    assert events and events[-1]["level"] == "warn"


def test_tracking_is_off_by_default(tmp_path):
    manager = RunManager(
        tmp_path / "data", deps_builder=_replay_deps_builder, sources_dir=_SOURCES_DIR
    )
    assert manager._tracker is None
